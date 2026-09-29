// =============================================================================
// conv_layer_engine.v
//
// Generic valid/stride-1 convolution engine. Instantiated twice by the top
// level with different parameters (conv1: 1->6ch 5x5, conv2: 6->16ch 3x3).
//
// Dataflow (output-stationary, output-channel parallel):
//   For every output spatial position (oy,ox):
//     For every tap t = 0..(CIN*K*K - 1)  [(cin,ky,kx) unrolled]:
//       - ONE activation value in[cin][oy+ky][ox+kx] is read from the input
//         feature map and BROADCAST to every active PE.
//       - Each active PE p independently reads its OWN weight for output
//         channel (group*NUM_PE + p) at this tap from a per-port weight ROM,
//         and accumulates act * weight into its own accumulator.
//     After the last tap, each active PE holds the full output-channel sum
//     for this (oy,ox). ReLU + requantization is applied per PE and the
//     result is written back to the output feature map -- one channel per
//     cycle, since the output feature map is a single-write-port memory
//     (see README.md Section 12 for why this multiplexed-writeback
//     trade-off was chosen over a wider, multi-port output memory).
//   If COUT > NUM_PE, the whole spatial sweep repeats once per group of
//   NUM_PE output channels (conv2: COUT=16, NUM_PE=8 -> 2 groups).
//
// Because all output channels in a group share the same input activation
// window, only the WEIGHT differs per PE -- exactly the "output-channel
// parallel" convolution mapping described in the architecture writeup.
// =============================================================================

module conv_layer_engine #(
    parameter CIN        = 1,
    parameter COUT       = 6,
    parameter IN_H       = 28,
    parameter IN_W       = 28,
    parameter K          = 5,
    parameter OUT_H      = 24,
    parameter OUT_W      = 24,
    parameter NUM_PE     = 8,
    parameter ACT_W      = 8,
    parameter WGT_W      = 8,
    parameter ACC_W      = 32,
    parameter SHIFT      = 4,
    parameter IN_ADDR_W  = 11,   // must fit CIN*IN_H*IN_W
    parameter OUT_ADDR_W = 12,   // must fit COUT*OUT_H*OUT_W
    parameter WGT_ADDR_W = 10,   // must fit COUT*CIN*K*K per PE port
    parameter BIAS_ADDR_W= 5,    // must fit COUT
    parameter WGT_FILE   = "",
    parameter BIAS_FILE  = ""
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    // input feature-map read port (external single-port RAM, 1-cycle latency)
    output wire [IN_ADDR_W-1:0]        in_addr,
    input  wire signed [ACT_W-1:0]     in_data,

    // output feature-map write port (external single-port RAM)
    output reg                         out_we,
    output reg  [OUT_ADDR_W-1:0]       out_addr,
    output reg  signed [ACT_W-1:0]     out_data,

    // performance counters (pulses, one per event -- aggregated externally)
    // Per-PE-lane event vectors, one bit per lane, straight from the PE
    // array -- NOT OR-reduced. A cycle where e.g. 5 of 8 lanes perform a
    // real MAC must count as 5 MAC operations, not 1; population-counting
    // these vectors (done in cnn_core.v) is what makes that correct.
    output wire [NUM_PE-1:0]           perf_mac_vec,
    output wire [NUM_PE-1:0]           perf_skip_vec
);

    localparam NTAPS      = CIN * K * K;
    localparam NUM_GROUPS = (COUT + NUM_PE - 1) / NUM_PE;

    localparam TAP_BITS = (NTAPS  <= 1) ? 1 : $clog2(NTAPS);
    localparam CIN_BITS = (CIN    <= 1) ? 1 : $clog2(CIN);
    localparam K_BITS   = (K      <= 1) ? 1 : $clog2(K);
    localparam OY_BITS  = (OUT_H  <= 1) ? 1 : $clog2(OUT_H);
    localparam OX_BITS  = (OUT_W  <= 1) ? 1 : $clog2(OUT_W);
    localparam GRP_BITS = (NUM_GROUPS <= 1) ? 1 : $clog2(NUM_GROUPS);
    localparam PE_BITS  = (NUM_PE <= 1) ? 1 : $clog2(NUM_PE);

    reg [GRP_BITS-1:0] group_cnt;
    reg [OY_BITS-1:0]  oy_cnt;
    reg [OX_BITS-1:0]  ox_cnt;
    reg [TAP_BITS-1:0] t_cnt;
    reg [CIN_BITS-1:0] cin_cnt;
    reg [K_BITS-1:0]   ky_cnt;
    reg [K_BITS-1:0]   kx_cnt;
    reg [PE_BITS-1:0]  wpe_cnt;

    localparam S_IDLE  = 2'd0;
    localparam S_MAC   = 2'd1;
    localparam S_WRITE = 2'd2;
    reg [1:0] state;

    wire [PE_BITS:0] active_count =
        (((group_cnt + 1) * NUM_PE) <= COUT) ? NUM_PE[PE_BITS:0]
                                              : (COUT - group_cnt * NUM_PE);

    wire last_kx  = (kx_cnt  == K-1);
    wire last_ky  = (ky_cnt  == K-1);
    wire last_cin = (cin_cnt == CIN-1);
    wire last_tap = (t_cnt   == NTAPS-1);

    // ---------------------------------------------------------------
    // Input activation address (broadcast to every active PE)
    // ---------------------------------------------------------------
    wire [IN_ADDR_W-1:0] in_row = oy_cnt + ky_cnt;
    wire [IN_ADDR_W-1:0] in_col = ox_cnt + kx_cnt;
    assign in_addr = cin_cnt * (IN_H*IN_W) + in_row * IN_W + in_col;

    // ---------------------------------------------------------------
    // Weight ROM: NUM_PE combinational read ports, one per PE lane.
    // Port p reads output channel (group*NUM_PE + p)'s weight for tap t.
    // ---------------------------------------------------------------
    wire [NUM_PE*WGT_ADDR_W-1:0] wgt_addr_flat;
    wire [NUM_PE*WGT_W-1:0]      wgt_data_flat;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : WADDR
            assign wgt_addr_flat[gi*WGT_ADDR_W +: WGT_ADDR_W] =
                (group_cnt * NUM_PE + gi) * NTAPS + t_cnt;
        end
    endgenerate

    weight_rom #(
        .WIDTH(WGT_W), .DEPTH(COUT*NTAPS), .ADDR_W(WGT_ADDR_W),
        .NUM_PORTS(NUM_PE), .INIT_FILE(WGT_FILE)
    ) u_wrom (
        .addr_flat(wgt_addr_flat),
        .data_flat(wgt_data_flat)
    );

    // ---------------------------------------------------------------
    // Bias memory: one combinational-equivalent lookup per PE lane.
    // Small (<=16 entries), so modeled the same way as the weight ROM.
    // ---------------------------------------------------------------
    wire [NUM_PE*BIAS_ADDR_W-1:0] bias_addr_flat;
    wire [NUM_PE*ACC_W-1:0]       bias_data_flat;

    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : BADDR
            assign bias_addr_flat[gi*BIAS_ADDR_W +: BIAS_ADDR_W] = group_cnt * NUM_PE + gi;
        end
    endgenerate

    weight_rom #(
        .WIDTH(ACC_W), .DEPTH(COUT), .ADDR_W(BIAS_ADDR_W),
        .NUM_PORTS(NUM_PE), .INIT_FILE(BIAS_FILE)
    ) u_brom (
        .addr_flat(bias_addr_flat),
        .data_flat(bias_data_flat)
    );

    // ---------------------------------------------------------------
    // PE array
    // ---------------------------------------------------------------
    wire mac_running = (state == S_MAC);

    wire [NUM_PE-1:0] en_vec;
    wire [NUM_PE-1:0] clr_vec;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : ENVEC
            assign en_vec[gi]  = mac_running && (gi < active_count);
            assign clr_vec[gi] = en_vec[gi] && (t_cnt == 0);
        end
    endgenerate

    wire signed [NUM_PE*ACT_W-1:0] act_in_flat;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : ACTBCAST
            assign act_in_flat[gi*ACT_W +: ACT_W] = in_data;
        end
    endgenerate

    wire signed [NUM_PE*ACC_W-1:0] acc_out_flat;
    wire [NUM_PE-1:0] mac_skipped_vec;
    wire [NUM_PE-1:0] mac_valid_vec;

    pe_array #(
        .NUM_PE(NUM_PE), .ACT_W(ACT_W), .WGT_W(WGT_W), .ACC_W(ACC_W)
    ) u_pes (
        .clk(clk), .rst_n(rst_n),
        .en(en_vec), .acc_clear(clr_vec),
        .act_in_flat(act_in_flat),
        .wgt_in_flat(wgt_data_flat),
        .acc_out_flat(acc_out_flat),
        .mac_skipped(mac_skipped_vec),
        .mac_valid(mac_valid_vec)
    );

    assign perf_mac_vec  = mac_valid_vec;
    assign perf_skip_vec = mac_skipped_vec;

    // ---------------------------------------------------------------
    // Per-PE ReLU + requantization, muxed out by wpe_cnt during S_WRITE
    // ---------------------------------------------------------------
    wire signed [ACC_W-1:0] sel_acc  = acc_out_flat [wpe_cnt*ACC_W  +: ACC_W];
    wire signed [ACC_W-1:0] sel_bias = bias_data_flat[wpe_cnt*ACC_W +: ACC_W];
    wire signed [ACT_W-1:0] sel_relu;

    relu_quant #(.ACC_W(ACC_W), .OUT_W(ACT_W), .SHIFT(SHIFT)) u_rq (
        .acc_in(sel_acc), .bias_in(sel_bias), .act_out(sel_relu)
    );

    // ---------------------------------------------------------------
    // Control FSM
    // ---------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            out_we    <= 1'b0;
            group_cnt <= 0;
            oy_cnt    <= 0;
            ox_cnt    <= 0;
            t_cnt     <= 0;
            cin_cnt   <= 0;
            ky_cnt    <= 0;
            kx_cnt    <= 0;
            wpe_cnt   <= 0;
        end else begin
            out_we <= 1'b0; // default: pulse
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        group_cnt <= 0;
                        oy_cnt    <= 0;
                        ox_cnt    <= 0;
                        t_cnt     <= 0;
                        cin_cnt   <= 0;
                        ky_cnt    <= 0;
                        kx_cnt    <= 0;
                        state     <= S_MAC;
                    end
                end

                S_MAC: begin
                    if (last_tap) begin
                        wpe_cnt <= 0;
                        state   <= S_WRITE;
                    end else begin
                        t_cnt <= t_cnt + 1'b1;
                        if (last_kx) begin
                            kx_cnt <= 0;
                            if (last_ky) begin
                                ky_cnt  <= 0;
                                cin_cnt <= cin_cnt + 1'b1;
                            end else begin
                                ky_cnt <= ky_cnt + 1'b1;
                            end
                        end else begin
                            kx_cnt <= kx_cnt + 1'b1;
                        end
                    end
                end

                S_WRITE: begin
                    out_we   <= 1'b1;
                    out_addr <= (group_cnt * NUM_PE + wpe_cnt) * (OUT_H*OUT_W)
                                + oy_cnt * OUT_W + ox_cnt;
                    out_data <= sel_relu;

                    if (wpe_cnt == active_count - 1) begin
                        // finished writing every channel of this group for (oy,ox)
                        t_cnt   <= 0;
                        cin_cnt <= 0;
                        ky_cnt  <= 0;
                        kx_cnt  <= 0;

                        if (ox_cnt == OUT_W-1) begin
                            ox_cnt <= 0;
                            if (oy_cnt == OUT_H-1) begin
                                oy_cnt <= 0;
                                if (group_cnt == NUM_GROUPS-1) begin
                                    busy  <= 1'b0;
                                    done  <= 1'b1;
                                    state <= S_IDLE;
                                end else begin
                                    group_cnt <= group_cnt + 1'b1;
                                    state     <= S_MAC;
                                end
                            end else begin
                                oy_cnt <= oy_cnt + 1'b1;
                                state  <= S_MAC;
                            end
                        end else begin
                            ox_cnt <= ox_cnt + 1'b1;
                            state  <= S_MAC;
                        end
                    end else begin
                        wpe_cnt <= wpe_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

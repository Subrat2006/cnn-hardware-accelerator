// =============================================================================
// fc_layer_engine.v
//
// Generic fully-connected layer engine, reused for FC1 (400->120) and
// FC2 (120->10). Deliberately built on the SAME per-PE MAC/accumulate style
// as conv_layer_engine.v -- this is the "shared PE-array across conv and FC"
// design decision: rather than a bespoke systolic array for FC, the same
// broadcast-activation / per-PE-weight structure is reused, just without the
// 2D spatial (oy,ox) loop that convolution needs.
//
// For output neuron group g (NUM_PE neurons per group):
//   for t = 0 .. N_IN-1:
//     - one activation x[t] is broadcast to every active PE
//     - PE p reads its own weight w[(g*NUM_PE+p)][t] and accumulates
//   after the last t, each PE holds y_(g*NUM_PE+p) = sum_t x[t]*w[..][t].
//
// Two outputs are produced per finished neuron:
//   - `out_data` / `out_we` / `out_addr` : ReLU + INT8 requantized activation
//      (used by FC1, which feeds FC2)
//   - `raw_data` / `raw_we` / `raw_addr` : raw (accumulator + bias) INT32
//      score, no activation (used by FC2, the final classifier layer, whose
//      output is read back directly through AXI as the classification score)
// An instantiation only needs to wire up whichever pair its downstream
// consumer needs.
// =============================================================================

module fc_layer_engine #(
    parameter N_IN        = 400,
    parameter N_OUT        = 120,
    parameter NUM_PE       = 8,
    parameter ACT_W        = 8,
    parameter WGT_W        = 8,
    parameter ACC_W        = 32,
    parameter SHIFT        = 6,
    parameter IN_ADDR_W    = 9,    // must fit N_IN
    parameter OUT_ADDR_W   = 7,    // must fit N_OUT
    parameter WGT_ADDR_W   = 16,   // must fit N_OUT*N_IN per PE port
    parameter BIAS_ADDR_W  = 7,    // must fit N_OUT
    parameter WGT_FILE     = "",
    parameter BIAS_FILE    = ""
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    // input activation vector read port (external single-port RAM)
    output wire [IN_ADDR_W-1:0]        in_addr,
    input  wire signed [ACT_W-1:0]     in_data,

    // requantized INT8 activation writeback (feeds next FC layer)
    output reg                         out_we,
    output reg  [OUT_ADDR_W-1:0]       out_addr,
    output reg  signed [ACT_W-1:0]     out_data,

    // raw INT32 (acc+bias) score writeback (final classifier layer)
    output reg                         raw_we,
    output reg  [OUT_ADDR_W-1:0]       raw_addr,
    output reg  signed [ACC_W-1:0]     raw_data,

    // Per-PE-lane event vectors (not OR-reduced) -- see conv_layer_engine.v
    // for why population-counting these (in cnn_core.v) matters.
    output wire [NUM_PE-1:0]           perf_mac_vec,
    output wire [NUM_PE-1:0]           perf_skip_vec
);

    localparam NUM_GROUPS = (N_OUT + NUM_PE - 1) / NUM_PE;

    localparam TAP_BITS = (N_IN <= 1) ? 1 : $clog2(N_IN);
    localparam GRP_BITS = (NUM_GROUPS <= 1) ? 1 : $clog2(NUM_GROUPS);
    localparam PE_BITS  = (NUM_PE <= 1) ? 1 : $clog2(NUM_PE);

    reg [GRP_BITS-1:0] group_cnt;
    reg [TAP_BITS-1:0] t_cnt;
    reg [PE_BITS-1:0]  wpe_cnt;

    localparam S_IDLE  = 2'd0;
    localparam S_MAC   = 2'd1;
    localparam S_WRITE = 2'd2;
    reg [1:0] state;

    wire [PE_BITS:0] active_count =
        (((group_cnt + 1) * NUM_PE) <= N_OUT) ? NUM_PE[PE_BITS:0]
                                               : (N_OUT - group_cnt * NUM_PE);

    wire last_tap = (t_cnt == N_IN-1);

    // input activation address: same feature index broadcast to every PE
    assign in_addr = t_cnt;

    // weight ROM: NUM_PE ports
    wire [NUM_PE*WGT_ADDR_W-1:0] wgt_addr_flat;
    wire [NUM_PE*WGT_W-1:0]      wgt_data_flat;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : WADDR
            assign wgt_addr_flat[gi*WGT_ADDR_W +: WGT_ADDR_W] =
                (group_cnt * NUM_PE + gi) * N_IN + t_cnt;
        end
    endgenerate

    weight_rom #(
        .WIDTH(WGT_W), .DEPTH(N_OUT*N_IN), .ADDR_W(WGT_ADDR_W),
        .NUM_PORTS(NUM_PE), .INIT_FILE(WGT_FILE)
    ) u_wrom (
        .addr_flat(wgt_addr_flat),
        .data_flat(wgt_data_flat)
    );

    // bias lookup: NUM_PE ports
    wire [NUM_PE*BIAS_ADDR_W-1:0] bias_addr_flat;
    wire [NUM_PE*ACC_W-1:0]       bias_data_flat;

    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : BADDR
            assign bias_addr_flat[gi*BIAS_ADDR_W +: BIAS_ADDR_W] = group_cnt * NUM_PE + gi;
        end
    endgenerate

    weight_rom #(
        .WIDTH(ACC_W), .DEPTH(N_OUT), .ADDR_W(BIAS_ADDR_W),
        .NUM_PORTS(NUM_PE), .INIT_FILE(BIAS_FILE)
    ) u_brom (
        .addr_flat(bias_addr_flat),
        .data_flat(bias_data_flat)
    );

    // PE array
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

    wire signed [ACC_W-1:0] sel_acc  = acc_out_flat [wpe_cnt*ACC_W  +: ACC_W];
    wire signed [ACC_W-1:0] sel_bias = bias_data_flat[wpe_cnt*ACC_W +: ACC_W];
    wire signed [ACC_W-1:0] sel_sum  = sel_acc + sel_bias;
    wire signed [ACT_W-1:0] sel_relu;

    relu_quant #(.ACC_W(ACC_W), .OUT_W(ACT_W), .SHIFT(SHIFT)) u_rq (
        .acc_in(sel_acc), .bias_in(sel_bias), .act_out(sel_relu)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            out_we    <= 1'b0;
            raw_we    <= 1'b0;
            group_cnt <= 0;
            t_cnt     <= 0;
            wpe_cnt   <= 0;
        end else begin
            out_we <= 1'b0; // default: pulse
            raw_we <= 1'b0;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        group_cnt <= 0;
                        t_cnt     <= 0;
                        state     <= S_MAC;
                    end
                end

                S_MAC: begin
                    if (last_tap) begin
                        wpe_cnt <= 0;
                        state   <= S_WRITE;
                    end else begin
                        t_cnt <= t_cnt + 1'b1;
                    end
                end

                S_WRITE: begin
                    out_we   <= 1'b1;
                    out_addr <= group_cnt * NUM_PE + wpe_cnt;
                    out_data <= sel_relu;

                    raw_we   <= 1'b1;
                    raw_addr <= group_cnt * NUM_PE + wpe_cnt;
                    raw_data <= sel_sum;

                    if (wpe_cnt == active_count - 1) begin
                        t_cnt <= 0;
                        if (group_cnt == NUM_GROUPS-1) begin
                            busy  <= 1'b0;
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            group_cnt <= group_cnt + 1'b1;
                            state     <= S_MAC;
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

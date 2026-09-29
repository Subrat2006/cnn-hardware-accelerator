// =============================================================================
// pool_engine.v
//
// Generic sequential 2x2 / stride-2 max-pooling controller. Instantiated
// twice by the top level (pool1: 6ch 24x24->12x12, pool2: 16ch 10x10->5x5)
// with different parameters. Walks every (channel, output row, output col)
// window, fetches its four inputs from a single-read-port synchronous
// memory (one value per cycle, since the feature-map RAM has only one read
// port), and writes the max to the output feature map.
//
// This is a correctness-first, non-pipelined implementation: 5 cycles per
// output pixel (4 to fetch inputs + 1 to compute/write). Pipelining across
// window boundaries is a documented future optimization (see
// README.md Section 12) -- pooling is a tiny fraction of
// total accelerator runtime compared to the convolution and FC engines, so
// it was not worth the added control complexity within the hackathon
// timeline.
// =============================================================================

module pool_engine #(
    parameter CH         = 6,
    parameter IN_H       = 24,
    parameter IN_W       = 24,
    parameter OUT_H      = 12,
    parameter OUT_W      = 12,
    parameter DATA_W     = 8,
    parameter IN_ADDR_W  = 10,   // must fit CH*IN_H*IN_W addresses
    parameter OUT_ADDR_W = 10    // must fit CH*OUT_H*OUT_W addresses
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    // input feature-map read port (external single-port RAM, 1-cycle latency)
    output wire [IN_ADDR_W-1:0]        in_addr,
    input  wire signed [DATA_W-1:0]    in_data,

    // output feature-map write port (external single-port RAM)
    output reg                         out_we,
    output reg  [OUT_ADDR_W-1:0]       out_addr,
    output reg  signed [DATA_W-1:0]    out_data
);

    localparam IN_HW  = IN_H * IN_W;
    localparam OUT_HW = OUT_H * OUT_W;

    localparam CH_BITS = (CH    <= 1) ? 1 : $clog2(CH);
    localparam OY_BITS = (OUT_H <= 1) ? 1 : $clog2(OUT_H);
    localparam OX_BITS = (OUT_W <= 1) ? 1 : $clog2(OUT_W);

    reg [CH_BITS-1:0] ch_cnt;
    reg [OY_BITS-1:0] oy_cnt;
    reg [OX_BITS-1:0] ox_cnt;
    reg [2:0]         phase; // 0..4

    reg signed [DATA_W-1:0] v0, v1, v2;

    localparam S_IDLE = 1'b0;
    localparam S_RUN  = 1'b1;
    reg state;

    // Combinational addresses for the *current* (ch_cnt,oy_cnt,ox_cnt) window.
    wire [IN_ADDR_W-1:0] base  = ch_cnt * IN_HW;
    wire [IN_ADDR_W-1:0] row0  = (oy_cnt << 1);
    wire [IN_ADDR_W-1:0] col0  = (ox_cnt << 1);
    wire [IN_ADDR_W-1:0] addr0 = base + row0 * IN_W + col0;
    wire [IN_ADDR_W-1:0] addr1 = addr0 + 1'b1;
    wire [IN_ADDR_W-1:0] addr2 = addr0 + IN_W;
    wire [IN_ADDR_W-1:0] addr3 = addr2 + 1'b1;

    // Address presented to the external RAM THIS cycle, purely combinational
    // on `phase` -- no one-cycle-ahead pipelining, so it always tracks the
    // window currently pointed to by ch_cnt/oy_cnt/ox_cnt (updated at phase 4).
    assign in_addr = (phase == 3'd0) ? addr0 :
                      (phase == 3'd1) ? addr1 :
                      (phase == 3'd2) ? addr2 :
                      addr3;

    wire signed [DATA_W-1:0] max_val;
    max_pool2x2 #(.WIDTH(DATA_W)) u_max (
        .a(v0), .b(v1), .c(v2), .d(in_data),
        .max_out(max_val)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            busy   <= 1'b0;
            done   <= 1'b0;
            out_we <= 1'b0;
            ch_cnt <= 0;
            oy_cnt <= 0;
            ox_cnt <= 0;
            phase  <= 0;
        end else begin
            out_we <= 1'b0; // default: single-cycle pulse
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy   <= 1'b1;
                        ch_cnt <= 0;
                        oy_cnt <= 0;
                        ox_cnt <= 0;
                        phase  <= 0;
                        state  <= S_RUN;
                    end
                end

                S_RUN: begin
                    case (phase)
                        // phase0: in_addr==addr0 this cycle -> next edge, dout=mem[addr0]
                        0: phase <= 1;
                        // phase1: dout==mem[addr0] (=v0); in_addr==addr1 this cycle
                        1: begin
                            v0    <= in_data;
                            phase <= 2;
                        end
                        // phase2: dout==mem[addr1] (=v1); in_addr==addr2 this cycle
                        2: begin
                            v1    <= in_data;
                            phase <= 3;
                        end
                        // phase3: dout==mem[addr2] (=v2); in_addr==addr3 this cycle
                        3: begin
                            v2    <= in_data;
                            phase <= 4;
                        end
                        // phase4: dout==mem[addr3]; compute max(v0,v1,v2,in_data) and write
                        4: begin
                            out_we   <= 1'b1;
                            out_addr <= ch_cnt * OUT_HW + oy_cnt * OUT_W + ox_cnt;
                            out_data <= max_val;

                            if (ox_cnt == OUT_W-1) begin
                                ox_cnt <= 0;
                                if (oy_cnt == OUT_H-1) begin
                                    oy_cnt <= 0;
                                    if (ch_cnt == CH-1) begin
                                        busy  <= 1'b0;
                                        done  <= 1'b1;
                                        state <= S_IDLE;
                                    end else begin
                                        ch_cnt <= ch_cnt + 1'b1;
                                    end
                                end else begin
                                    oy_cnt <= oy_cnt + 1'b1;
                                end
                            end else begin
                                ox_cnt <= ox_cnt + 1'b1;
                            end

                            phase <= 0;
                        end
                        default: phase <= 0;
                    endcase
                end
            endcase
        end
    end

endmodule

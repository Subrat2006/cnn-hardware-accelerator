// =============================================================================
// axi4_lite_slave.v
//
// AXI4-Lite slave register interface, running entirely in the AXI clock
// domain (axi_aclk). This is the only block in the design that talks to the
// external processor/test-system:
//   - REG_CTRL        (W)  : bit0 = start (pulse). Bits [31:1] reserved
//                             (an earlier draft used bit1 for a "soft
//                             reset" that only rewound the top-level
//                             sequencer FSM without resetting the child
//                             layer engines, performance counters, or
//                             FIFO state -- an incomplete/incorrect
//                             feature, so it was removed rather than kept
//                             half-implemented; see README Section 9)
//   - REG_STATUS      (R)  : bit0 = busy, bit1 = done, bit2 = input_loaded
//                             (all synchronized in from the core clock
//                             domain by the caller -- see top level)
//   - REG_INPUT_DATA  (W)  : push next input-image byte (lower 8 bits of
//                             wdata) into the input stream; produces a
//                             single-cycle input_wr_valid/input_wr_data pair
//                             that the top level feeds straight into
//                             async_fifo's write port (same clock domain,
//                             no extra CDC needed here)
//   - REG_PERF_*      (R)  : cycle / MAC / skipped-MAC counters
//   - REG_OUT_BASE+4i (R)  : classification score for class i (i=0..9)
//
// Compliance, stated precisely: this implements the AXI4-Lite register
// interface for the targeted single-master access pattern; AWVALID and
// WVALID are intentionally coupled (expected together, one register access
// at a time -- matching the accompanying testbench's BFM and typical
// bare-metal register-poking software) to reduce control complexity, rather
// than implementing the full independent-channel-skew AXI4 write-ordering
// rules a general-purpose slave would need. This is NOT "a fully
// AXI4-Lite-compliant slave" and should not be described as one; say
// instead that it implements the register interface for this access
// pattern, and that a production IP would support independent channel
// arrival. WSTRB is similarly accepted but not honored (every write is
// treated as a full 32-bit write) -- fine for this design's write-only,
// word-sized registers, but worth naming explicitly rather than leaving
// implicit.
// =============================================================================

module axi4_lite_slave #(
    parameter ADDR_W = 8,
    parameter DATA_W = 32
)(
    input  wire                      axi_aclk,
    input  wire                      axi_aresetn,

    // -------- AXI4-Lite write address/data/response --------
    input  wire [ADDR_W-1:0]         s_axi_awaddr,
    input  wire                      s_axi_awvalid,
    output reg                       s_axi_awready,

    input  wire [DATA_W-1:0]         s_axi_wdata,
    input  wire [DATA_W/8-1:0]       s_axi_wstrb,
    input  wire                      s_axi_wvalid,
    output reg                       s_axi_wready,

    output reg  [1:0]                s_axi_bresp,
    output reg                       s_axi_bvalid,
    input  wire                      s_axi_bready,

    // -------- AXI4-Lite read address/data --------
    input  wire [ADDR_W-1:0]         s_axi_araddr,
    input  wire                      s_axi_arvalid,
    output reg                       s_axi_arready,

    output reg  [DATA_W-1:0]         s_axi_rdata,
    output reg  [1:0]                s_axi_rresp,
    output reg                       s_axi_rvalid,
    input  wire                      s_axi_rready,

    // -------- to core (control strobe, axi_aclk domain) --------
    output reg                       ctrl_start_pulse,

    output reg                       input_wr_valid,
    output reg  [7:0]                input_wr_data,

    // -------- from core (already synchronized into axi_aclk domain) --------
    input  wire                      busy_sync,
    input  wire                      done_sync,
    input  wire                      input_loaded_sync,
    input  wire [31:0]               perf_cycles,
    input  wire [31:0]               perf_macs,
    input  wire [31:0]               perf_skip,
    input  wire [31:0]               out_score_0,
    input  wire [31:0]               out_score_1,
    input  wire [31:0]               out_score_2,
    input  wire [31:0]               out_score_3,
    input  wire [31:0]               out_score_4,
    input  wire [31:0]               out_score_5,
    input  wire [31:0]               out_score_6,
    input  wire [31:0]               out_score_7,
    input  wire [31:0]               out_score_8,
    input  wire [31:0]               out_score_9
);

`include "cnn_pkg.vh"

    // ---------------- write channel FSM ----------------
    localparam W_IDLE = 1'b0, W_RESP = 1'b1;
    reg w_state;

    wire wr_fire = s_axi_awvalid && s_axi_wvalid && (w_state == W_IDLE);

    always @(posedge axi_aclk or negedge axi_aresetn) begin
        if (!axi_aresetn) begin
            w_state             <= W_IDLE;
            s_axi_awready       <= 1'b0;
            s_axi_wready        <= 1'b0;
            s_axi_bvalid        <= 1'b0;
            s_axi_bresp         <= 2'b00;
            ctrl_start_pulse    <= 1'b0;
            input_wr_valid      <= 1'b0;
            input_wr_data       <= 8'h00;
        end else begin
            ctrl_start_pulse    <= 1'b0;
            input_wr_valid      <= 1'b0;
            s_axi_awready       <= 1'b0;
            s_axi_wready        <= 1'b0;

            case (w_state)
                W_IDLE: begin
                    if (wr_fire) begin
                        s_axi_awready <= 1'b1;
                        s_axi_wready  <= 1'b1;

                        case (s_axi_awaddr[7:0])
                            `REG_CTRL: begin
                                ctrl_start_pulse    <= s_axi_wdata[0];
                            end
                            `REG_INPUT_DATA: begin
                                input_wr_valid <= 1'b1;
                                input_wr_data  <= s_axi_wdata[7:0];
                            end
                            default: ; // writes to read-only/unmapped regs are ignored
                        endcase

                        s_axi_bresp  <= 2'b00; // OKAY
                        s_axi_bvalid <= 1'b1;
                        w_state      <= W_RESP;
                    end
                end

                W_RESP: begin
                    if (s_axi_bvalid && s_axi_bready) begin
                        s_axi_bvalid <= 1'b0;
                        w_state      <= W_IDLE;
                    end
                end
            endcase
        end
    end

    // ---------------- read channel FSM ----------------
    localparam R_IDLE = 1'b0, R_DATA = 1'b1;
    reg r_state;

    always @(posedge axi_aclk or negedge axi_aresetn) begin
        if (!axi_aresetn) begin
            r_state       <= R_IDLE;
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'h0;
        end else begin
            s_axi_arready <= 1'b0;

            case (r_state)
                R_IDLE: begin
                    if (s_axi_arvalid) begin
                        s_axi_arready <= 1'b1;
                        s_axi_rresp   <= 2'b00; // OKAY

                        case (s_axi_araddr[7:0])
                            `REG_STATUS:
                                s_axi_rdata <= {29'b0, input_loaded_sync, done_sync, busy_sync};
                            `REG_PERF_CYCLES: s_axi_rdata <= perf_cycles;
                            `REG_PERF_MACS:   s_axi_rdata <= perf_macs;
                            `REG_PERF_SKIP:   s_axi_rdata <= perf_skip;
                            `REG_OUT_BASE + 8'd0:  s_axi_rdata <= out_score_0;
                            `REG_OUT_BASE + 8'd4:  s_axi_rdata <= out_score_1;
                            `REG_OUT_BASE + 8'd8:  s_axi_rdata <= out_score_2;
                            `REG_OUT_BASE + 8'd12: s_axi_rdata <= out_score_3;
                            `REG_OUT_BASE + 8'd16: s_axi_rdata <= out_score_4;
                            `REG_OUT_BASE + 8'd20: s_axi_rdata <= out_score_5;
                            `REG_OUT_BASE + 8'd24: s_axi_rdata <= out_score_6;
                            `REG_OUT_BASE + 8'd28: s_axi_rdata <= out_score_7;
                            `REG_OUT_BASE + 8'd32: s_axi_rdata <= out_score_8;
                            `REG_OUT_BASE + 8'd36: s_axi_rdata <= out_score_9;
                            default: s_axi_rdata <= 32'hDEAD_BEEF; // unmapped address
                        endcase

                        s_axi_rvalid <= 1'b1;
                        r_state      <= R_DATA;
                    end
                end

                R_DATA: begin
                    if (s_axi_rvalid && s_axi_rready) begin
                        s_axi_rvalid <= 1'b0;
                        r_state      <= R_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

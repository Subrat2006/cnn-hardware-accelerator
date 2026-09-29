// =============================================================================
// cnn_top.v
//
// Top-level accelerator subsystem:
//
//     AXI4-Lite (axi_aclk)  <--CDC-->  Accelerator core (core_clk)
//
// This is the IP block described by the problem statement: an AXI-attached
// CNN accelerator whose AXI interface and compute core run on independent
// clocks, connected only through the CDC primitives below -- never directly.
//
// CDC inventory (three different techniques, each chosen for what it moves):
//   1. start : a single-cycle pulse         -> cdc_pulse_sync.v
//        (toggle + double-sync + edge-detect; safe for any clock ratio)
//   2. busy / done / input_loaded : slow-changing level status bits
//                                    -> cdc_sync_2ff.v (plain double-flop)
//   3. input image bytes : a genuine multi-bit streaming data path
//                                    -> async_fifo.v (gray-code pointers)
//   4. perf counters / classification scores : held stable once the core
//      asserts `done`. `done` itself is transferred through the 2FF
//      synchronizer in #2, and software is only expected to read these
//      buses after observing the synchronized `done` flag -- a
//      bundled-data / stable-before-enable protocol. No separate
//      synchronizer is layered on top of the counters/scores themselves:
//      a 2FF synchronizer on a many-bit bus that changes several bits per
//      cycle would NOT safely synchronize it (different bits could
//      resolve metastability on different cycles, corrupting the value),
//      so the safety here comes specifically from the data having
//      already gone quiet before the flag is observed, not from the bus
//      being "static" in some general sense.
// =============================================================================

module cnn_top #(
    parameter AXI_ADDR_W = 8,
    parameter AXI_DATA_W = 32,
    parameter FIFO_ADDR_W = 10   // input FIFO depth = 1024, comfortably >= 784 pixels
)(
    // ---------------- AXI-Lite clock domain ----------------
    input  wire                     axi_aclk,
    input  wire                     axi_aresetn,

    input  wire [AXI_ADDR_W-1:0]    s_axi_awaddr,
    input  wire                     s_axi_awvalid,
    output wire                     s_axi_awready,

    input  wire [AXI_DATA_W-1:0]    s_axi_wdata,
    input  wire [AXI_DATA_W/8-1:0]  s_axi_wstrb,
    input  wire                     s_axi_wvalid,
    output wire                     s_axi_wready,

    output wire [1:0]               s_axi_bresp,
    output wire                     s_axi_bvalid,
    input  wire                     s_axi_bready,

    input  wire [AXI_ADDR_W-1:0]    s_axi_araddr,
    input  wire                     s_axi_arvalid,
    output wire                     s_axi_arready,

    output wire [AXI_DATA_W-1:0]    s_axi_rdata,
    output wire [1:0]               s_axi_rresp,
    output wire                     s_axi_rvalid,
    input  wire                     s_axi_rready,

    // ---------------- accelerator core clock domain ----------------
    input  wire                     core_clk,
    input  wire                     core_rst_n
);

    // ------------------------------------------------------------------
    // AXI-domain control strobe -> core domain (pulse CDC)
    // ------------------------------------------------------------------
    wire ctrl_start_pulse_axi;
    wire start_pulse_core;

    cdc_pulse_sync u_start_cdc (
        .src_clk(axi_aclk), .src_rst_n(axi_aresetn), .pulse_in(ctrl_start_pulse_axi),
        .dst_clk(core_clk), .dst_rst_n(core_rst_n), .pulse_out(start_pulse_core)
    );

    // ------------------------------------------------------------------
    // Input pixel stream: AXI writes -> async FIFO (wr side = axi_aclk)
    //                                -> core reads   (rd side = core_clk)
    // ------------------------------------------------------------------
    wire        input_wr_valid;
    wire [7:0]  input_wr_data;
    wire        fifo_full;

    wire        fifo_rd_en;
    wire [7:0]  fifo_rd_data;
    wire        fifo_empty;

    async_fifo #(.WIDTH(8), .ADDR_W(FIFO_ADDR_W)) u_input_fifo (
        .wr_clk(axi_aclk), .wr_rst_n(axi_aresetn),
        .wr_en(input_wr_valid), .wr_data(input_wr_data), .full(fifo_full),
        .rd_clk(core_clk), .rd_rst_n(core_rst_n),
        .rd_en(fifo_rd_en), .rd_data(fifo_rd_data), .empty(fifo_empty)
    );
    // NOTE: fifo_full is intentionally unused by the AXI slave: the FIFO is
    // sized (2**FIFO_ADDR_W = 1024) to comfortably hold an entire 28x28=784
    // pixel frame, so for this fixed, known-size, single-frame-at-a-time
    // accelerator, back-pressuring individual AXI writes was judged not
    // worth the extra protocol complexity. A generic streaming accelerator
    // would instead expose fifo_full and make the AXI write wait / return
    // SLVERR on overflow -- called out here as a scoped-out extension.

    // ------------------------------------------------------------------
    // Core status -> AXI domain (2FF level sync)
    // ------------------------------------------------------------------
    wire busy_core, done_core, input_loaded_core;
    wire busy_axi, done_axi, input_loaded_axi;

    cdc_sync_2ff #(.WIDTH(3)) u_status_cdc (
        .dst_clk(axi_aclk), .dst_rst_n(axi_aresetn),
        .async_in ({input_loaded_core, done_core, busy_core}),
        .sync_out ({input_loaded_axi, done_axi, busy_axi})
    );

    // ------------------------------------------------------------------
    // Perf counters and classification scores: stable-then-flagged buses,
    // read directly (see file header note #4)
    // ------------------------------------------------------------------
    wire [31:0] perf_cycles_core, perf_macs_core, perf_skip_core;
    wire [31:0] score0,score1,score2,score3,score4,score5,score6,score7,score8,score9;

    // ------------------------------------------------------------------
    // Accelerator core
    // ------------------------------------------------------------------
    cnn_core u_core (
        .core_clk(core_clk), .core_rst_n(core_rst_n),
        .start_pulse(start_pulse_core),
        .fifo_rd_en(fifo_rd_en), .fifo_rd_data(fifo_rd_data), .fifo_empty(fifo_empty),
        .busy(busy_core), .done(done_core), .input_loaded(input_loaded_core),
        .perf_cycles(perf_cycles_core), .perf_macs(perf_macs_core), .perf_skip(perf_skip_core),
        .out_score_0(score0), .out_score_1(score1), .out_score_2(score2), .out_score_3(score3),
        .out_score_4(score4), .out_score_5(score5), .out_score_6(score6), .out_score_7(score7),
        .out_score_8(score8), .out_score_9(score9)
    );

    // ------------------------------------------------------------------
    // AXI4-Lite slave (entirely axi_aclk domain)
    // ------------------------------------------------------------------
    axi4_lite_slave #(.ADDR_W(AXI_ADDR_W), .DATA_W(AXI_DATA_W)) u_axi (
        .axi_aclk(axi_aclk), .axi_aresetn(axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .ctrl_start_pulse(ctrl_start_pulse_axi),
        .input_wr_valid(input_wr_valid), .input_wr_data(input_wr_data),
        .busy_sync(busy_axi), .done_sync(done_axi), .input_loaded_sync(input_loaded_axi),
        .perf_cycles(perf_cycles_core), .perf_macs(perf_macs_core), .perf_skip(perf_skip_core),
        .out_score_0(score0), .out_score_1(score1), .out_score_2(score2), .out_score_3(score3), .out_score_4(score4),
        .out_score_5(score5), .out_score_6(score6), .out_score_7(score7), .out_score_8(score8), .out_score_9(score9)
    );

endmodule

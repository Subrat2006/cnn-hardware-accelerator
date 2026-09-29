// =============================================================================
// cnn_core.v
//
// The accelerator core -- everything that runs in the "core_clk" domain,
// separate from the AXI-Lite clock ("axi_aclk"). This is the block whose
// clock the problem statement calls out as running at a different frequency
// from the AXI interface, motivating the CDC in cnn_top.v.
//
// Structure: one dedicated pair of memories between every two pipeline
// stages (img_mem -> conv1 -> conv1_out_mem -> pool1 -> pool1_out_mem ->
// conv2 -> conv2_out_mem -> pool2 -> pool2_out_mem(=fc1 input) -> fc1 ->
// fc1_out_mem -> fc2 -> score registers), sequenced by a single top-level
// FSM that starts each engine and waits for its `done` pulse before moving
// on. Because exactly one engine ever touches a given memory's address port
// at a time, each shared port is a simple mux keyed off which stage is
// currently active.
// =============================================================================

`include "cnn_pkg.vh"

module cnn_core (
    input  wire        core_clk,
    input  wire        core_rst_n,

    input  wire        start_pulse,     // 1-cycle, already synchronized into core_clk

    // input-image stream (read side of the async FIFO, core_clk domain)
    output wire         fifo_rd_en,
    input  wire [7:0]   fifo_rd_data,
    input  wire         fifo_empty,

    output wire         busy,
    output wire         done,
    output wire         input_loaded,

    output wire [31:0]  perf_cycles,
    output wire [31:0]  perf_macs,
    output wire [31:0]  perf_skip,

    output wire [31:0]  out_score_0, out_score_1, out_score_2, out_score_3, out_score_4,
    output wire [31:0]  out_score_5, out_score_6, out_score_7, out_score_8, out_score_9
);

    // -------------------------------------------------------------------
    // Top-level sequencer
    // -------------------------------------------------------------------
    localparam SEQ_IDLE  = 4'd0;
    localparam SEQ_LOAD  = 4'd1;
    localparam SEQ_CONV1 = 4'd2;
    localparam SEQ_POOL1 = 4'd3;
    localparam SEQ_CONV2 = 4'd4;
    localparam SEQ_POOL2 = 4'd5;
    localparam SEQ_FC1   = 4'd6;
    localparam SEQ_FC2   = 4'd7;
    localparam SEQ_DONE  = 4'd8;

    reg [3:0] seq_state;

    assign busy = (seq_state != SEQ_IDLE) && (seq_state != SEQ_DONE);
    assign done = (seq_state == SEQ_DONE);

    // -------------------------------------------------------------------
    // Input-image loader: drains the async FIFO into img_mem while in
    // SEQ_LOAD. FIFO read has a 1-cycle latency, so `fifo_rd_en` is
    // delayed by one cycle to align the write-enable/address with the
    // data actually becoming valid.
    // -------------------------------------------------------------------
    localparam IMG_DEPTH = `IMG_H * `IMG_W; // 784
    reg [9:0] img_load_cnt;
    reg       fifo_rd_en_d;
    reg [9:0] img_load_addr_d;

    assign fifo_rd_en   = !fifo_empty && (img_load_cnt < IMG_DEPTH) && (seq_state == SEQ_LOAD);
    assign input_loaded = (img_load_cnt == IMG_DEPTH);

    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            img_load_cnt    <= 10'd0;
            fifo_rd_en_d    <= 1'b0;
            img_load_addr_d <= 10'd0;
        end else if (start_pulse) begin
            // Fresh run: reset the loader unconditionally on every START,
            // not just out of SEQ_IDLE. FIX (previously a real bug): the
            // sequencer's SEQ_DONE -> SEQ_LOAD transition (a repeated,
            // back-to-back inference) never passes through SEQ_IDLE, so
            // gating this reset on "seq_state == SEQ_IDLE" left
            // img_load_cnt stuck at 784 after the first run and silently
            // skipped loading the second image. Gating on start_pulse
            // itself covers both the first run (start_pulse fires while
            // still in SEQ_IDLE) and every subsequent run (start_pulse
            // fires while still in SEQ_DONE) correctly. See
            // tb_cnn_top.v's second back-to-back inference, which is
            // exactly the test that would have caught this.
            img_load_cnt    <= 10'd0;
            fifo_rd_en_d    <= 1'b0;
            img_load_addr_d <= 10'd0;
        end else begin
            fifo_rd_en_d    <= fifo_rd_en;
            img_load_addr_d <= img_load_cnt;
            if (fifo_rd_en) img_load_cnt <= img_load_cnt + 1'b1;
        end
    end

    // -------------------------------------------------------------------
    // Sequencer FSM
    // -------------------------------------------------------------------
    wire conv1_done, pool1_done, conv2_done, pool2_done, fc1_done, fc2_done;
    reg  conv1_start, pool1_start, conv2_start, pool2_start, fc1_start, fc2_start;

    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            seq_state   <= SEQ_IDLE;
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            fc1_start   <= 1'b0;
            fc2_start   <= 1'b0;
        end else begin
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            fc1_start   <= 1'b0;
            fc2_start   <= 1'b0;

            case (seq_state)
                SEQ_IDLE: if (start_pulse) seq_state <= SEQ_LOAD;

                SEQ_LOAD: if (input_loaded && !fifo_rd_en_d) begin
                    conv1_start <= 1'b1;
                    seq_state   <= SEQ_CONV1;
                end

                SEQ_CONV1: if (conv1_done) begin
                    pool1_start <= 1'b1;
                    seq_state   <= SEQ_POOL1;
                end

                SEQ_POOL1: if (pool1_done) begin
                    conv2_start <= 1'b1;
                    seq_state   <= SEQ_CONV2;
                end

                SEQ_CONV2: if (conv2_done) begin
                    pool2_start <= 1'b1;
                    seq_state   <= SEQ_POOL2;
                end

                SEQ_POOL2: if (pool2_done) begin
                    fc1_start <= 1'b1;
                    seq_state <= SEQ_FC1;
                end

                SEQ_FC1: if (fc1_done) begin
                    fc2_start <= 1'b1;
                    seq_state <= SEQ_FC2;
                end

                SEQ_FC2: if (fc2_done) begin
                    seq_state <= SEQ_DONE;
                end

                SEQ_DONE: if (start_pulse) seq_state <= SEQ_LOAD; // allow a fresh run

                default: seq_state <= SEQ_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------
    // img_mem (784 x 8b): loader writes, conv1 reads
    // -------------------------------------------------------------------
    wire [9:0] c1_in_addr;
    wire signed [7:0] c1_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(784), .ADDR_W(10)) u_img_mem (
        .clk(core_clk),
        .we  (fifo_rd_en_d),
        .addr(fifo_rd_en_d ? img_load_addr_d : c1_in_addr),
        .din (fifo_rd_data),
        .dout(c1_in_data)
    );

    // -------------------------------------------------------------------
    // conv1 (1x28x28 -> 6x24x24)
    // -------------------------------------------------------------------
    wire        c1_out_we;
    wire [11:0] c1_out_addr;
    wire signed [7:0] c1_out_data;
    wire [`NUM_PE-1:0] c1_mac_vec, c1_skip_vec;

    conv_layer_engine #(
        .CIN(`C1_CIN), .COUT(`C1_COUT), .IN_H(`IMG_H), .IN_W(`IMG_W), .K(`C1_K),
        .OUT_H(`C1_OUT_H), .OUT_W(`C1_OUT_W), .NUM_PE(`NUM_PE),
        .ACT_W(`ACT_W), .WGT_W(`WGT_W), .ACC_W(`ACC_W), .SHIFT(`SHIFT_CONV1),
        .IN_ADDR_W(10), .OUT_ADDR_W(12), .WGT_ADDR_W(8), .BIAS_ADDR_W(3),
        .WGT_FILE("model/hex/conv1_w.hex"), .BIAS_FILE("model/hex/conv1_b.hex")
    ) u_conv1 (
        .clk(core_clk), .rst_n(core_rst_n), .start(conv1_start),
        .busy(), .done(conv1_done),
        .in_addr(c1_in_addr), .in_data(c1_in_data),
        .out_we(c1_out_we), .out_addr(c1_out_addr), .out_data(c1_out_data),
        .perf_mac_vec(c1_mac_vec), .perf_skip_vec(c1_skip_vec)
    );

    // -------------------------------------------------------------------
    // conv1_out_mem / pool1_in (3456 x 8b): conv1 writes, pool1 reads
    // -------------------------------------------------------------------
    wire [11:0] p1_in_addr;
    wire signed [7:0] p1_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(3456), .ADDR_W(12)) u_conv1_out_mem (
        .clk(core_clk),
        .we  (c1_out_we),
        .addr(c1_out_we ? c1_out_addr : p1_in_addr),
        .din (c1_out_data),
        .dout(p1_in_data)
    );

    // -------------------------------------------------------------------
    // pool1 (6x24x24 -> 6x12x12)
    // -------------------------------------------------------------------
    wire        p1_out_we;
    wire [9:0]  p1_out_addr;
    wire signed [7:0] p1_out_data;

    pool_engine #(
        .CH(`C1_COUT), .IN_H(`C1_OUT_H), .IN_W(`C1_OUT_W),
        .OUT_H(`P1_OUT_H), .OUT_W(`P1_OUT_W), .DATA_W(8),
        .IN_ADDR_W(12), .OUT_ADDR_W(10)
    ) u_pool1 (
        .clk(core_clk), .rst_n(core_rst_n), .start(pool1_start),
        .busy(), .done(pool1_done),
        .in_addr(p1_in_addr), .in_data(p1_in_data),
        .out_we(p1_out_we), .out_addr(p1_out_addr), .out_data(p1_out_data)
    );

    // -------------------------------------------------------------------
    // pool1_out_mem / conv2_in (864 x 8b): pool1 writes, conv2 reads
    // -------------------------------------------------------------------
    wire [9:0] c2_in_addr;
    wire signed [7:0] c2_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(864), .ADDR_W(10)) u_pool1_out_mem (
        .clk(core_clk),
        .we  (p1_out_we),
        .addr(p1_out_we ? p1_out_addr : c2_in_addr),
        .din (p1_out_data),
        .dout(c2_in_data)
    );

    // -------------------------------------------------------------------
    // conv2 (6x12x12 -> 16x10x10)
    // -------------------------------------------------------------------
    wire        c2_out_we;
    wire [10:0] c2_out_addr;
    wire signed [7:0] c2_out_data;
    wire [`NUM_PE-1:0] c2_mac_vec, c2_skip_vec;

    conv_layer_engine #(
        .CIN(`C2_CIN), .COUT(`C2_COUT), .IN_H(`P1_OUT_H), .IN_W(`P1_OUT_W), .K(`C2_K),
        .OUT_H(`C2_OUT_H), .OUT_W(`C2_OUT_W), .NUM_PE(`NUM_PE),
        .ACT_W(`ACT_W), .WGT_W(`WGT_W), .ACC_W(`ACC_W), .SHIFT(`SHIFT_CONV2),
        .IN_ADDR_W(10), .OUT_ADDR_W(11), .WGT_ADDR_W(10), .BIAS_ADDR_W(4),
        .WGT_FILE("model/hex/conv2_w.hex"), .BIAS_FILE("model/hex/conv2_b.hex")
    ) u_conv2 (
        .clk(core_clk), .rst_n(core_rst_n), .start(conv2_start),
        .busy(), .done(conv2_done),
        .in_addr(c2_in_addr), .in_data(c2_in_data),
        .out_we(c2_out_we), .out_addr(c2_out_addr), .out_data(c2_out_data),
        .perf_mac_vec(c2_mac_vec), .perf_skip_vec(c2_skip_vec)
    );

    // -------------------------------------------------------------------
    // conv2_out_mem / pool2_in (1600 x 8b): conv2 writes, pool2 reads
    // -------------------------------------------------------------------
    wire [10:0] p2_in_addr;
    wire signed [7:0] p2_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(1600), .ADDR_W(11)) u_conv2_out_mem (
        .clk(core_clk),
        .we  (c2_out_we),
        .addr(c2_out_we ? c2_out_addr : p2_in_addr),
        .din (c2_out_data),
        .dout(p2_in_data)
    );

    // -------------------------------------------------------------------
    // pool2 (16x10x10 -> 16x5x5)
    // -------------------------------------------------------------------
    wire       p2_out_we;
    wire [8:0] p2_out_addr;
    wire signed [7:0] p2_out_data;

    pool_engine #(
        .CH(`C2_COUT), .IN_H(`C2_OUT_H), .IN_W(`C2_OUT_W),
        .OUT_H(`P2_OUT_H), .OUT_W(`P2_OUT_W), .DATA_W(8),
        .IN_ADDR_W(11), .OUT_ADDR_W(9)
    ) u_pool2 (
        .clk(core_clk), .rst_n(core_rst_n), .start(pool2_start),
        .busy(), .done(pool2_done),
        .in_addr(p2_in_addr), .in_data(p2_in_data),
        .out_we(p2_out_we), .out_addr(p2_out_addr), .out_data(p2_out_data)
    );

    // -------------------------------------------------------------------
    // pool2_out_mem == fc1_in_mem (400 x 8b): pool2 writes, fc1 reads
    // (flatten is "free" -- the channel-major addressing chosen for conv2's
    //  output already matches the 400-element FC1 input ordering)
    // -------------------------------------------------------------------
    wire [8:0] fc1_in_addr;
    wire signed [7:0] fc1_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(400), .ADDR_W(9)) u_pool2_out_mem (
        .clk(core_clk),
        .we  (p2_out_we),
        .addr(p2_out_we ? p2_out_addr : fc1_in_addr),
        .din (p2_out_data),
        .dout(fc1_in_data)
    );

    // -------------------------------------------------------------------
    // FC1 (400 -> 120), ReLU
    // -------------------------------------------------------------------
    wire        fc1_out_we;
    wire [6:0]  fc1_out_addr;
    wire signed [7:0] fc1_out_data;
    wire [`NUM_PE-1:0] fc1_mac_vec, fc1_skip_vec;

    fc_layer_engine #(
        .N_IN(`FC1_IN), .N_OUT(`FC1_OUT), .NUM_PE(`NUM_PE),
        .ACT_W(`ACT_W), .WGT_W(`WGT_W), .ACC_W(`ACC_W), .SHIFT(`SHIFT_FC1),
        .IN_ADDR_W(9), .OUT_ADDR_W(7), .WGT_ADDR_W(16), .BIAS_ADDR_W(7),
        .WGT_FILE("model/hex/fc1_w.hex"), .BIAS_FILE("model/hex/fc1_b.hex")
    ) u_fc1 (
        .clk(core_clk), .rst_n(core_rst_n), .start(fc1_start),
        .busy(), .done(fc1_done),
        .in_addr(fc1_in_addr), .in_data(fc1_in_data),
        .out_we(fc1_out_we), .out_addr(fc1_out_addr), .out_data(fc1_out_data),
        .raw_we(), .raw_addr(), .raw_data(),
        .perf_mac_vec(fc1_mac_vec), .perf_skip_vec(fc1_skip_vec)
    );

    // -------------------------------------------------------------------
    // fc1_out_mem == fc2_in_mem (120 x 8b): fc1 writes, fc2 reads
    // -------------------------------------------------------------------
    wire [6:0] fc2_in_addr;
    wire signed [7:0] fc2_in_data;

    sync_ram #(.WIDTH(8), .DEPTH(120), .ADDR_W(7)) u_fc1_out_mem (
        .clk(core_clk),
        .we  (fc1_out_we),
        .addr(fc1_out_we ? fc1_out_addr : fc2_in_addr),
        .din (fc1_out_data),
        .dout(fc2_in_data)
    );

    // -------------------------------------------------------------------
    // FC2 (120 -> 10), raw INT32 classification scores, no activation
    // -------------------------------------------------------------------
    wire        fc2_raw_we;
    wire [3:0]  fc2_raw_addr;
    wire signed [31:0] fc2_raw_data;
    wire [`NUM_PE-1:0] fc2_mac_vec, fc2_skip_vec;

    fc_layer_engine #(
        .N_IN(`FC2_IN), .N_OUT(`FC2_OUT), .NUM_PE(`NUM_PE),
        .ACT_W(`ACT_W), .WGT_W(`WGT_W), .ACC_W(`ACC_W), .SHIFT(0),
        .IN_ADDR_W(7), .OUT_ADDR_W(4), .WGT_ADDR_W(11), .BIAS_ADDR_W(4),
        .WGT_FILE("model/hex/fc2_w.hex"), .BIAS_FILE("model/hex/fc2_b.hex")
    ) u_fc2 (
        .clk(core_clk), .rst_n(core_rst_n), .start(fc2_start),
        .busy(), .done(fc2_done),
        .in_addr(fc2_in_addr), .in_data(fc2_in_data),
        .out_we(), .out_addr(), .out_data(),
        .raw_we(fc2_raw_we), .raw_addr(fc2_raw_addr), .raw_data(fc2_raw_data),
        .perf_mac_vec(fc2_mac_vec), .perf_skip_vec(fc2_skip_vec)
    );

    // -------------------------------------------------------------------
    // Score registers (10 x 32b), written one at a time by fc2
    // -------------------------------------------------------------------
    reg signed [31:0] score [0:9];
    integer si;
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            for (si = 0; si < 10; si = si + 1) score[si] <= 32'sd0;
        end else if (fc2_raw_we) begin
            score[fc2_raw_addr] <= fc2_raw_data;
        end
    end

    assign out_score_0 = score[0]; assign out_score_1 = score[1];
    assign out_score_2 = score[2]; assign out_score_3 = score[3];
    assign out_score_4 = score[4]; assign out_score_5 = score[5];
    assign out_score_6 = score[6]; assign out_score_7 = score[7];
    assign out_score_8 = score[8]; assign out_score_9 = score[9];

    // -------------------------------------------------------------------
    // Performance counters: population count, NOT an OR-reduction.
    //
    // Every active PE lane in a layer's current group fires TOGETHER each
    // cycle (they share the same broadcast activation, so they all either
    // execute or all zero-gate on a given tap). A cycle where e.g. 6 lanes
    // fire is 6 MAC operations, not 1 -- so a plain OR of the per-lane
    // vectors (an earlier version of this file did exactly that) silently
    // undercounts by up to NUM_PE-to-1. Fixed by summing an explicit
    // popcount of each engine's per-lane vector every cycle; only one
    // engine is ever active at a time, so summing across all four engines
    // is safe (the idle ones contribute 0).
    // -------------------------------------------------------------------
    function [3:0] popcount8;
        input [`NUM_PE-1:0] vec;
        integer k;
        begin
            popcount8 = 0;
            for (k = 0; k < `NUM_PE; k = k + 1)
                popcount8 = popcount8 + vec[k];
        end
    endfunction

    wire [3:0] c1_mac_cnt  = popcount8(c1_mac_vec);
    wire [3:0] c1_skip_cnt = popcount8(c1_skip_vec);
    wire [3:0] c2_mac_cnt  = popcount8(c2_mac_vec);
    wire [3:0] c2_skip_cnt = popcount8(c2_skip_vec);
    wire [3:0] fc1_mac_cnt  = popcount8(fc1_mac_vec);
    wire [3:0] fc1_skip_cnt = popcount8(fc1_skip_vec);
    wire [3:0] fc2_mac_cnt  = popcount8(fc2_mac_vec);
    wire [3:0] fc2_skip_cnt = popcount8(fc2_skip_vec);

    wire [3:0] perf_mac_inc  = c1_mac_cnt  + c2_mac_cnt  + fc1_mac_cnt  + fc2_mac_cnt;
    wire [3:0] perf_skip_inc = c1_skip_cnt + c2_skip_cnt + fc1_skip_cnt + fc2_skip_cnt;

    perf_counters #(.W(32), .INC_W(4)) u_perf (
        .clk(core_clk), .rst_n(core_rst_n),
        .busy(busy), .mac_inc(perf_mac_inc), .skip_inc(perf_skip_inc),
        .clear(start_pulse),
        .cycle_count(perf_cycles), .mac_count(perf_macs), .skip_count(perf_skip)
    );

endmodule

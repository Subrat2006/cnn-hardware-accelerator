// =============================================================================
// tb_cnn_top.v
//
// Self-checking testbench for cnn_top. Drives a simple AXI4-Lite
// bus-functional model (BFM) from a "software" initial block and runs
// THREE checks, not one:
//
//   Test 1 -- normal inference: stream the 784-byte image, start, poll
//             done, read back PERF_* and the 10 scores, compare bit-exactly
//             against tb/expected_scores.hex (produced by
//             model/generate_vectors.py running the same integer
//             arithmetic in Python -- see model/golden_model.py).
//   Test 2 -- REPEATED back-to-back inference: run the exact same
//             sequence a second time with NO reset in between (only a
//             fresh CTRL.start), and check run #2's scores match run #1's
//             bit-exactly. This is the test that catches the loader
//             counter reset bug fixed in cnn_core.v (the sequencer's
//             SEQ_DONE -> SEQ_LOAD transition never passes through
//             SEQ_IDLE, so a reset gated on "seq_state==SEQ_IDLE" alone
//             left img_load_cnt stuck at 784 and silently skipped loading
//             the second image).
//   Test 3 -- performance-counter invariant: PERF_MACS + PERF_SKIP must
//             equal 222,000 exactly. This is a hard architectural
//             constant (total MAC-lane tap-events across every layer:
//             Conv1 6*576*25=86,400 + Conv2 2*8*100*54=86,400 + FC1
//             15*8*400=48,000 + FC2 (8+2)*120=1,200 = 222,000), fixed by
//             the network's dimensions and PE count alone -- it does NOT
//             depend on the actual weight/image values, so it holds for
//             ANY test vectors and is a clean, data-independent sanity
//             check on the performance-counter instrumentation itself.
//
// Implementation note: the two inference runs below are written out in
// full rather than factored into a shared task with array-typed ports, so
// the file only relies on conservative Verilog-2001-style constructs and
// compiles the same way across simulator versions.
//
// Results are also written to sim/rtl_results.txt in a simple KEY=VALUE
// format that model/update_readme_results.py parses to patch the actual
// numbers into README.md automatically -- see README Section 4.
//
// AXI clock and core clock are deliberately different frequencies to
// exercise the CDC logic (cdc_pulse_sync / cdc_sync_2ff / async_fifo).
//
// Run with (Icarus Verilog, from the project ROOT directory):
//   iverilog -g2012 -o sim/sim.out -I rtl ^
//     rtl/mac_unit.v rtl/pe_array.v rtl/sync_ram.v ^
//     rtl/weight_rom.v rtl/relu_quant.v rtl/max_pool2x2.v rtl/pool_engine.v ^
//     rtl/conv_layer_engine.v rtl/fc_layer_engine.v rtl/perf_counters.v ^
//     rtl/cdc_sync_2ff.v rtl/cdc_pulse_sync.v rtl/async_fifo.v ^
//     rtl/axi4_lite_slave.v rtl/cnn_core.v rtl/cnn_top.v tb/tb_cnn_top.v
//   vvp sim/sim.out
// (see sim/run_sim.sh / sim/run_sim.bat for the ready-made version of this)
// =============================================================================

`timescale 1ns/1ps
`include "cnn_pkg.vh"

module tb_cnn_top;

    localparam integer EXPECTED_TOTAL_MAC_EVENTS = 222_000;

    // ---------------- clocks / reset ----------------
    reg axi_aclk = 0;
    reg core_clk = 0;
    reg axi_aresetn = 0;
    reg core_rst_n  = 0;

    always #5.0  axi_aclk = ~axi_aclk;   // 100 MHz AXI clock
    always #3.33 core_clk = ~core_clk;   // ~150 MHz core clock (different domain)

    // ---------------- AXI4-Lite signals ----------------
    reg  [7:0]  s_axi_awaddr;
    reg         s_axi_awvalid;
    wire        s_axi_awready;

    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;

    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;

    reg  [7:0]  s_axi_araddr;
    reg         s_axi_arvalid;
    wire        s_axi_arready;

    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    cnn_top dut (
        .axi_aclk(axi_aclk), .axi_aresetn(axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .core_clk(core_clk), .core_rst_n(core_rst_n)
    );

    // ---------------- AXI4-Lite BFM tasks ----------------
    // Both tasks below bound their handshake-wait loops (MAX_WAIT cycles)
    // and print a diagnostic + $finish instead of hanging silently if the
    // DUT never responds -- see README "Troubleshooting" for what each
    // message means.
    localparam integer MAX_WAIT = 200;

    task axi_write(input [7:0] addr, input [31:0] data);
        integer wait_cnt;
        begin
            @(posedge axi_aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;
            @(posedge axi_aclk);
            wait_cnt = 0;
            while (!(s_axi_awready && s_axi_wready)) begin
                @(posedge axi_aclk);
                wait_cnt = wait_cnt + 1;
                if (wait_cnt >= MAX_WAIT) begin
                    $display(" *** axi_write(addr=0x%0h): AWREADY/WREADY never asserted after %0d cycles -- AXI slave FSM is stuck. ***", addr, wait_cnt);
                    $finish;
                end
            end
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;
            wait_cnt = 0;
            while (!s_axi_bvalid) begin
                @(posedge axi_aclk);
                wait_cnt = wait_cnt + 1;
                if (wait_cnt >= MAX_WAIT) begin
                    $display(" *** axi_write(addr=0x%0h): BVALID never asserted after %0d cycles -- AXI slave FSM is stuck. ***", addr, wait_cnt);
                    $finish;
                end
            end
            @(posedge axi_aclk);
            s_axi_bready <= 1'b0;
        end
    endtask

    task axi_read(input [7:0] addr, output [31:0] data);
        integer wait_cnt;
        begin
            @(posedge axi_aclk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;
            @(posedge axi_aclk);
            wait_cnt = 0;
            while (!s_axi_arready) begin
                @(posedge axi_aclk);
                wait_cnt = wait_cnt + 1;
                if (wait_cnt >= MAX_WAIT) begin
                    $display(" *** axi_read(addr=0x%0h): ARREADY never asserted after %0d cycles -- AXI slave FSM is stuck. ***", addr, wait_cnt);
                    $finish;
                end
            end
            s_axi_arvalid <= 1'b0;
            wait_cnt = 0;
            while (!s_axi_rvalid) begin
                @(posedge axi_aclk);
                wait_cnt = wait_cnt + 1;
                if (wait_cnt >= MAX_WAIT) begin
                    $display(" *** axi_read(addr=0x%0h): RVALID never asserted after %0d cycles -- AXI slave FSM is stuck. ***", addr, wait_cnt);
                    $finish;
                end
            end
            data = s_axi_rdata;
            @(posedge axi_aclk);
            s_axi_rready <= 1'b0;
        end
    endtask

    // one whole inference, folded into a task with SCALAR ports only
    // (no array-typed task ports, for maximum simulator compatibility);
    // the caller passes an index telling it which of the two global
    // score arrays below to fill in.
    reg [31:0] score_run1 [0:9];
    reg [31:0] score_run2 [0:9];

    // ---------------- test vectors (declared before the task that reads
    // image_mem, so nothing here relies on a simulator tolerating a
    // forward reference) ----------------
    reg [7:0]  image_mem [0:783];
    reg [31:0] expected_scores [0:9];

    initial begin
        $readmemh("tb/input_image.hex", image_mem);
        $readmemh("tb/expected_scores.hex", expected_scores);
    end

    localparam integer MAX_POLLS = 5000; // ~3x the expected worst case;
                                          // see README "Troubleshooting"

    task run_inference(input integer run_idx,
                        output [31:0] out_cycles,
                        output [31:0] out_macs,
                        output [31:0] out_skip);
        reg [31:0] rdata;
        reg [31:0] one_score;
        integer k;
        integer poll_count;
        begin
            $display(" [run %0d] loading 784-byte image via AXI ...", run_idx);
            for (k = 0; k < 784; k = k + 1) begin
                axi_write(`REG_INPUT_DATA, {24'b0, image_mem[k]});
                if (k % 196 == 195)
                    $display("   ... %0d / 784 bytes written", k + 1);
            end
            $display(" [run %0d] image loaded, issuing START", run_idx);

            axi_write(`REG_CTRL, 32'h0000_0001); // start

            rdata = 0;
            poll_count = 0;
            while (!rdata[1]) begin
                axi_read(`REG_STATUS, rdata);
                poll_count = poll_count + 1;
                if (poll_count % 200 == 0)
                    $display("   ... still polling STATUS (poll #%0d, STATUS=0x%0h, busy=%0b) -- if this number keeps climbing the simulation IS progressing, just slowly",
                              poll_count, rdata, rdata[0]);
                if (poll_count >= MAX_POLLS) begin
                    $display(" *** [run %0d] STATUS.done never went high after %0d polls. ***", run_idx, poll_count);
                    $display(" *** This means the RTL is genuinely stuck (not just slow) -- likely stalled in SEQ_LOAD or a layer engine. ***");
                    $display(" *** Last STATUS read: 0x%0h (busy=%0b, done=%0b, input_loaded=%0b) ***", rdata, rdata[0], rdata[1], rdata[2]);
                    $display(" *** Stopping here instead of waiting for the global timeout -- see README \"Troubleshooting\". ***");
                    $finish;
                end
                repeat (20) @(posedge axi_aclk);
            end
            $display(" [run %0d] done after %0d polls", run_idx, poll_count);

            axi_read(`REG_PERF_CYCLES, out_cycles);
            axi_read(`REG_PERF_MACS,   out_macs);
            axi_read(`REG_PERF_SKIP,   out_skip);

            for (k = 0; k < 10; k = k + 1) begin
                axi_read(`REG_OUT_BASE + k * 4, one_score);
                if (run_idx == 1) score_run1[k] = one_score;
                else              score_run2[k] = one_score;
            end
        end
    endtask

    // ---------------- main sequence ----------------
    integer i;
    integer pred_class;
    reg signed [31:0] best_score;

    reg [31:0] perf_cycles_run1, perf_macs_run1, perf_skip_run1;
    reg [31:0] perf_cycles_run2, perf_macs_run2, perf_skip_run2;

    integer errors_run1, errors_run2, repeat_mismatches;
    reg invariant1_ok, invariant2_ok, repeat_ok, run1_ok, run2_ok, overall_ok;
    integer fd;

    initial begin
        s_axi_awaddr = 0; s_axi_awvalid = 0;
        s_axi_wdata  = 0; s_axi_wstrb = 0; s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_araddr = 0; s_axi_arvalid = 0; s_axi_rready = 0;

        // reset pulse on both domains
        repeat (5) @(posedge axi_aclk);
        axi_aresetn = 0; core_rst_n = 0;
        repeat (10) @(posedge axi_aclk);
        axi_aresetn = 1; core_rst_n = 1;
        repeat (5) @(posedge axi_aclk);

        // ---------------- Test 1: normal inference ----------------
        $display("===========================================================");
        $display(" TEST 1: normal inference");
        $display("===========================================================");
        run_inference(1, perf_cycles_run1, perf_macs_run1, perf_skip_run1);

        errors_run1 = 0;
        for (i = 0; i < 10; i = i + 1) begin
            if (score_run1[i] !== expected_scores[i]) begin
                errors_run1 = errors_run1 + 1;
                $display(" class %0d: RTL=%0d  golden=%0d   *** MISMATCH ***",
                          i, $signed(score_run1[i]), $signed(expected_scores[i]));
            end else begin
                $display(" class %0d: %0d   (match)", i, $signed(score_run1[i]));
            end
        end
        run1_ok = (errors_run1 == 0);

        $display(" total cycles        : %0d", perf_cycles_run1);
        $display(" real MAC ops        : %0d", perf_macs_run1);
        $display(" zero-gated (skipped): %0d", perf_skip_run1);
        $display(" sparsity (skip%%)    : %.1f%%",
                  100.0 * perf_skip_run1 / (perf_macs_run1 + perf_skip_run1));
        invariant1_ok = ((perf_macs_run1 + perf_skip_run1) == EXPECTED_TOTAL_MAC_EVENTS);
        $display(" invariant PERF_MACS + PERF_SKIP == %0d : %s",
                  EXPECTED_TOTAL_MAC_EVENTS, invariant1_ok ? "PASS" : "FAIL");
        $display(" run 1 vs golden model                  : %s", run1_ok ? "PASS" : "FAIL");

        // ---------------- Test 2: repeated back-to-back inference ----------------
        // No reset here on purpose -- this exercises the sequencer's
        // SEQ_DONE -> SEQ_LOAD path (see MUST-FIX #2 in README Section 13).
        $display("===========================================================");
        $display(" TEST 2: repeated (back-to-back) inference, no reset");
        $display("===========================================================");
        run_inference(2, perf_cycles_run2, perf_macs_run2, perf_skip_run2);

        errors_run2 = 0;
        repeat_mismatches = 0;
        for (i = 0; i < 10; i = i + 1) begin
            if (score_run2[i] !== expected_scores[i]) errors_run2 = errors_run2 + 1;
            if (score_run2[i] !== score_run1[i]) begin
                repeat_mismatches = repeat_mismatches + 1;
                $display(" class %0d: run1=%0d  run2=%0d   *** REPEAT MISMATCH ***",
                          i, $signed(score_run1[i]), $signed(score_run2[i]));
            end
        end
        run2_ok       = (errors_run2 == 0);
        repeat_ok     = (repeat_mismatches == 0);
        invariant2_ok = ((perf_macs_run2 + perf_skip_run2) == EXPECTED_TOTAL_MAC_EVENTS);

        $display(" run 2 vs golden model                  : %s", run2_ok ? "PASS" : "FAIL");
        $display(" run 2 == run 1 (bit-exact repeat)       : %s", repeat_ok ? "PASS" : "FAIL");
        $display(" invariant PERF_MACS + PERF_SKIP == %0d : %s",
                  EXPECTED_TOTAL_MAC_EVENTS, invariant2_ok ? "PASS" : "FAIL");

        // ---------------- predicted class (argmax of run 1) ----------------
        pred_class = 0;
        best_score = $signed(score_run1[0]);
        for (i = 1; i < 10; i = i + 1) begin
            if ($signed(score_run1[i]) > best_score) begin
                best_score = $signed(score_run1[i]);
                pred_class = i;
            end
        end

        overall_ok = run1_ok && run2_ok && repeat_ok && invariant1_ok && invariant2_ok;

        $display("===========================================================");
        $display(" OVERALL RESULT: %s", overall_ok ? "PASS" : "FAIL");
        $display("===========================================================");

        // ---------------- machine-readable results file ----------------
        // Parsed by model/update_readme_results.py to patch README.md's
        // results section automatically -- see README Section 4.
        fd = $fopen("sim/rtl_results.txt", "w");
        if (fd) begin
            $fwrite(fd, "PERF_CYCLES=%0d\n", perf_cycles_run1);
            $fwrite(fd, "PERF_MACS=%0d\n", perf_macs_run1);
            $fwrite(fd, "PERF_SKIP=%0d\n", perf_skip_run1);
            $fwrite(fd, "PREDICTED_CLASS=%0d\n", pred_class);
            $fwrite(fd, "RUN1_MATCH=%s\n", run1_ok ? "PASS" : "FAIL");
            $fwrite(fd, "RUN2_MATCH=%s\n", run2_ok ? "PASS" : "FAIL");
            $fwrite(fd, "REPEAT_MATCH=%s\n", repeat_ok ? "PASS" : "FAIL");
            $fwrite(fd, "INVARIANT_MATCH=%s\n", (invariant1_ok && invariant2_ok) ? "PASS" : "FAIL");
            $fwrite(fd, "OVERALL=%s\n", overall_ok ? "PASS" : "FAIL");
            $fclose(fd);
            $display("Wrote sim/rtl_results.txt");
        end else begin
            $display("WARNING: could not open sim/rtl_results.txt for writing -- run from the project ROOT directory.");
        end

        $finish;
    end

    // safety timeout
    initial begin
        #4_000_000;
        $display("TIMEOUT: accelerator never asserted done. Failing.");
        $finish;
    end

endmodule

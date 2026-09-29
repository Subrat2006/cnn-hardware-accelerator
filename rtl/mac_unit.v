// =============================================================================
// mac_unit.v
//
// One multiply-accumulate lane with activation-zero gating implemented as
// explicit OPERAND ISOLATION, not just a conditional accumulate.
//
// Zero gating: ReLU produces many exact-zero activations downstream. When
// the activation input for a MAC operation is zero, the product is
// trivially zero and contributes nothing to the accumulator. A weaker
// implementation would compute `product = act_in * wgt_in` unconditionally
// every cycle and only gate whether the *accumulator* updates -- which
// still lets the multiplier's internal nodes toggle every cycle in
// response to act_in/wgt_in, whether or not the result is ever used. This
// implementation instead isolates the multiplier's OWN inputs: on a
// zero-gated cycle, both `gated_act` and `gated_wgt` are forced to a
// constant zero, so the multiplier sees a fixed zero-zero input pair
// rather than whatever act_in/wgt_in happen to be. That is what makes the
// "reduced switching activity" claim in the report literal rather than
// only behavioral -- see README Section 11.1. `mac_skipped` is still
// exposed so the accelerator can count how many operations were actually
// elided (see perf_counters.v).
// =============================================================================

module mac_unit #(
    parameter ACT_W = 8,
    parameter WGT_W = 8,
    parameter ACC_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          en,         // valid MAC op this cycle
    input  wire                          acc_clear,   // start new output accumulation
    input  wire signed [ACT_W-1:0]       act_in,
    input  wire signed [WGT_W-1:0]       wgt_in,
    output reg  signed [ACC_W-1:0]       acc_out,
    output wire                          mac_skipped, // this cycle's op was zero-gated
    output wire                          mac_valid    // this cycle's op actually ran (for perf counting)
);

    wire is_zero_act = (act_in == {ACT_W{1'b0}});
    wire do_mac       = en & ~is_zero_act;

    assign mac_skipped = en & is_zero_act;
    assign mac_valid   = do_mac;

    // Explicit operand isolation (see file header). On a zero-gated
    // cycle both multiplier inputs are forced to zero rather than left
    // to float at whatever act_in/wgt_in currently are.
    wire signed [ACT_W-1:0] gated_act = do_mac ? act_in : {ACT_W{1'b0}};
    wire signed [WGT_W-1:0] gated_wgt = do_mac ? wgt_in : {WGT_W{1'b0}};
    wire signed [ACT_W+WGT_W-1:0] product = gated_act * gated_wgt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= {ACC_W{1'b0}};
        end else if (acc_clear) begin
            // acc_clear takes priority: start a fresh output element.
            // If a MAC also happens on this same cycle (do_mac=1), fold
            // it in directly instead of wasting a cycle just to reset
            // to zero.
            if (do_mac)
                acc_out <= {{(ACC_W-(ACT_W+WGT_W)){product[ACT_W+WGT_W-1]}}, product};
            else
                acc_out <= {ACC_W{1'b0}};
        end else if (do_mac) begin
            acc_out <= acc_out + {{(ACC_W-(ACT_W+WGT_W)){product[ACT_W+WGT_W-1]}}, product};
        end
        // else: zero-gated cycle -> hold acc_out
    end

endmodule

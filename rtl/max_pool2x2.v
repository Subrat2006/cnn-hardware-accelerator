// =============================================================================
// max_pool2x2.v
//
// Pure combinational 2x2 / stride-2 max compare tree. Max pooling needs no
// arithmetic (no adders, no divider) unlike average pooling, which is why
// it was chosen over average pooling for this accelerator (see
// README.md Section 5.4).
// =============================================================================

module max_pool2x2 #(
    parameter WIDTH = 8
)(
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    input  wire signed [WIDTH-1:0] c,
    input  wire signed [WIDTH-1:0] d,
    output wire signed [WIDTH-1:0] max_out
);

    wire signed [WIDTH-1:0] max_ab = (a > b) ? a : b;
    wire signed [WIDTH-1:0] max_cd = (c > d) ? c : d;
    assign max_out = (max_ab > max_cd) ? max_ab : max_cd;

endmodule

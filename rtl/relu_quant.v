// =============================================================================
// relu_quant.v
//
// Combinational post-processing applied once per completed output element:
//   1. sum = accumulator + bias            (int32)
//   2. relu = max(sum, 0)                   (ReLU)
//   3. shifted = relu >>> SHIFT              (arithmetic right shift = cheap
//                                             fixed-point requantization back
//                                             down to the layer's storage
//                                             width -- this is the same
//                                             hardware-friendly scaling used
//                                             by real INT8 quantized
//                                             accelerators, avoiding a real
//                                             divider)
//   4. saturate to OUT_W (signed) range
//
// Since step 2 already forces the value non-negative, saturation only ever
// needs to clip the upper bound.
// =============================================================================

module relu_quant #(
    parameter ACC_W  = 32,
    parameter OUT_W  = 8,
    parameter SHIFT  = 4
)(
    input  wire signed [ACC_W-1:0] acc_in,
    input  wire signed [ACC_W-1:0] bias_in,
    output wire signed [OUT_W-1:0] act_out
);

    wire signed [ACC_W-1:0] sum       = acc_in + bias_in;
    wire signed [ACC_W-1:0] relu_val  = sum[ACC_W-1] ? {ACC_W{1'b0}} : sum;
    wire signed [ACC_W-1:0] shifted   = relu_val >>> SHIFT;

    localparam signed [ACC_W-1:0] MAX_VAL = (1 <<< (OUT_W-1)) - 1;

    assign act_out = (shifted > MAX_VAL) ? MAX_VAL[OUT_W-1:0] : shifted[OUT_W-1:0];

endmodule

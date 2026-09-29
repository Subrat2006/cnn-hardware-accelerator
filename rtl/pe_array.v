// =============================================================================
// pe_array.v
//
// NUM_PE independent MAC lanes. Every layer engine (conv1, conv2, fc1, fc2)
// instantiates its own copy of this array, so one lane design is written
// and verified once and reused by convolution and fully-connected layers
// alike. Narrow layers leave some lanes idle (conv1 uses 6 of 8), which is
// the price of a single, simple, re-usable datapath.
// =============================================================================

module pe_array #(
    parameter NUM_PE = 8,
    parameter ACT_W  = 8,
    parameter WGT_W  = 8,
    parameter ACC_W  = 32
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire [NUM_PE-1:0]                 en,
    input  wire [NUM_PE-1:0]                 acc_clear,
    input  wire signed [NUM_PE*ACT_W-1:0]    act_in_flat,
    input  wire signed [NUM_PE*WGT_W-1:0]    wgt_in_flat,
    output wire signed [NUM_PE*ACC_W-1:0]    acc_out_flat,
    output wire [NUM_PE-1:0]                 mac_skipped,
    output wire [NUM_PE-1:0]                 mac_valid
);

    genvar i;
    generate
        for (i = 0; i < NUM_PE; i = i + 1) begin : PE
            mac_unit #(
                .ACT_W(ACT_W),
                .WGT_W(WGT_W),
                .ACC_W(ACC_W)
            ) u_mac (
                .clk        (clk),
                .rst_n      (rst_n),
                .en         (en[i]),
                .acc_clear  (acc_clear[i]),
                .act_in     (act_in_flat[i*ACT_W +: ACT_W]),
                .wgt_in     (wgt_in_flat[i*WGT_W +: WGT_W]),
                .acc_out    (acc_out_flat[i*ACC_W +: ACC_W]),
                .mac_skipped(mac_skipped[i]),
                .mac_valid  (mac_valid[i])
            );
        end
    endgenerate

endmodule

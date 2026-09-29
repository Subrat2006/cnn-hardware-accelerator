// =============================================================================
// cdc_pulse_sync.v
//
// Safely transfers a single-cycle PULSE (e.g. the AXI-domain "start" strobe)
// into another clock domain, regardless of the two clocks' frequency ratio.
// A pulse is too short to sample reliably with a plain 2FF synchronizer (it
// could be missed entirely if it doesn't line up with the destination
// clock edge), so this uses the standard toggle + double-synchronize +
// edge-detect handshake instead: the source domain toggles a level signal
// on every pulse, the destination domain double-flops that level (safe,
// because a level held for a full source-clock period is always long
// enough to be caught), and an edge detector on the synchronized signal
// regenerates a clean one-cycle pulse in the destination domain.
// =============================================================================

module cdc_pulse_sync (
    input  wire src_clk,
    input  wire src_rst_n,
    input  wire pulse_in,

    input  wire dst_clk,
    input  wire dst_rst_n,
    output wire pulse_out
);

    reg toggle_src;
    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n)      toggle_src <= 1'b0;
        else if (pulse_in)   toggle_src <= ~toggle_src;
    end

    reg [2:0] sync_dst;
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) sync_dst <= 3'b000;
        else            sync_dst <= {sync_dst[1:0], toggle_src};
    end

    // rising or falling edge on the synchronized toggle == one source pulse
    assign pulse_out = sync_dst[2] ^ sync_dst[1];

endmodule

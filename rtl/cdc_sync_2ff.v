// =============================================================================
// cdc_sync_2ff.v
//
// Standard double-flop synchronizer for LEVEL signals (e.g. busy/done status
// bits) crossing from one clock domain to another. Only safe for signals
// that change slowly relative to both clocks and where a small, bounded
// synchronization latency (2 destination clock cycles) is acceptable --
// exactly the case for status flags. NOT used for multi-bit data buses that
// can change every cycle (those go through async_fifo.v instead).
// =============================================================================

module cdc_sync_2ff #(
    parameter WIDTH = 1
)(
    input  wire               dst_clk,
    input  wire                dst_rst_n,
    input  wire [WIDTH-1:0]    async_in,
    output reg  [WIDTH-1:0]    sync_out
);

    reg [WIDTH-1:0] meta;

    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            meta     <= {WIDTH{1'b0}};
            sync_out <= {WIDTH{1'b0}};
        end else begin
            meta     <= async_in;
            sync_out <= meta;
        end
    end

endmodule

// =============================================================================
// perf_counters.v
//
// Free-running counters exposed through AXI (PERF_CYCLES, PERF_MACS,
// PERF_SKIP).
//
// IMPORTANT (fixed after review): `mac_inc` / `skip_inc` must already be the
// POPULATION COUNT (number of lanes that fired this cycle), not a single
// OR-reduced "did anything happen" bit. An earlier version of this design
// used `mac_event = |mac_valid_vec` and incremented by 1 per cycle
// regardless of how many of the 8 PE lanes actually fired -- that silently
// under-counted whenever more than one lane was active in the same cycle
// (which is the common case here, since every active PE in a layer's
// current group fires together). cnn_core.v now computes the real
// per-cycle population count across every layer engine (only one engine is
// ever active at a time, so summing every engine's count is safe) and
// passes THAT in here. This is what makes PERF_MACS / PERF_SKIP -- and the
// sparsity percentage derived from them -- actually mean "number of MAC
// operations", not "number of cycles with >=1 MAC operation".
// =============================================================================

module perf_counters #(
    parameter W        = 32,
    parameter INC_W     = 4     // bits needed for the largest possible per-cycle increment (NUM_PE)
)(
    input  wire            clk,
    input  wire            rst_n,
    input  wire             busy,      // increments cycle_count every cycle busy=1
    input  wire [INC_W-1:0] mac_inc,   // number of lanes that did a real MAC this cycle
    input  wire [INC_W-1:0] skip_inc,  // number of lanes that were zero-gated this cycle
    input  wire             clear,     // synchronous clear (pulse) before a new run

    output reg  [W-1:0] cycle_count,
    output reg  [W-1:0] mac_count,
    output reg  [W-1:0] skip_count
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= {W{1'b0}};
            mac_count   <= {W{1'b0}};
            skip_count  <= {W{1'b0}};
        end else if (clear) begin
            cycle_count <= {W{1'b0}};
            mac_count   <= {W{1'b0}};
            skip_count  <= {W{1'b0}};
        end else begin
            if (busy) cycle_count <= cycle_count + 1'b1;
            mac_count  <= mac_count  + mac_inc;
            skip_count <= skip_count + skip_inc;
        end
    end

endmodule

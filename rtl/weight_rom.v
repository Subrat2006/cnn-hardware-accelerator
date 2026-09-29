// =============================================================================
// weight_rom.v
//
// Weight storage with NUM_PORTS independent, combinational (asynchronous)
// read ports -- one per active PE lane -- so that all PEs can fetch their
// own weight for the current tap in the same cycle.
//
// Design-trade-off note (see README.md Section 8.4):
// modeling the ROM with NUM_PORTS simultaneous async reads is a behavioral
// convenience, but the instances of this module are NOT all small. The
// convolution weight stores are small (150 entries for conv1, 864 for
// conv2), but FC1's is not: at 400x120 = 48,000 8-bit weights it is by far
// the largest storage structure in the whole design, and is the one most
// likely to actually strain an 8-read-port combinational-lookup model
// during synthesis. On an FPGA the small conv ROMs synthesize naturally
// into per-PE distributed LUTRAM/ROM replicas; FC1's 48,000-entry, 8-port
// ROM is a different proposition and may require genuine memory
// replication (8 copies of a 48,000-entry table) or substantial mux/LUT
// resources to realize as 8 independent combinational read ports. On an
// ASIC flow any of these would instead need either N replicated ROM
// macros or a true N-read-port SRAM macro / register-file. This is called
// out explicitly rather than glossed over, since it is exactly the kind
// of memory-architecture trade-off real accelerator designs have to
// make -- and is precisely why README Section 8.4 discusses
// synthesis run (LUT/FF/BRAM usage) rather than trusting this logical
// description alone.
// =============================================================================

module weight_rom #(
    parameter WIDTH     = 8,
    parameter DEPTH     = 150,
    parameter ADDR_W    = 8,
    parameter NUM_PORTS = 8,
    parameter INIT_FILE = ""
)(
    input  wire [NUM_PORTS*ADDR_W-1:0] addr_flat,
    output wire [NUM_PORTS*WIDTH-1:0]  data_flat
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    integer i;
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end else begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] = {WIDTH{1'b0}};
        end
    end

    genvar p;
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : RD_PORT
            assign data_flat[p*WIDTH +: WIDTH] = mem[addr_flat[p*ADDR_W +: ADDR_W]];
        end
    endgenerate

endmodule

// =============================================================================
// sync_ram.v
//
// Generic single read/write port synchronous RAM. Used for feature-map
// storage (input image buffer, conv1/pool1/conv2/pool2 activations) and for
// per-layer bias storage. Synchronous read matches typical FPGA block-RAM /
// ASIC SRAM behavior (1-cycle read latency).
// =============================================================================

module sync_ram #(
    parameter WIDTH      = 8,
    parameter DEPTH      = 1024,
    parameter ADDR_W     = 10,
    parameter INIT_FILE  = ""      // optional $readmemh init file, "" = none
)(
    input  wire                   clk,
    input  wire                   we,
    input  wire [ADDR_W-1:0]      addr,
    input  wire [WIDTH-1:0]       din,
    output reg  [WIDTH-1:0]       dout
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

    always @(posedge clk) begin
        if (we) mem[addr] <= din;
        dout <= mem[addr];
    end

endmodule

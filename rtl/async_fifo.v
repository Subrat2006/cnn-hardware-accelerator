// =============================================================================
// async_fifo.v
//
// Standard dual-clock asynchronous FIFO using gray-coded read/write pointers
// with double-flop synchronization across the clock boundary (the classic
// Cummings-style async FIFO). Used to move the streamed input image bytes
// from the AXI-Lite clock domain into the accelerator core clock domain,
// since this is genuinely a continuous multi-bit DATA stream (unlike the
// start/busy/done status bits, which only need cdc_sync_2ff.v /
// cdc_pulse_sync.v) and a plain synchronizer would corrupt multi-bit values
// sampled mid-transition.
//
// DEPTH = 2**ADDR_W. Gray coding guarantees that only one bit of the
// pointer changes per increment, which is what makes synchronizing a
// multi-bit pointer across an asynchronous boundary safe.
// =============================================================================

module async_fifo #(
    parameter WIDTH  = 8,
    parameter ADDR_W = 4        // depth = 2**ADDR_W
)(
    // write side (AXI clock domain)
    input  wire              wr_clk,
    input  wire               wr_rst_n,
    input  wire               wr_en,
    input  wire [WIDTH-1:0]   wr_data,
    output wire                full,

    // read side (accelerator core clock domain)
    input  wire               rd_clk,
    input  wire               rd_rst_n,
    input  wire               rd_en,
    output reg  [WIDTH-1:0]   rd_data,
    output wire                empty
);

    localparam DEPTH = (1 << ADDR_W);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    reg  [ADDR_W:0] wr_bin, wr_gray;
    reg  [ADDR_W:0] rd_bin, rd_gray;
    reg  [ADDR_W:0] rd_gray_sync1, rd_gray_sync2; // read ptr synced into wr_clk domain
    reg  [ADDR_W:0] wr_gray_sync1, wr_gray_sync2; // write ptr synced into rd_clk domain

    // ---------------- write side ----------------
    wire [ADDR_W:0] wr_bin_next  = wr_bin + (wr_en & ~full);
    wire [ADDR_W:0] wr_gray_next = (wr_bin_next >> 1) ^ wr_bin_next;

    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            wr_bin  <= {(ADDR_W+1){1'b0}};
            wr_gray <= {(ADDR_W+1){1'b0}};
        end else begin
            wr_bin  <= wr_bin_next;
            wr_gray <= wr_gray_next;
        end
    end

    always @(posedge wr_clk) begin
        if (wr_en && !full) mem[wr_bin[ADDR_W-1:0]] <= wr_data;
    end

    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            rd_gray_sync1 <= {(ADDR_W+1){1'b0}};
            rd_gray_sync2 <= {(ADDR_W+1){1'b0}};
        end else begin
            rd_gray_sync1 <= rd_gray;
            rd_gray_sync2 <= rd_gray_sync1;
        end
    end

    assign full = (wr_gray_next == {~rd_gray_sync2[ADDR_W:ADDR_W-1], rd_gray_sync2[ADDR_W-2:0]});

    // ---------------- read side ----------------
    wire [ADDR_W:0] rd_bin_next  = rd_bin + (rd_en & ~empty);
    wire [ADDR_W:0] rd_gray_next = (rd_bin_next >> 1) ^ rd_bin_next;

    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            rd_bin  <= {(ADDR_W+1){1'b0}};
            rd_gray <= {(ADDR_W+1){1'b0}};
            rd_data <= {WIDTH{1'b0}};
        end else begin
            rd_bin  <= rd_bin_next;
            rd_gray <= rd_gray_next;
            if (rd_en && !empty) rd_data <= mem[rd_bin[ADDR_W-1:0]];
        end
    end

    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            wr_gray_sync1 <= {(ADDR_W+1){1'b0}};
            wr_gray_sync2 <= {(ADDR_W+1){1'b0}};
        end else begin
            wr_gray_sync1 <= wr_gray;
            wr_gray_sync2 <= wr_gray_sync1;
        end
    end

    assign empty = (rd_gray_next == wr_gray_sync2);

endmodule

// =============================================================================
// cnn_pkg.vh
// Central configuration for the CNN accelerator.
//
// IMPORTANT: These constants MUST stay in sync with model/config.py, which
// generates the weight/bias/image hex files consumed via $readmemh and the
// expected golden output. If you change dimensions or shift amounts here,
// regenerate the vectors with model/generate_vectors.py using matching values.
// =============================================================================

`ifndef CNN_PKG_VH
`define CNN_PKG_VH

// ---------------------------------------------------------------------------
// Numerical precision
// ---------------------------------------------------------------------------
`define ACT_W        8      // activation / weight bit-width (signed)
`define WGT_W         8
`define ACC_W        32     // accumulator bit-width (signed)

// ---------------------------------------------------------------------------
// Network topology (fixed, compact LeNet-style)
// Input: 28x28x1
// Conv1: 5x5, 6 filters,  valid, stride 1  -> 24x24x6
// Pool1: 2x2 max, stride 2                  -> 12x12x6
// Conv2: 3x3, 16 filters, valid, stride 1   -> 10x10x16
// Pool2: 2x2 max, stride 2                  -> 5x5x16
// Flatten (channel-major: idx = ch*25 + row*5 + col) -> 400
// FC1:   400 -> 120  (ReLU)
// FC2:   120 -> 10   (raw scores, no activation)
// ---------------------------------------------------------------------------
`define IMG_H         28
`define IMG_W         28

`define C1_CIN         1
`define C1_COUT        6
`define C1_K           5
`define C1_OUT_H      24
`define C1_OUT_W      24

`define P1_OUT_H      12
`define P1_OUT_W      12

`define C2_CIN         6
`define C2_COUT       16
`define C2_K           3
`define C2_OUT_H      10
`define C2_OUT_W      10

`define P2_OUT_H       5
`define P2_OUT_W       5

`define FC1_IN        400   // = P2_OUT_H * P2_OUT_W * C2_COUT
`define FC1_OUT       120

`define FC2_IN        120
`define FC2_OUT        10

// ---------------------------------------------------------------------------
// PE array width shared by every layer engine
// ---------------------------------------------------------------------------
`define NUM_PE          8

// ---------------------------------------------------------------------------
// Requantization shifts applied after (accumulator + bias) and ReLU.
// Chosen empirically for the synthetic (random, small-magnitude) test
// weights shipped with this repo -- see model/config.py. Re-tune if you
// swap in real trained weights.
// ---------------------------------------------------------------------------
`define SHIFT_CONV1     4
`define SHIFT_CONV2     6
`define SHIFT_FC1       6
// FC2 (final classifier layer) is left as a raw int32 score: no ReLU, no shift.

// ---------------------------------------------------------------------------
// AXI4-Lite register map (byte addresses, word aligned)
// ---------------------------------------------------------------------------
`define REG_CTRL        8'h00   // [0]=start (pulse); [31:1] reserved
`define REG_STATUS      8'h04   // [0]=busy, [1]=done, [2]=input_loaded
`define REG_INPUT_DATA  8'h08   // write: push next input pixel (lower 8 bits)
`define REG_PERF_CYCLES 8'h0C
`define REG_PERF_MACS   8'h10
`define REG_PERF_SKIP   8'h14
`define REG_OUT_BASE    8'h40   // REG_OUT_BASE + 4*i -> score for class i (i=0..9)

`endif // CNN_PKG_VH

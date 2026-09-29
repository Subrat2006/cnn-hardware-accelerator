"""
config.py

Mirrors rtl/cnn_pkg.vh. If you change dimensions, precision, or shift
amounts here, update cnn_pkg.vh to match (and vice versa) -- the RTL and
this golden model must agree exactly for the self-checking testbench to
pass bit-for-bit.
"""

# ---- network topology --------------------------------------------------
IMG_H, IMG_W = 28, 28

C1_CIN, C1_COUT, C1_K = 1, 6, 5
C1_OUT_H, C1_OUT_W = 24, 24          # 28 - 5 + 1

P1_OUT_H, P1_OUT_W = 12, 12          # 24 / 2

C2_CIN, C2_COUT, C2_K = 6, 16, 3
C2_OUT_H, C2_OUT_W = 10, 10          # 12 - 3 + 1

P2_OUT_H, P2_OUT_W = 5, 5            # 10 / 2

FC1_IN, FC1_OUT = 400, 120           # 5*5*16
FC2_IN, FC2_OUT = 120, 10

NUM_PE = 8

# ---- precision ----------------------------------------------------------
ACT_BITS = 8      # signed
WGT_BITS = 8      # signed
ACC_BITS = 32     # signed

ACT_MIN, ACT_MAX = -(1 << (ACT_BITS - 1)), (1 << (ACT_BITS - 1)) - 1   # -128..127

# ---- requantization shifts (must match `SHIFT_* in cnn_pkg.vh) ----------
SHIFT_CONV1 = 4
SHIFT_CONV2 = 6
SHIFT_FC1   = 6
# FC2 (final layer): no ReLU, no shift -- raw int32 score.

# ---- verification-vector generation ranges (fixed-seed INT8 parameters) --
# See README "Swapping in a trained model".
WEIGHT_RANGE = (-8, 8)     # inclusive randint range for conv/fc weights
BIAS_RANGE   = (-16, 16)   # inclusive randint range for biases
IMG_RANGE    = (0, 127)    # inclusive randint range for the input "pixels"

RANDOM_SEED = 42

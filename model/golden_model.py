"""
golden_model.py

A pure NumPy/Python reference implementation of the accelerator's forward
pass, written to match the RTL's INTEGER arithmetic exactly (same
accumulate order does not matter for a plain sum, but the SAME shift-based
requantization, the SAME ReLU-then-shift-then-saturate sequence, and the
SAME channel-major flatten order are all required for the two to agree
bit-for-bit). See README.md (Section 5) for why each of these choices
was made in hardware.

All feature maps are kept in (Channels, Height, Width) order throughout
("CHW"), and NumPy's default C-order .flatten() is used everywhere a tensor
needs linearizing -- that single convention is what makes the addressing
in every RTL module (img_mem, conv*_out_mem, pool*_out_mem, fc*_in) line
up with this model with no extra bookkeeping.
"""

import numpy as np
from config import (
    ACT_MAX, SHIFT_CONV1, SHIFT_CONV2, SHIFT_FC1,
)


def conv2d_valid(x, w, b, shift):
    """
    x: (Cin, H, W) int
    w: (Cout, Cin, K, K) int
    b: (Cout,) int
    Returns: (Cout, OH, OW) int8-range values (ReLU + requantized).
    """
    Cout, Cin, K, _ = w.shape
    _, H, W = x.shape
    OH, OW = H - K + 1, W - K + 1

    out = np.zeros((Cout, OH, OW), dtype=np.int64)

    x64 = x.astype(np.int64)
    w64 = w.astype(np.int64)

    for oc in range(Cout):
        for oy in range(OH):
            for ox in range(OW):
                window = x64[:, oy:oy + K, ox:ox + K]
                acc = int(np.sum(window * w64[oc]))
                s = acc + int(b[oc])
                r = max(s, 0)                # ReLU
                shifted = r >> shift         # r >= 0, so >> == floor division
                q = min(shifted, ACT_MAX)    # saturate upper bound only
                out[oc, oy, ox] = q

    return out.astype(np.int32)


def maxpool2x2(x):
    """x: (C, H, W) -> (C, H//2, W//2), stride-2 max pooling."""
    C, H, W = x.shape
    OH, OW = H // 2, W // 2
    out = np.zeros((C, OH, OW), dtype=x.dtype)
    for c in range(C):
        for oy in range(OH):
            for ox in range(OW):
                a = x[c, 2 * oy,     2 * ox]
                b = x[c, 2 * oy,     2 * ox + 1]
                cc = x[c, 2 * oy + 1, 2 * ox]
                d = x[c, 2 * oy + 1, 2 * ox + 1]
                out[c, oy, ox] = max(a, b, cc, d)
    return out


def fc_layer(x, w, b, shift, apply_relu):
    """
    x: (N_in,) int
    w: (N_out, N_in) int
    b: (N_out,) int
    apply_relu=True  -> ReLU + requantize to int8 range (hidden layer)
    apply_relu=False -> raw int32 (accumulator + bias) score (final layer)
    """
    N_out = w.shape[0]
    x64 = x.astype(np.int64)
    w64 = w.astype(np.int64)
    out = np.zeros(N_out, dtype=np.int64)

    for o in range(N_out):
        acc = int(np.dot(x64, w64[o]))
        s = acc + int(b[o])
        if apply_relu:
            r = max(s, 0)
            shifted = r >> shift
            q = min(shifted, ACT_MAX)
            out[o] = q
        else:
            out[o] = s

    return out


def forward(img, weights):
    """
    img: (28, 28) array of pixel values (0..127 range used by this repo)
    weights: dict with conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b,
             fc2_w, fc2_b
    Returns: (scores (10,) int32, intermediates dict)
    """
    x = img.reshape(1, 28, 28).astype(np.int32)

    c1 = conv2d_valid(x, weights['conv1_w'], weights['conv1_b'], SHIFT_CONV1)
    p1 = maxpool2x2(c1)

    c2 = conv2d_valid(p1, weights['conv2_w'], weights['conv2_b'], SHIFT_CONV2)
    p2 = maxpool2x2(c2)

    flat = p2.flatten()  # channel-major, matches FC1 input addressing

    h1 = fc_layer(flat, weights['fc1_w'], weights['fc1_b'], SHIFT_FC1, apply_relu=True)
    scores = fc_layer(h1, weights['fc2_w'], weights['fc2_b'], 0, apply_relu=False)

    intermediates = dict(conv1=c1, pool1=p1, conv2=c2, pool2=p2, fc1=h1)
    return scores.astype(np.int32), intermediates

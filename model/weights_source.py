"""
weights_source.py

Single source of truth for the weights/bias/image arrays used by every
script in this repo: generate_vectors.py, cycle_model.py,
generate_figures.py, and update_readme_results.py all import
get_weights_and_image() from HERE rather than each generating their own
copy.

This matters: if you swap in real trained/quantized weights (see README
Section 4.5) by editing only one of those four scripts, the other three
would keep silently reconstructing the old synthetic random weights from
the RNG seed, and every number they report would quietly desync from
what the RTL actually ran on. Editing only this one function keeps all
four scripts consistent automatically.
"""

import numpy as np
import config as cfg


def get_weights_and_image():
    """
    Returns (weights: dict, img: np.ndarray[IMG_H, IMG_W]).

    Default implementation: small-magnitude synthetic random INT8
    weights/biases and a random INT8-range image, from a fixed seed
    (cfg.RANDOM_SEED) so the RTL has a deterministic parameter set to
    compute and check against (see README, "Swapping in a trained model").

    TO USE REAL TRAINED WEIGHTS: replace the body of this function with
    code that loads your own trained-then-INT8-quantized conv1_w/
    conv1_b/conv2_w/conv2_b/fc1_w/fc1_b/fc2_w/fc2_b arrays (same shapes
    as below) and a real test image, instead of the rng.integers(...)
    calls. No other file needs to change.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    def rand_int8(shape):
        lo, hi = cfg.WEIGHT_RANGE
        return rng.integers(lo, hi + 1, size=shape, dtype=np.int32)

    def rand_bias(shape):
        lo, hi = cfg.BIAS_RANGE
        return rng.integers(lo, hi + 1, size=shape, dtype=np.int32)

    weights = {
        'conv1_w': rand_int8((cfg.C1_COUT, cfg.C1_CIN, cfg.C1_K, cfg.C1_K)),
        'conv1_b': rand_bias((cfg.C1_COUT,)),
        'conv2_w': rand_int8((cfg.C2_COUT, cfg.C2_CIN, cfg.C2_K, cfg.C2_K)),
        'conv2_b': rand_bias((cfg.C2_COUT,)),
        'fc1_w':   rand_int8((cfg.FC1_OUT, cfg.FC1_IN)),
        'fc1_b':   rand_bias((cfg.FC1_OUT,)),
        'fc2_w':   rand_int8((cfg.FC2_OUT, cfg.FC2_IN)),
        'fc2_b':   rand_bias((cfg.FC2_OUT,)),
    }

    img_lo, img_hi = cfg.IMG_RANGE
    img = rng.integers(img_lo, img_hi + 1, size=(cfg.IMG_H, cfg.IMG_W), dtype=np.int32)

    return weights, img

"""
generate_vectors.py

Generates a full, self-consistent set of test vectors for the accelerator:

  1. Weights/biases/image from weights_source.get_weights_and_image() --
     by default small-magnitude synthetic random INT8 values, NOT trained
     weights. See README "Swapping in a trained model" (and
     model/weights_source.py's own docstring) for how to swap in a real
     quantized model -- edit ONLY weights_source.py; every other script
     (including this one) picks up the change automatically.
  2. Runs golden_model.forward() to get the bit-exact expected classification
     scores.
  3. Writes every weight/bias/image array to $readmemh-compatible hex files
     under model/hex/, plus tb/expected_scores.hex + a human-readable
     model/summary.txt.

Run this from the model/ directory:
    python generate_vectors.py

Re-run any time you change config.py's dimensions/ranges -- and keep
rtl/cnn_pkg.vh's `SHIFT_* / dimension defines in sync if you do.
"""

import os
import numpy as np

import config as cfg
from golden_model import forward
from weights_source import get_weights_and_image

OUT_DIR = os.path.join(os.path.dirname(__file__), "hex")
TB_DIR  = os.path.join(os.path.dirname(__file__), "..", "tb")


def to_hex_lines(arr, bits):
    """Flatten (C-order) and format each element as two's-complement hex,
    one value per line, matching Verilog $readmemh expectations."""
    mask = (1 << bits) - 1
    hexdigits = bits // 4
    flat = arr.flatten()
    lines = []
    for v in flat:
        uv = int(v) & mask
        lines.append(f"{uv:0{hexdigits}x}")
    return lines


def write_hex(path, arr, bits):
    lines = to_hex_lines(arr, bits)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}  ({len(lines)} values, {bits}-bit)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    weights, img = get_weights_and_image()

    print("Running golden model forward pass...")
    scores, inter = forward(img, weights)
    pred_class = int(np.argmax(scores))

    print("\nExpected classification scores (raw int32):")
    for i, s in enumerate(scores):
        marker = "  <-- argmax" if i == pred_class else ""
        print(f"  class {i}: {s}{marker}")
    print(f"\nPredicted class: {pred_class}")

    print("\nWriting hex vector files...")
    write_hex(os.path.join(OUT_DIR, "conv1_w.hex"), weights['conv1_w'], cfg.WGT_BITS)
    write_hex(os.path.join(OUT_DIR, "conv1_b.hex"), weights['conv1_b'], cfg.ACC_BITS)
    write_hex(os.path.join(OUT_DIR, "conv2_w.hex"), weights['conv2_w'], cfg.WGT_BITS)
    write_hex(os.path.join(OUT_DIR, "conv2_b.hex"), weights['conv2_b'], cfg.ACC_BITS)
    write_hex(os.path.join(OUT_DIR, "fc1_w.hex"),   weights['fc1_w'],   cfg.WGT_BITS)
    write_hex(os.path.join(OUT_DIR, "fc1_b.hex"),   weights['fc1_b'],   cfg.ACC_BITS)
    write_hex(os.path.join(OUT_DIR, "fc2_w.hex"),   weights['fc2_w'],   cfg.WGT_BITS)
    write_hex(os.path.join(OUT_DIR, "fc2_b.hex"),   weights['fc2_b'],   cfg.ACC_BITS)

    write_hex(os.path.join(TB_DIR, "input_image.hex"), img, cfg.ACT_BITS)
    write_hex(os.path.join(TB_DIR, "expected_scores.hex"), scores, cfg.ACC_BITS)

    # Sparsity stats (for the "sparsity-aware accelerator" story in the report)
    def sparsity(a):
        return 100.0 * np.sum(a == 0) / a.size

    with open(os.path.join(os.path.dirname(__file__), "summary.txt"), "w") as f:
        f.write("Synthetic test-vector generation summary\n")
        f.write("=========================================\n")
        f.write(f"Random seed: {cfg.RANDOM_SEED}\n\n")
        f.write("Expected classification scores (raw int32):\n")
        for i, s in enumerate(scores):
            f.write(f"  class {i}: {s}\n")
        f.write(f"\nPredicted class (argmax): {pred_class}\n\n")
        f.write("Post-ReLU activation sparsity (fraction of exact zeros in\n")
        f.write("each layer's OUTPUT feature map, before the next pooling step):\n")
        f.write(f"  conv1 output : {sparsity(inter['conv1']):.1f}%\n")
        f.write(f"  conv2 output : {sparsity(inter['conv2']):.1f}%\n")
        f.write(f"  fc1   output : {sparsity(inter['fc1']):.1f}%\n")
        f.write(
            "\nNOTE -- these are NOT the same number as the hardware's\n"
            "PERF_SKIP-derived sparsity percentage. Max pooling means the\n"
            "next layer's MAC array sees fewer zeros than a layer's raw\n"
            "output has (a 2x2 window is zero only if all four inputs were\n"
            "zero), so the fraction of MAC operations actually zero-gated\n"
            "in hardware is generally LOWER than these raw output-zero\n"
            "fractions once pooling sits in between. For the exact,\n"
            "hardware-accurate MAC/skip counts (matching PERF_MACS/\n"
            "PERF_SKIP), run: python cycle_model.py\n"
            "\n"
            "Also: these are placeholder/random weights for RTL functional\n"
            "verification, not a trained classifier -- sparsity numbers here\n"
            "are illustrative of the zero-gating mechanism, not a claim about\n"
            "a real trained LeNet-5's sparsity.\n"
        )

    print("\nDone. See model/summary.txt for sparsity stats to quote in the report.")


if __name__ == "__main__":
    main()

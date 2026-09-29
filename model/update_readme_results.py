"""
update_readme_results.py

Reads sim/rtl_results.txt (written by tb/tb_cnn_top.v's $fwrite calls at
the end of simulation) and patches the block between the
<!-- RTL_RESULTS_START --> / <!-- RTL_RESULTS_END --> markers in
README.md with the real numbers -- automatically, so the README always
reflects the most recent simulation run rather than a stale hand-typed
table.

Also recomputes the analytical prediction fresh (via cycle_model.py, using
whatever weights/image are currently in model/hex + tb/*.hex) so the
cross-check error is always computed against the SAME test vectors the
simulation just ran on, not a stale hardcoded number.

Run from the model/ directory, AFTER sim/rtl_results.txt exists (i.e.
after sim/run_sim.bat / run_sim.sh has finished):
    python update_readme_results.py

This is normally called automatically by sim/run_sim.bat / run_sim.sh --
you shouldn't need to run it by hand.
"""

import os
import re
import sys

import numpy as np
import config as cfg
from golden_model import forward
from weights_source import get_weights_and_image
from cycle_model import conv_layer_cycles_and_macs, fc_layer_cycles_and_macs, pool_layer_cycles

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_FILE = os.path.join(ROOT, "sim", "rtl_results.txt")
README_FILE = os.path.join(ROOT, "README.md")

START_MARKER = "<!-- RTL_RESULTS_START -->"
END_MARKER = "<!-- RTL_RESULTS_END -->"


def parse_results_file(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            results[k.strip()] = v.strip()
    return results


def compute_analytical_prediction():
    """Re-derive the analytical cycle/MAC/skip prediction from
    weights_source.get_weights_and_image() -- the same single source of
    truth every other script uses -- so this always matches whatever
    test vectors the simulation actually ran on."""
    weights, img = get_weights_and_image()

    scores, inter = forward(img, weights)
    pred_class = int(np.argmax(scores))

    x0 = img.reshape(1, cfg.IMG_H, cfg.IMG_W)
    c1c, c1m, c1s, _ = conv_layer_cycles_and_macs(x0, weights['conv1_w'], cfg.NUM_PE)
    p1c = pool_layer_cycles(cfg.C1_COUT, cfg.P1_OUT_H, cfg.P1_OUT_W)
    c2c, c2m, c2s, _ = conv_layer_cycles_and_macs(inter['pool1'], weights['conv2_w'], cfg.NUM_PE)
    p2c = pool_layer_cycles(cfg.C2_COUT, cfg.P2_OUT_H, cfg.P2_OUT_W)
    flat = inter['pool2'].flatten()
    fc1c, fc1m, fc1s = fc_layer_cycles_and_macs(flat, weights['fc1_w'], cfg.NUM_PE)
    fc2c, fc2m, fc2s = fc_layer_cycles_and_macs(inter['fc1'], weights['fc2_w'], cfg.NUM_PE)

    load_cycles = cfg.IMG_H * cfg.IMG_W
    total_cycles = load_cycles + c1c + p1c + c2c + p2c + fc1c + fc2c
    total_mac = c1m + c2m + fc1m + fc2m
    total_skip = c1s + c2s + fc1s + fc2s

    return {
        "total_cycles": total_cycles,
        "total_mac": total_mac,
        "total_skip": total_skip,
        "pred_class": pred_class,
    }


def build_results_block(rtl, analytical):
    rtl_cycles = int(rtl.get("PERF_CYCLES", -1))
    rtl_macs = int(rtl.get("PERF_MACS", -1))
    rtl_skip = int(rtl.get("PERF_SKIP", -1))
    rtl_class = rtl.get("PREDICTED_CLASS", "?")

    model_cycles = analytical["total_cycles"]
    error_pct = 100.0 * abs(rtl_cycles - model_cycles) / model_cycles if model_cycles else float("nan")
    mac_total_rtl = rtl_macs + rtl_skip
    invariant_ok = (mac_total_rtl == 222000)

    lines = []
    lines.append("```")
    lines.append("RTL SIMULATION (from sim/rtl_results.txt, auto-generated)")
    lines.append("------------------------------------------------")
    lines.append(f"Expected class (golden model)  : {analytical['pred_class']}")
    lines.append(f"RTL predicted class            : {rtl_class}")
    lines.append(f"Run 1 vs golden model           : {rtl.get('RUN1_MATCH', '?')}")
    lines.append(f"Run 2 vs golden model           : {rtl.get('RUN2_MATCH', '?')}")
    lines.append(f"Run 2 == Run 1 (repeat-inference): {rtl.get('REPEAT_MATCH', '?')}")
    lines.append(f"PERF_MACS + PERF_SKIP == 222000 : {rtl.get('INVARIANT_MATCH', '?')} ({mac_total_rtl})")
    lines.append(f"PERF_CYCLES                     : {rtl_cycles}")
    lines.append(f"PERF_MACS                       : {rtl_macs}")
    lines.append(f"PERF_SKIP                       : {rtl_skip}")
    lines.append(f"OVERALL                         : {rtl.get('OVERALL', '?')}")
    lines.append("```")
    lines.append("")
    lines.append("| | Total cycles |")
    lines.append("|---|---|")
    lines.append(f"| Analytical model (model/cycle_model.py) | {model_cycles:,} |")
    lines.append(f"| RTL simulation (PERF_CYCLES) | {rtl_cycles:,} |")
    lines.append(f"| Error = \\|C_RTL - C_model\\| / C_model | {error_pct:.2f}% |")
    lines.append("")
    lines.append("The schedule model covers the per-layer MAC loops, writeback "
                  "and pooling cycles exactly. The residual difference is the "
                  "handshake and FIFO read latency between stages (Section 14.1).")

    return "\n".join(lines)


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found.")
        print("Run sim/run_sim.bat (or run_sim.sh) first -- it runs the RTL")
        print("simulation and writes this file before calling this script.")
        sys.exit(1)

    rtl = parse_results_file(RESULTS_FILE)
    analytical = compute_analytical_prediction()
    block = build_results_block(rtl, analytical)

    with open(README_FILE, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    if not pattern.search(content):
        print("ERROR: could not find RTL_RESULTS_START/END markers in README.md.")
        print("The results table may have been edited by hand -- restore the")
        print("markers (see README.md Section 13.3) and re-run this script.")
        sys.exit(1)

    new_content = pattern.sub(START_MARKER + "\n" + block + "\n" + END_MARKER, content)

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("Patched README.md Section 13.3 with results from sim/rtl_results.txt.")
    print(f"  PERF_CYCLES={rtl.get('PERF_CYCLES')}  (analytical prediction: {analytical['total_cycles']})")
    print(f"  OVERALL={rtl.get('OVERALL')}")


if __name__ == "__main__":
    main()

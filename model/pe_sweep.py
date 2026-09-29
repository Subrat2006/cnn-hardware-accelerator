"""
pe_sweep.py

Design-space sweep over the number of MAC lanes (NUM_PE). For every
candidate width it evaluates, from the deterministic tap schedule used by
the RTL controllers, the total cycle count, the average lane utilization,
and the silicon cost in MAC lanes (four layer engines, each with NUM_PE
lanes).

Run from the model/ directory:
    python pe_sweep.py
"""

import numpy as np
import config as cfg
from golden_model import forward
from weights_source import get_weights_and_image
from cycle_model import (conv_layer_cycles_and_macs, fc_layer_cycles_and_macs,
                         pool_layer_cycles)

PE_CANDIDATES = (2, 4, 8, 16)
CLOCK_MHZ = 150.0
N_ENGINES = 4


def sweep():
    weights, img = get_weights_and_image()
    _, inter = forward(img, weights)
    x0 = img.reshape(1, cfg.IMG_H, cfg.IMG_W)
    flat = inter['pool2'].flatten()

    rows = []
    for n in PE_CANDIDATES:
        c1, m1, s1, _ = conv_layer_cycles_and_macs(x0, weights['conv1_w'], n)
        c2, m2, s2, _ = conv_layer_cycles_and_macs(inter['pool1'], weights['conv2_w'], n)
        f1, fm1, fs1 = fc_layer_cycles_and_macs(flat, weights['fc1_w'], n)
        f2, fm2, fs2 = fc_layer_cycles_and_macs(inter['fc1'], weights['fc2_w'], n)
        p1 = pool_layer_cycles(cfg.C1_COUT, cfg.P1_OUT_H, cfg.P1_OUT_W)
        p2 = pool_layer_cycles(cfg.C2_COUT, cfg.P2_OUT_H, cfg.P2_OUT_W)
        load = cfg.IMG_H * cfg.IMG_W
        total = load + c1 + p1 + c2 + p2 + f1 + f2

        events = (m1 + s1) + (m2 + s2) + (fm1 + fs1) + (fm2 + fs2)
        # lane-slots offered during tap cycles = n lanes x tap cycles
        tap_cycles = (
            -(-cfg.C1_COUT // n) * cfg.C1_OUT_H * cfg.C1_OUT_W * cfg.C1_CIN * cfg.C1_K ** 2
            + -(-cfg.C2_COUT // n) * cfg.C2_OUT_H * cfg.C2_OUT_W * cfg.C2_CIN * cfg.C2_K ** 2
            + -(-cfg.FC1_OUT // n) * cfg.FC1_IN
            + -(-cfg.FC2_OUT // n) * cfg.FC2_IN
        )
        util = 100.0 * events / (n * tap_cycles)
        rows.append({
            "pe": n,
            "cycles": total,
            "latency_us": total / CLOCK_MHZ,
            "util": util,
            "lanes": n * N_ENGINES,
            "events": events,
        })
    base = next(r for r in rows if r["pe"] == 8)["cycles"]
    for r in rows:
        r["speedup_vs_8"] = base / r["cycles"]
        r["cycles_x_lanes"] = r["cycles"] * r["lanes"]
    return rows


def main():
    rows = sweep()
    print(f"{'PE':>3} {'cycles':>8} {'lat(us)':>9} {'util%':>7} {'lanes':>6} {'cyc*lanes':>10}")
    for r in rows:
        print(f"{r['pe']:>3} {r['cycles']:>8,} {r['latency_us']:>9.1f} "
              f"{r['util']:>7.1f} {r['lanes']:>6} {r['cycles_x_lanes']:>10,}")


if __name__ == "__main__":
    main()

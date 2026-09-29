"""
cycle_model.py

An ANALYTICAL MODEL of the accelerator's deterministic compute schedule
-- not a claim of RTL-exact cycle count. It predicts what the RTL's
PERF_MACS / PERF_SKIP registers should read (exactly), and what
PERF_CYCLES should read (closely, but see the caveat below), for a given
set of weights + input image, without running a Verilog simulator.

MAC/skip counts are exact, not approximate. This is possible because the
activation broadcast to every active PE lane in a group is the SAME value
each tap, so the real-vs-skipped decision is identical across every
active lane on that tap:

    per tap: (activation == 0) ? skip_count += active_lanes
                                : mac_count  += active_lanes

which this script computes directly from the same intermediate feature
maps golden_model.py already produces, using the exact same nested-loop
order (group -> spatial position -> tap) the RTL's control FSM uses. The
TOTAL of executed + skipped MAC-lane events is additionally a hard,
data-independent architectural constant (222,000 for this network's
dimensions and PE count) -- see tb/tb_cnn_top.v's invariant check.

Cycle count is a close analytical approximation, not an exact model. The
per-layer MAC-loop cycle counts computed here ARE exact (one cycle per
tap plus one cycle per output-channel writeback, per the FSMs in
conv_layer_engine.v / fc_layer_engine.v / pool_engine.v). What this script
does NOT model precisely is (a) the input-load phase's exact pipeline
latency through the loader FSM in cnn_core.v (approximated here as
IMG_H*IMG_W cycles, when it may be a small constant more due to the
async-FIFO's 1-cycle read latency), and (b) any small, fixed handshake/
bubble cycles the top-level sequencer spends transitioning between layer
engines. Expect the real PERF_CYCLES to be within a small, bounded number
of cycles of this prediction -- not necessarily bit-identical -- until
you've actually measured it. See README.md Section 14 for how
model/update_readme_results.py computes and reports this gap once you
have a real simulation run.

Run from the model/ directory, after generate_vectors.py:
    python cycle_model.py
"""

import numpy as np
import config as cfg
from golden_model import forward
from weights_source import get_weights_and_image


def conv_layer_cycles_and_macs(x, w, num_pe):
    """
    x: (Cin, IN_H, IN_W) -- the layer's INPUT feature map (already computed
       by the golden model for the previous layer)
    w: (Cout, Cin, K, K) -- this layer's weights (only used for shape; the
       zero/nonzero decision depends only on the ACTIVATION, per the
       hardware's zero-gating rule -- see rtl/mac_unit.v)
    Returns: (cycles, mac_ops, skip_ops)
    """
    Cout, Cin, K, _ = w.shape
    _, IN_H, IN_W = x.shape
    OUT_H, OUT_W = IN_H - K + 1, IN_W - K + 1
    NTAPS = Cin * K * K
    NUM_GROUPS = -(-Cout // num_pe)  # ceil

    cycles = 0
    mac_ops = 0
    skip_ops = 0

    for g in range(NUM_GROUPS):
        active = min(num_pe, Cout - g * num_pe)
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                cycles += NTAPS      # one cycle per tap (S_MAC)
                cycles += active     # one cycle per channel written back (S_WRITE)
                for cin in range(Cin):
                    for ky in range(K):
                        for kx in range(K):
                            act = int(x[cin, oy + ky, ox + kx])
                            if act == 0:
                                skip_ops += active
                            else:
                                mac_ops += active

    return cycles, mac_ops, skip_ops, (OUT_H, OUT_W)


def pool_layer_cycles(ch, out_h, out_w):
    return ch * out_h * out_w * 5  # 5 cycles/window, see rtl/pool_engine.v


def fc_layer_cycles_and_macs(x, w, num_pe):
    """
    x: (N_in,) -- this layer's input activation vector
    w: (N_out, N_in)
    """
    N_out, N_in = w.shape
    NUM_GROUPS = -(-N_out // num_pe)

    cycles = 0
    mac_ops = 0
    skip_ops = 0

    for g in range(NUM_GROUPS):
        active = min(num_pe, N_out - g * num_pe)
        cycles += N_in    # one cycle per tap
        cycles += active  # one cycle per neuron written back
        for t in range(N_in):
            act = int(x[t])
            if act == 0:
                skip_ops += active
            else:
                mac_ops += active

    return cycles, mac_ops, skip_ops


def main():
    weights, img = get_weights_and_image()

    scores, inter = forward(img, weights)

    x0 = img.reshape(1, cfg.IMG_H, cfg.IMG_W)

    c1_cycles, c1_mac, c1_skip, (c1_oh, c1_ow) = conv_layer_cycles_and_macs(
        x0, weights['conv1_w'], cfg.NUM_PE)
    p1_cycles = pool_layer_cycles(cfg.C1_COUT, cfg.P1_OUT_H, cfg.P1_OUT_W)

    c2_cycles, c2_mac, c2_skip, (c2_oh, c2_ow) = conv_layer_cycles_and_macs(
        inter['pool1'], weights['conv2_w'], cfg.NUM_PE)
    p2_cycles = pool_layer_cycles(cfg.C2_COUT, cfg.P2_OUT_H, cfg.P2_OUT_W)

    flat = inter['pool2'].flatten()
    fc1_cycles, fc1_mac, fc1_skip = fc_layer_cycles_and_macs(flat, weights['fc1_w'], cfg.NUM_PE)
    fc2_cycles, fc2_mac, fc2_skip = fc_layer_cycles_and_macs(inter['fc1'], weights['fc2_w'], cfg.NUM_PE)

    load_cycles = cfg.IMG_H * cfg.IMG_W  # 784, approximate (see note below)

    total_cycles = (load_cycles + c1_cycles + p1_cycles + c2_cycles
                     + p2_cycles + fc1_cycles + fc2_cycles)
    total_mac  = c1_mac  + c2_mac  + fc1_mac  + fc2_mac
    total_skip = c1_skip + c2_skip + fc1_skip + fc2_skip
    sparsity = 100.0 * total_skip / (total_mac + total_skip)

    core_freq_mhz = 150.0  # matches tb_cnn_top.v's core_clk generation
    latency_us = total_cycles / (core_freq_mhz * 1.0)  # cycles / MHz = microseconds

    mem_bytes = (
        784                                            # img_mem
        + cfg.C1_COUT * c1_oh * c1_ow                  # conv1_out_mem
        + cfg.C1_COUT * cfg.P1_OUT_H * cfg.P1_OUT_W    # pool1_out_mem
        + cfg.C2_COUT * c2_oh * c2_ow                  # conv2_out_mem
        + cfg.C2_COUT * cfg.P2_OUT_H * cfg.P2_OUT_W    # pool2_out_mem / fc1_in
        + cfg.FC1_OUT                                  # fc1_out_mem
        + 10 * 4                                        # score regs (32b x10)
        + (cfg.C1_COUT * cfg.C1_CIN * cfg.C1_K * cfg.C1_K)       # conv1 weights
        + (cfg.C2_COUT * cfg.C2_CIN * cfg.C2_K * cfg.C2_K)       # conv2 weights
        + (cfg.FC1_OUT * cfg.FC1_IN)                    # fc1 weights
        + (cfg.FC2_OUT * cfg.FC2_IN)                    # fc2 weights
        + (cfg.C1_COUT + cfg.C2_COUT + cfg.FC1_OUT + cfg.FC2_OUT) * 4  # biases (32b)
    )

    print("=" * 65)
    print(" Analytically predicted PERF_* results (see cycle_model.py)")
    print(" -- cross-check these against a real RTL simulation run.")
    print("=" * 65)
    print(f" Load (input stream)      : {load_cycles:>8} cycles  (approx., see note)")
    print(f" Conv1                    : {c1_cycles:>8} cycles   MAC={c1_mac:<7} skip={c1_skip}")
    print(f" Pool1                    : {p1_cycles:>8} cycles")
    print(f" Conv2                    : {c2_cycles:>8} cycles   MAC={c2_mac:<7} skip={c2_skip}")
    print(f" Pool2                    : {p2_cycles:>8} cycles")
    print(f" FC1                      : {fc1_cycles:>8} cycles   MAC={fc1_mac:<7} skip={fc1_skip}")
    print(f" FC2                      : {fc2_cycles:>8} cycles   MAC={fc2_mac:<7} skip={fc2_skip}")
    print("-" * 65)
    print(f" TOTAL CYCLES (analytical)  : {total_cycles}  (see header docstring for caveats)")
    print(f" TOTAL executed MACs        : {total_mac}")
    print(f" TOTAL zero-gated (skip)    : {total_skip}")
    print(f" TOTAL MAC-lane events      : {total_mac + total_skip}  (invariant: must equal 222000)")
    print(f" Sparsity (skip %)          : {sparsity:.1f}%")
    print(f" Core clock (testbench)     : {core_freq_mhz:.0f} MHz")
    print(f" Predicted latency          : {latency_us:.2f} us")
    print(f" Logical storage footprint  : {mem_bytes} bytes (~{mem_bytes/1024:.1f} KB) -- NOT a physical FPGA resource estimate, see README Section 8.4")
    print("=" * 65)
    print(" NOTE on 'Load' cycles: this is an approximation (one core-clock")
    print(" cycle per pixel), ignoring the async-FIFO's fixed 1-cycle read")
    print(" latency and clock-ratio effects. It is a small, bounded additive")
    print(" constant next to the ~40k+ cycles spent in conv/FC, and does not")
    print(" affect the MAC/skip counts at all -- only report total latency")
    print(" as approximate until you've read PERF_CYCLES back from a real")
    print(" simulation run.")


if __name__ == "__main__":
    main()

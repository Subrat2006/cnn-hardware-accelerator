#!/usr/bin/env bash
# Run from the project ROOT directory:  bash sim/run_sim.sh
#
# Full pipeline, one command:
#   1. Regenerate weight/bias/image/expected-score hex vectors
#   2. Compile every RTL file + the testbench with Icarus Verilog
#   3. Run the simulation (writes sim/rtl_results.txt)
#   4. Auto-patch README.md's results section (Section 13.3) with the real
#      PERF_CYCLES/PERF_MACS/PERF_SKIP/PASS-FAIL numbers just produced
#   5. Regenerate every figure under docs/images/ so the README's charts
#      also reflect the current test vectors
#
# After this finishes, README.md IS the up-to-date report -- no manual
# copy-pasting of numbers required.
set -e

cd "$(dirname "$0")/.."   # move to project root regardless of cwd

mkdir -p sim

echo "== [1/5] Regenerating test vectors (weights/bias/image + golden scores) =="
(cd model && python3 generate_vectors.py)

echo "== [2/5] Compiling RTL + testbench with Icarus Verilog =="
iverilog -g2012 -o sim/sim.out -I rtl \
    rtl/mac_unit.v \
    rtl/pe_array.v \
    rtl/sync_ram.v \
    rtl/weight_rom.v \
    rtl/relu_quant.v \
    rtl/max_pool2x2.v \
    rtl/pool_engine.v \
    rtl/conv_layer_engine.v \
    rtl/fc_layer_engine.v \
    rtl/perf_counters.v \
    rtl/cdc_sync_2ff.v \
    rtl/cdc_pulse_sync.v \
    rtl/async_fifo.v \
    rtl/axi4_lite_slave.v \
    rtl/cnn_core.v \
    rtl/cnn_top.v \
    tb/tb_cnn_top.v

echo "== [3/5] Running simulation (this also writes sim/rtl_results.txt) =="
echo "   This prints progress every ~196 image bytes and every ~200 status"
echo "   polls. A quiet gap of up to a minute or two between those lines is"
echo "   normal (Icarus Verilog is an interpreter, not a compiled simulator)."
echo "   If it looks fully stuck for much longer, see README.md Section 4.4"
echo "   'Troubleshooting' -- the testbench will print a diagnostic and stop"
echo "   on its own rather than hang forever."
vvp sim/sim.out

echo "== [4/5] Patching README.md Section 13.3 with the real results =="
if [ -f sim/rtl_results.txt ]; then
    (cd model && python3 update_readme_results.py)
else
    echo "WARNING: sim/rtl_results.txt was not created -- the simulation"
    echo "likely hit the safety timeout or crashed before finishing. See"
    echo "the vvp output above. Skipping the README patch step."
fi

echo "== [5/5] Regenerating figures under docs/images/ =="
(cd model && python3 generate_figures.py)

echo ""
echo "Done. README.md now reflects this run's actual results and figures."

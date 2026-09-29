@echo off
REM Run from the project ROOT directory:  sim\run_sim.bat
REM Requires Icarus Verilog (iverilog + vvp) on PATH -- see README.md
REM Section 4 "Build, simulate, reproduce".
REM Requires Python 3 with numpy + matplotlib on PATH.
REM
REM Full pipeline, one command:
REM   1. Regenerate weight/bias/image/expected-score hex vectors
REM   2. Compile every RTL file + the testbench with Icarus Verilog
REM   3. Run the simulation (writes sim\rtl_results.txt)
REM   4. Auto-patch README.md's results section (Section 13.3) with the
REM      real PERF_CYCLES/PERF_MACS/PERF_SKIP/PASS-FAIL numbers just
REM      produced
REM   5. Regenerate every figure under docs\images\ so the README's
REM      charts also reflect the current test vectors
REM
REM After this finishes, README.md IS the up-to-date report -- no manual
REM copy-pasting of numbers required.

cd /d "%~dp0\.."

echo == [1/5] Regenerating test vectors (weights/bias/image + golden scores) ==
pushd model
python generate_vectors.py
if errorlevel 1 goto :error
popd

if not exist sim mkdir sim

echo == [2/5] Compiling RTL + testbench with Icarus Verilog ==
iverilog -g2012 -o sim\sim.out -I rtl ^
    rtl\mac_unit.v ^
    rtl\pe_array.v ^
    rtl\sync_ram.v ^
    rtl\weight_rom.v ^
    rtl\relu_quant.v ^
    rtl\max_pool2x2.v ^
    rtl\pool_engine.v ^
    rtl\conv_layer_engine.v ^
    rtl\fc_layer_engine.v ^
    rtl\perf_counters.v ^
    rtl\cdc_sync_2ff.v ^
    rtl\cdc_pulse_sync.v ^
    rtl\async_fifo.v ^
    rtl\axi4_lite_slave.v ^
    rtl\cnn_core.v ^
    rtl\cnn_top.v ^
    tb\tb_cnn_top.v
if errorlevel 1 goto :error

echo == [3/5] Running simulation (this also writes sim\rtl_results.txt) ==
echo    This prints progress every ~196 image bytes and every ~200 status
echo    polls. A quiet gap of up to a minute or two between those lines is
echo    normal (Icarus Verilog is an interpreter, not a compiled simulator).
echo    If it looks fully stuck for much longer, see README.md Section 4.4
echo    "Troubleshooting" -- the testbench will print a diagnostic and stop
echo    on its own rather than hang forever.
vvp sim\sim.out
if errorlevel 1 goto :error

echo == [4/5] Patching README.md Section 13.3 with the real results ==
if exist sim\rtl_results.txt (
    pushd model
    python update_readme_results.py
    popd
) else (
    echo WARNING: sim\rtl_results.txt was not created -- the simulation
    echo likely hit the safety timeout or crashed before finishing. See
    echo the vvp output above. Skipping the README patch step.
)

echo == [5/5] Regenerating figures under docs\images\ ==
pushd model
python generate_figures.py
popd

echo.
echo Done. README.md now reflects this run's actual results and figures.
goto :eof

:error
echo.
echo Build/run FAILED -- see the error above.
exit /b 1

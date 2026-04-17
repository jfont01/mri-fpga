#!/usr/bin/env bash
set -euo pipefail

CASE="Aij"

SYNTH_PATH="$FPGA_MRI_ROOT/rtl/synthesis/synth_$CASE"
TCL_PATH="$SYNTH_PATH/synth_compute_$CASE.tcl"

LOG_DIR="$SYNTH_PATH/logs"
mkdir -p "$LOG_DIR"

TCL_WIN=$(wslpath -w "$TCL_PATH")
LOG_WIN=$(wslpath -w "$LOG_DIR/vivado.log")
JOU_WIN=$(wslpath -w "$LOG_DIR/vivado.jou")

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch -source '$TCL_WIN' -log '$LOG_WIN' -journal '$JOU_WIN'"
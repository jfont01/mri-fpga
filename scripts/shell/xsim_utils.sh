#!/usr/bin/env bash

# Este archivo debe ser sourced desde set_env.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[xsim_utils.sh] ERROR: this file must be sourced, not executed." >&2
    exit 1
fi

run_xsim() {
    local case_dir="${1:-build}"
    local launch_dir
    local vivado_log_dir
    local vivado_journal
    local vivado_log
    local stamp

    shift || true
    launch_dir="$(pwd -P)"

    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[xsim_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi

    if [[ ! -f "$RUN_XSIM_TCL" ]]; then
        echo "[xsim_utils.sh] ERROR: missing run_xsim.tcl: $RUN_XSIM_TCL" >&2
        return 1
    fi

    if ! command -v vivado >/dev/null 2>&1; then
        echo "[xsim_utils.sh] ERROR: vivado command not found in PATH." >&2
        return 1
    fi

    vivado_log_dir="$PROJECT_ROOT/build/vivado_logs"
    mkdir -p "$vivado_log_dir"

    stamp="$(date +%Y%m%d_%H%M%S)_$$"
    vivado_journal="$vivado_log_dir/run_xsim_${stamp}.jou"
    vivado_log="$vivado_log_dir/run_xsim_${stamp}.log"

    vivado  -mode batch                                      \
            -notrace                                         \
            -journal "$vivado_journal"                       \
            -log "$vivado_log"                               \
            -source "$RUN_XSIM_TCL"     \
            -tclargs "$case_dir" "$launch_dir" "$@"
}
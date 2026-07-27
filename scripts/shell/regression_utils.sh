#!/usr/bin/env bash

# Este archivo debe ser sourced desde set_env.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[regression_utils.sh] ERROR: this file must be sourced, not executed." >&2
    exit 1
fi

run_regression_vm() {
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[regression_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi

    if [[ ! -f "$RUN_REGRESSION_VM_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_regression_vm.py: $RUN_REGRESSION_VM_PY" >&2
        return 1
    fi

    update_flist
    python3 "$RUN_REGRESSION_VM_PY" "$@"
}

run_regression_sim() {
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[regression_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi

    if [[ ! -f "$RUN_REGRESSION_SIM_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_regression_sim.py: $RUN_REGRESSION_SIM_PY" >&2
        return 1
    fi

    python3 "$RUN_REGRESSION_SIM_PY" "$@"
}

run_regression_synth() {

    if [[ ! -f "$RUN_REGRESSION_SYNTH_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_regression_synth.py: $RUN_REGRESSION_SYNTH_PY" >&2
        return 1
    fi

    update_flist
    python3 "$RUN_REGRESSION_SYNTH_PY" "$@"
}

run_regression_impl() {
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[regression_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi

    if [[ ! -f "$RUN_REGRESSION_IMPL_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_regression_impl.py: $RUN_REGRESSION_IMPL_PY" >&2
        return 1
    fi

    update_flist
    python3 "$RUN_REGRESSION_IMPL_PY" "$@"
}

run_compile_rtl() {
    if [[ ! -f "$RUN_COMPILE_RTL_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_compile_rtl.py" >&2
        return 1
    fi
    python3 "$RUN_COMPILE_RTL_PY" "$@"
}

run_lint() {
    if [[ ! -f "$RUN_LINT_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_lint.py" >&2
        return 1
    fi
    python3 "$RUN_LINT_PY" "$@"
}

run_gtest() {
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[regression_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi
 
    if [[ ! -f "$RUN_GTEST_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_gtest.py: $RUN_GTEST_PY" >&2
        return 1
    fi
 
    python3 "$RUN_GTEST_PY" "$@"
}

run_plot_fft1d () {
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        echo "[regression_utils.sh] ERROR: PROJECT_ROOT is not defined." >&2
        return 1
    fi
 
    if [[ ! -f "$RUN_PLOT_FFT1D_PY" ]]; then
        echo "[regression_utils.sh] ERROR: missing run_plot_fft1d.py: $RUN_PLOT_FFT1D_PY" >&2
        return 1
    fi

    python3 "$RUN_PLOT_FFT1D_PY" "$@"
}
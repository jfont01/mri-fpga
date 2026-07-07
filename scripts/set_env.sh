#!/usr/bin/env bash

# Este archivo debe ser sourced:
#   source set_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[set_env.sh] ERROR: use: source set_env.sh" >&2
    exit 1
fi

if [[ -z "${NEW_ARCH_TEST_DIR:-}" ]]; then
    echo "[set_env.sh] ERROR: NEW_ARCH_TEST_DIR is not defined." >&2
    return 1
fi



export PROJECT_ROOT="$NEW_ARCH_TEST_DIR"
export MODULES_ROOT="$PROJECT_ROOT/modules"
export FXP_MODEL_ROOT="$PROJECT_ROOT/model"
export RTLSIM_ROOT="$PROJECT_ROOT/rtlsim"

source "$PROJECT_ROOT/.venv/bin/activate"

export SCRIPTS_DIR="$PROJECT_ROOT/scripts"
export PYTHON_SCRIPTS_DIR="$SCRIPTS_DIR/python"
export TCL_SCRIPTS_DIR="$SCRIPTS_DIR/tcl"
export SHELL_SCRIPTS_DIR="$SCRIPTS_DIR/shell"

export CONSTRAINTS_UTILS_SH="$SHELL_SCRIPTS_DIR/constraint_utils.sh"
export FLIST_UTILS_SH="$SHELL_SCRIPTS_DIR/flist_utils.sh"
export MODULE_UTILS_SH="$SHELL_SCRIPTS_DIR/module_utils.sh"
export REGRESSION_UTILS_SH="$SHELL_SCRIPTS_DIR/regression_utils.sh"
export XSIM_UTILS_SH="$SHELL_SCRIPTS_DIR/xsim_utils.sh"

export RUN_XSIM_TCL="$TCL_SCRIPTS_DIR/run_xsim.tcl"
export RUN_SYNTH_TCL="$TCL_SCRIPTS_DIR/run_synth.tcl"
export RUN_IMPL_TCL="$TCL_SCRIPTS_DIR/run_impl.tcl"

export RUN_REGRESSION_VM_PY="$PYTHON_SCRIPTS_DIR/run_regression_vm.py"
export RUN_REGRESSION_SYNTH_PY="$PYTHON_SCRIPTS_DIR/run_regression_synth.py"
export RUN_REGRESSION_IMPL_PY="$PYTHON_SCRIPTS_DIR/run_regression_impl.py"
export RUN_REGRESSION_SIM_PY="$PYTHON_SCRIPTS_DIR/run_regression_sim.py"

export MAKEFILE="$SCRIPTS_DIR/Makefile"

export VIVADO_ROOT="/tools/Xilinx/Vivado"
export VIVADO_VERSION="2024.2"
export VIVADO_BIN="$VIVADO_ROOT/$VIVADO_VERSION/bin/vivado"
export VIVADO_SETTINGS_64_SH="$VIVADO_ROOT/$VIVADO_VERSION/settings64.sh"

source $VIVADO_SETTINGS_64_SH

log() {
    echo "[set_env.sh] $*"
}

source_project_script() {
    local script_path="$1"

    if [[ ! -f "$script_path" ]]; then
        echo "[set_env.sh] ERROR: required script not found: $script_path" >&2
        return 1
    fi

    source "$script_path"
}

help_env() {
    cat <<EOF
[set_env.sh] Available commands

Environment:
  PROJECT_ROOT    = $PROJECT_ROOT
  MODULES_ROOT    = $MODULES_ROOT
  FXP_MODEL_ROOT  = $FXP_MODEL_ROOT

Module management:
  create_module <module_name>
      Create a new module workspace under:
        \$MODULES_ROOT/<module_name>

  refresh_module_vars <module_name>
      Regenerate <module_name>_vars.sh using the current infrastructure layout.

  cd_module <module_name>
      Go to a module directory.

  list_modules
      List available modules.

Build helpers:
  update_flist
      Regenerate tb/synth/impl flists for the current module.

Simulation:
  run_xsim [case_dir] [DEFINES...]
      Run Vivado XSIM for the current module.

  run_regression_vm [options]
      Run simulation regression cases from:
        testbench/<module_name>_vm_regression.json

      Expected default build output:
        build/simulation/<CASE>/

      Examples:
        run_regression_vm --list-cases
        run_regression_vm --keep-going

Synthesis:
  run_regression_synth [options]
      Run synthesis regression cases from:
        synthesis/<module_name>_synth_regression.json

      Expected default build output:
        build/synthesis/<CASE>/

      Examples:
        run_regression_synth --list-cases
        run_regression_synth --keep-going

Generated module structure:
  modules/<module_name>/
    build/
      simulation/
      synthesis/
      implementation/
    flist/
    py/
    rtl/
    testbench/
      <module_name>_tb.sv
      <module_name>_regression.json
    synthesis/
      <module_name>_wrapper_synth.v
      <module_name>_synth_regression.json
    implementation/
      <module_name>_wrapper_impl.v
      <module_name>_impl_regression.json
    <module_name>_vars.sh

Deprecated in the new layout:
  modules/<module_name>/wrappers/
  modules/<module_name>/constraints/
  modules/<module_name>/<module_name>.conf

Example:
  create_module kernel_Aij

EOF
}

mkdir -p "$MODULES_ROOT"

source "$MODULE_UTILS_SH"

source_module_vars

source_project_script "$CONSTRAINTS_UTILS_SH" || return 1
source_project_script "$FLIST_UTILS_SH" || return 1
source_project_script "$MODULE_UTILS_SH" || return 1
source_project_script "$REGRESSION_UTILS_SH" || return 1
source_project_script "$XSIM_UTILS_SH" || return 1


log "environment loaded"
log "PROJECT_ROOT=$PROJECT_ROOT"
log "MODULES_ROOT=$MODULES_ROOT"
log "FXP_MODEL_ROOT=$FXP_MODEL_ROOT"
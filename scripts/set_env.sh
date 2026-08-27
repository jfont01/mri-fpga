# ============================== Common Functions ==============================
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

load_settings()
{
    local name="$1"
    local settings_file="$2"

    if [[ -z "$settings_file" ]]; then
        log "$name settings not configured"
        return 0
    fi

    if [[ ! -f "$settings_file" ]]; then
        echo "[set_env.sh] ERROR: $name settings file not found:" >&2
        echo "[set_env.sh]        $settings_file" >&2
        return 1
    fi

    source "$settings_file" || return 1

    log "$name environment loaded"
}
# ============================== Variables Checks ==============================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[set_env.sh] ERROR: use: source set_env.sh" >&2
    exit 1
fi

if [[ -z "${FPGA_MRI_ROOT:-}" ]]; then
    echo "[set_env.sh] ERROR: FPGA_MRI_ROOT is not defined." >&2
    return 1
fi

# ============================== Roots Paths ==============================
export PROJECT_ROOT="$FPGA_MRI_ROOT"
export MODULES_ROOT="$PROJECT_ROOT/modules"
export FXP_MODEL_ROOT="$PROJECT_ROOT/model"
export RTLSIM_ROOT="$PROJECT_ROOT/rtlsim"

# ============================== Dirs Paths ==============================
export SCRIPTS_DIR="$PROJECT_ROOT/scripts"
export PYTHON_SCRIPTS_DIR="$SCRIPTS_DIR/python"
export TCL_SCRIPTS_DIR="$SCRIPTS_DIR/tcl"
export SHELL_SCRIPTS_DIR="$SCRIPTS_DIR/shell"

# ============================== Shell Scripts ==============================
export CONSTRAINTS_UTILS_SH="$SHELL_SCRIPTS_DIR/constraint_utils.sh"
export FLIST_UTILS_SH="$SHELL_SCRIPTS_DIR/flist_utils.sh"
export MODULE_UTILS_SH="$SHELL_SCRIPTS_DIR/module_utils.sh"
export REGRESSION_UTILS_SH="$SHELL_SCRIPTS_DIR/regression_utils.sh"
export XSIM_UTILS_SH="$SHELL_SCRIPTS_DIR/xsim_utils.sh"
export PLATFORM_UTILS_SH="$SHELL_SCRIPTS_DIR/platform_utils.sh"
export IVERILOG_UTILS_SH="$SHELL_SCRIPTS_DIR/iverilog_utils.sh"

# ============================== TCL Scripts ==============================
export RUN_XSIM_TCL="$TCL_SCRIPTS_DIR/run_xsim.tcl"
export RUN_SYNTH_TCL="$TCL_SCRIPTS_DIR/run_synth.tcl"
export RUN_IMPL_TCL="$TCL_SCRIPTS_DIR/run_impl.tcl"

# ============================== Python Scripts ==============================
export RUN_REGRESSION_VM_PY="$PYTHON_SCRIPTS_DIR/run_regression_vm.py"
export RUN_REGRESSION_SYNTH_PY="$PYTHON_SCRIPTS_DIR/run_regression_synth.py"
export RUN_REGRESSION_IMPL_PY="$PYTHON_SCRIPTS_DIR/run_regression_impl.py"
export RUN_REGRESSION_SIM_PY="$PYTHON_SCRIPTS_DIR/run_regression_sim.py"
export RUN_COMPILE_RTL_PY="$PYTHON_SCRIPTS_DIR/run_compile_rtl.py"
export RUN_LINT_PY="$PYTHON_SCRIPTS_DIR/run_lint.py"
export RUN_GTEST_PY="$PYTHON_SCRIPTS_DIR/run_gtest.py"
export RUN_PLOT_FFT1D_PY="$PYTHON_SCRIPTS_DIR/run_plot_fft1d.py"


# ============================== Makefile ==============================
export MAKEFILE="$SCRIPTS_DIR/Makefile"

# ============================== Platform detect ==============================
source_project_script "$PLATFORM_UTILS_SH" || return 1
export PLATFORM="$(detect_platform)"
export SIM_BACKEND="$(select_sim_backend)"

if [[ "$PLATFORM" == "linux" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    log "venv activated in Linux"
else
    log "using MSYS Python packages installed with"
fi 
# ============================== AMD/Xilinx Tools ==============================

if platform_has_vivado; then

    load_settings "Vivado" "${VIVADO_SETTINGS_64_SH:-}" || return 1
    load_settings "Vitis"  "${VITIS_SETTINGS_64_SH:-}"  || return 1

else

    log "AMD/Xilinx tools unavailable on platform '$PLATFORM'"
    log "simulation backend: $SIM_BACKEND"

fi

# ============================== Platform detect ==============================
mkdir -p "$MODULES_ROOT"

source_project_script "$MODULE_UTILS_SH" || return 1
source_module_vars
source_project_script "$CONSTRAINTS_UTILS_SH" || return 1
source_project_script "$FLIST_UTILS_SH" || return 1
source_project_script "$REGRESSION_UTILS_SH" || return 1
if platform_has_vivado; then
    source_project_script "$XSIM_UTILS_SH" || return 1
else
    source_project_script "$IVERILOG_UTILS_SH" || return 1
fi

# ============================== Echos ==============================
log "environment loaded"
log "PROJECT_ROOT=$PROJECT_ROOT"
log "MODULES_ROOT=$MODULES_ROOT"
log "FXP_MODEL_ROOT=$FXP_MODEL_ROOT"


# ============================== Help function ==============================
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

Unit tests (modelo C++):
  run_gtest [options]
      Corre Google Test sobre el modelo, en modo fixed y double.
      Casos desde: testbench/<module>_gtest_regression.json
      
      Outputs: build/<case>_<mode>/reports/gtest.{rpt,xml}
      
      Examples:
        run_gtest --list-cases
        run_gtest --mode double
        run_gtest --case q2_14 --gtest-filter 'CmulModel.Overflow*'
        run_gtest --keep-going

Compile checks:
  run_compile_rtl [--testbench]
      Compila y elabora el RTL del módulo actual con los DEFAULTS de los
      parámetros (sin defines). No simula ni necesita vectores.

      --testbench / -t
          Incluye además el testbench y elabora con top = <module>_tb.
          Sirve para cazar errores de sintaxis del .sv sin correr el vm.
          
      Ejemplos:
        run_compile_rtl                # solo el RTL
        run_compile_rtl --testbench    # RTL + testbench

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
  create_module fft2d

EOF
}
# scripts/env/project_paths.sh
# Debe ser usado con: source scripts/env/project_paths.sh

# ==============================================================================
# Guards
# ==============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[project_paths.sh] ERROR: this script must be sourced, not executed."
  echo "[project_paths.sh] Use: source scripts/env/project_paths.sh"
  exit 1
fi

# ==============================================================================
# Colors, only if not already defined
# ==============================================================================
RED="${RED:-$'\033[0;31m'}"
GREEN="${GREEN:-$'\033[0;32m'}"
YELLOW="${YELLOW:-$'\033[1;33m'}"
CYAN="${CYAN:-$'\033[0;36m'}"
NC="${NC:-$'\033[0m'}"

printf "\n"
printf "[project_paths.sh] ${GREEN}Loading project paths...${NC}\n"
printf "\n"

# ==============================================================================
# Root
# ==============================================================================
if [[ -z "${FPGA_MRI_ROOT:-}" ]]; then
  printf "[project_paths.sh] ${RED}ERROR:${NC} FPGA_MRI_ROOT is not defined\n"
  return 1
fi

if [[ ! -d "$FPGA_MRI_ROOT" ]]; then
  printf "[project_paths.sh] ${RED}ERROR:${NC} FPGA_MRI_ROOT does not exist: %s\n" "$FPGA_MRI_ROOT"
  return 1
fi

export FPGA_MRI_ROOT
FPGA_MRI_ROOT="$(cd "$FPGA_MRI_ROOT" && pwd)"
export FPGA_MRI_ROOT

# ==============================================================================
# Global config
# ==============================================================================
export GLOBAL_CONFIG_CONF="$FPGA_MRI_ROOT/global_config.conf"

# Backward compatibility
export GLOBAL_CONF_PATH="$GLOBAL_CONFIG_CONF"
export TRACK_CONF="$GLOBAL_CONFIG_CONF"
# ==============================================================================
# Tracks
# ==============================================================================
export TRACK_ROOT="$FPGA_MRI_ROOT/tracks"
#export TRACK_CONF="$TRACK_ROOT/track.conf"

# ==============================================================================
# RTL paths
# ==============================================================================
export RTL_ROOT="$FPGA_MRI_ROOT/rtl"
export RTL_CONSTRAINTS_DIR="$RTL_ROOT/constraints"
export RTL_SRC_DIR="$RTL_ROOT/src"
export RTL_TESTBENCH_DIR="$RTL_ROOT/testbench"

# RTL source files
export RTL_CAST_SV="$RTL_SRC_DIR/ops/cast.sv"
export RTL_CMUL_SV="$RTL_SRC_DIR/ops/cmul.sv"
export RTL_CSUM_SV="$RTL_SRC_DIR/ops/csum.sv"
export RTL_CSUB_SV="$RTL_SRC_DIR/ops/csub.sv"
export RTL_DIV_RESTORING_SV="$RTL_SRC_DIR/ops/div_restoring.sv"
export RTL_COMPUTE_AIJ_SV="$RTL_SRC_DIR/sense/compute_Aij.sv"
export RTL_COMPUTE_BI_SV="$RTL_SRC_DIR/sense/compute_bi.sv"

# RTL testbenches
export RTL_TB_COMPUTE_AIJ_SV="$RTL_TESTBENCH_DIR/sense/tb_compute_Aij.sv"
export RTL_TB_COMPUTE_BI_SV="$RTL_TESTBENCH_DIR/sense/tb_compute_bi.sv"
export RTL_TB_DIV_RESTORING_SV="$RTL_TESTBENCH_DIR/ops/tb_div_restoring.sv"

# RTL constraints
export RTL_CLOCK_AIJ_XDC="$RTL_CONSTRAINTS_DIR/clock_Aij.xdc"
export RTL_CLOCK_BI_XDC="$RTL_CONSTRAINTS_DIR/clock_bi.xdc"

# ==============================================================================
# Python paths
# ==============================================================================
export PY_ROOT="$FPGA_MRI_ROOT/py"
export PY_RUNNER="$PY_ROOT/runner"
export PY_FXP_MODEL_ROOT="$PY_ROOT/fxp_model"
export PY_NPY_DATA_ROOT="$PY_ROOT/npy_data"
export PY_FFT2D_ROOT="$PY_ROOT/fft2d"
export PY_SENSE_ROOT="$PY_ROOT/sense"
export PY_GEN_ROOT="$PY_ROOT/gen"
export PY_QUANTIZER_ROOT="$PY_ROOT/quantizer"

export PY_SENSE_FP_DIR="$PY_SENSE_ROOT/fp"
export PY_SENSE_FXP_DIR="$PY_SENSE_ROOT/fxp"
export PY_SENSE_REPORTER_DIR="$PY_SENSE_ROOT/reporter"

export PY_FFT2D_FXP_DIR="$PY_FFT2D_ROOT/fxp"
export PY_FXP_MODEL_TEST_DIR="$PY_FXP_MODEL_ROOT/test"

# Python model files
export PY_FXP_MODEL_FXP="$PY_FXP_MODEL_ROOT/fxp.py"
export PY_FXP_MODEL_CFXP="$PY_FXP_MODEL_ROOT/cfxp.py"
export PY_FXP_MODEL_CFXPTENSOR="$PY_FXP_MODEL_ROOT/cfxptensor.py"

# Generation scripts
export PY_GEN_PHANTOM="$PY_GEN_ROOT/gen_phantom.py"
export PY_GEN_SMAPS="$PY_GEN_ROOT/gen_smaps.py"
export PY_GEN_COILS="$PY_GEN_ROOT/gen_coils.py"
export PY_GEN_KSPACE="$PY_GEN_ROOT/gen_kspace.py"
export PY_GEN_ALIASED_KSPACE="$PY_GEN_ROOT/gen_aliased_kspace.py"
export PY_GEN_COIL_ALIASED="$PY_GEN_ROOT/gen_coil_aliased.py"

# Quantizer scripts
export PY_QUANTIZER_MAIN="$PY_QUANTIZER_ROOT/quantizer.py"
export PY_QUANTIZER_COMPLEX_3D="$PY_QUANTIZER_ROOT/quantize_complex_tensor_3d.py"
export PY_QUANTIZER_HELPERS="$PY_QUANTIZER_ROOT/helpers.py"
export PY_QUANTIZER_DAT_SAVER="$PY_QUANTIZER_ROOT/fxp_dat_saver.py"

# FFT2D FXP scripts
export PY_IFFT2D_FXP_RUNNER="$PY_FFT2D_FXP_DIR/fxp_ifft2d_runner.py"
export PY_FFT1D_FXP="$PY_FFT2D_FXP_DIR/fft1d.py"
export PY_FFT2D_FXP="$PY_FFT2D_FXP_DIR/fft2d.py"
export PY_FFT2D_FXP_DAT_SAVER="$PY_FFT2D_FXP_DIR/fxp_dat_saver.py"

# SENSE FP scripts
export PY_SENSE_FP_RUNNER="$PY_SENSE_FP_DIR/fp_sense_runner.py"
export PY_SENSE_FP_COMPUTE_A="$PY_SENSE_FP_DIR/fp_compute_A.py"
export PY_SENSE_FP_COMPUTE_B="$PY_SENSE_FP_DIR/fp_compute_b.py"
export PY_SENSE_FP_COMPUTE_I="$PY_SENSE_FP_DIR/fp_compute_I.py"
export PY_SENSE_FP_COMPUTE_M_HAT="$PY_SENSE_FP_DIR/fp_compute_m_hat_ldlh.py"

# SENSE FXP scripts
export PY_SENSE_FXP_RUNNER="$PY_SENSE_FXP_DIR/fxp_sense_runner.py"

export PY_SENSE_FXP_COMPUTE_A="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_A.py"
export PY_SENSE_FXP_COMPUTE_B="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_b.py"
export PY_SENSE_FXP_COMPUTE_D="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_D.py"
export PY_SENSE_FXP_COMPUTE_I="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_I.py"
export PY_SENSE_FXP_COMPUTE_L="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_L.py"
export PY_SENSE_FXP_COMPUTE_X="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_x.py"
export PY_SENSE_FXP_COMPUTE_Z="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_z.py"
export PY_SENSE_FXP_COMPUTE_M_HAT="$PY_SENSE_FXP_DIR/singleprocess/fxp_compute_m_hat.py"

export PY_SENSE_FXP_MP_COMPUTE_A="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_A.py"
export PY_SENSE_FXP_MP_COMPUTE_B="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_b.py"
export PY_SENSE_FXP_MP_COMPUTE_D="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_D.py"
export PY_SENSE_FXP_MP_COMPUTE_L="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_L.py"
export PY_SENSE_FXP_MP_COMPUTE_X="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_x.py"
export PY_SENSE_FXP_MP_COMPUTE_Z="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_z.py"
export PY_SENSE_FXP_MP_COMPUTE_M_HAT="$PY_SENSE_FXP_DIR/multiprocess/fxp_multiprocessing_compute_m_hat.py"

export PY_SENSE_FXP_DAT_SAVER="$PY_SENSE_FXP_DIR/helpers/fxp_dat_saver.py"
export PY_SENSE_FXP_RPT_WRITER="$PY_SENSE_FXP_DIR/helpers/fxp_rpt_writer.py"
export PY_SENSE_FXP_STATS="$PY_SENSE_FXP_DIR/helpers/fxp_stats.py"
export PY_SENSE_FXP_SAVE_TENSOR_PNG="$PY_SENSE_FXP_DIR/helpers/fxp_save_tensor_png.py"

# Reporter scripts
export PY_SENSE_REPORTER_RUNNER="$PY_SENSE_REPORTER_DIR/sense_reporter_runner.py"
export PY_SENSE_CSV_SNR="$PY_SENSE_ROOT/csv_snr.py"
export PY_SENSE_PLOT_SNR="$PY_SENSE_ROOT/plot_snr_vs_lsb.py"

# ==============================================================================
# Shell/Tcl scripts
# ==============================================================================
export SCRIPTS_ROOT="$FPGA_MRI_ROOT/scripts"

export PY_SCRIPTS_PATH="$SCRIPTS_ROOT/py"
export PY_GEN_SCRIPT="$PY_SCRIPTS_PATH/run_gen.sh"
export PY_QUANTIZER_SCRIPT="$PY_SCRIPTS_PATH/run_quantizer.sh"
export PY_SENSE_FXP_SCRIPT="$PY_SCRIPTS_PATH/run_sense_fxp.sh"
export PY_SENSE_FP_SCRIPT="$PY_SCRIPTS_PATH/run_sense_fp.sh"
export PY_SENSE_REPORTER_SCRIPT="$PY_SCRIPTS_PATH/run_sense_reporter.sh"
export PY_IFFT2D_FXP_SCRIPT="$PY_SCRIPTS_PATH/run_ifft2d_fxp.sh"
export PY_RUNNER_SCRIPT="$PY_SCRIPTS_PATH/run.sh"

export RTL_SCRIPTS_PATH="$SCRIPTS_ROOT/rtl"
export RTL_SCRIPTS_SYNTH_PATH="$RTL_SCRIPTS_PATH/synth"
export RTL_SCRIPTS_TB_PATH="$RTL_SCRIPTS_PATH/tb"
export RTL_SCRIPTS_VM_PATH="$RTL_SCRIPTS_PATH/vm"

export RTL_SYNTH_SCRIPT="$RTL_SCRIPTS_SYNTH_PATH/run_synth.sh"
export RTL_SYNTH_TCL="$RTL_SCRIPTS_SYNTH_PATH/synth_case.tcl"
export RTL_XSIM_SCRIPT="$RTL_SCRIPTS_TB_PATH/run_xsim.sh"
export RTL_VM_SCRIPT="$RTL_SCRIPTS_VM_PATH/run_vm.sh"
export RTL_VM_PY="$RTL_SCRIPTS_VM_PATH/run_vm.py"
export RTL_CREATE_RELEASE_SCRIPT="$RTL_SCRIPTS_PATH/create_release.sh"
export CREATE_RELEASE_COMMON_HELPER_SH="$RTL_SCRIPTS_PATH/helpers/create_release_common.sh"
export TRACK_RELEASE_HELPER_SH="$RTL_SCRIPTS_PATH/helpers/track_release_helper.sh"
export TRACK_MANIFEST_HELPER_SH="$RTL_SCRIPTS_PATH/helpers/track_manifest_helper.sh"

# ==============================================================================
# Tool paths
# ==============================================================================
export PY_VENV_ACTIVATE="$PY_ROOT/.venv/bin/activate"
export VIVADO_SETTINGS="${VIVADO_SETTINGS:-/tools/Xilinx/Vivado/2024.2/settings64.sh}"

# ==============================================================================
# Check helpers
# ==============================================================================
_PROJECT_PATHS_ERRORS=0

_check_dir() {
  local path="$1"

  if [[ ! -d "$path" ]]; then
    printf "[project_paths.sh] ${RED}MISSING DIR :${NC} %s\n" "$path"
    _PROJECT_PATHS_ERRORS=$((_PROJECT_PATHS_ERRORS + 1))
  else
    printf "[project_paths.sh] OK dir      : %s\n" "$path"
  fi
}

_check_file() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    printf "[project_paths.sh] ${RED}MISSING FILE:${NC} %s\n" "$path"
    _PROJECT_PATHS_ERRORS=$((_PROJECT_PATHS_ERRORS + 1))
  else
    printf "[project_paths.sh] OK file     : %s\n" "$path"
  fi
}

# ==============================================================================
# Required directories
# ==============================================================================
PROJECT_REQUIRED_DIRS=(
  "$FPGA_MRI_ROOT"
  "$TRACK_ROOT"

  "$RTL_ROOT"
  "$RTL_CONSTRAINTS_DIR"
  "$RTL_SRC_DIR"
  "$RTL_TESTBENCH_DIR"

  "$PY_ROOT"
  "$PY_RUNNER"
  "$PY_FXP_MODEL_ROOT"
  "$PY_NPY_DATA_ROOT"
  "$PY_FFT2D_ROOT"
  "$PY_FFT2D_FXP_DIR"
  "$PY_SENSE_ROOT"
  "$PY_SENSE_FP_DIR"
  "$PY_SENSE_FXP_DIR"
  "$PY_SENSE_REPORTER_DIR"
  "$PY_GEN_ROOT"
  "$PY_QUANTIZER_ROOT"
  "$PY_FXP_MODEL_TEST_DIR"

  "$SCRIPTS_ROOT"
  "$PY_SCRIPTS_PATH"
  "$RTL_SCRIPTS_PATH"
  "$RTL_SCRIPTS_SYNTH_PATH"
  "$RTL_SCRIPTS_TB_PATH"
  "$RTL_SCRIPTS_VM_PATH"
)

# ==============================================================================
# Required files
# ==============================================================================
PROJECT_REQUIRED_FILES=(
  # Config
  "$GLOBAL_CONF_PATH"
  "$TRACK_CONF"

  # Environment/tools
  "$PY_VENV_ACTIVATE"
  "$VIVADO_SETTINGS"

  # Python runner scripts
  "$PY_RUNNER_SCRIPT"
  "$PY_GEN_SCRIPT"
  "$PY_QUANTIZER_SCRIPT"
  "$PY_IFFT2D_FXP_SCRIPT"
  "$PY_SENSE_FP_SCRIPT"
  "$PY_SENSE_FXP_SCRIPT"
  "$PY_SENSE_REPORTER_SCRIPT"

  "$CREATE_RELEASE_COMMON_HELPER_SH"
  "$TRACK_RELEASE_HELPER_SH"
  "$TRACK_MANIFEST_HELPER_SH"

  # RTL runner scripts
  "$RTL_CREATE_RELEASE_SCRIPT"
  "$RTL_XSIM_SCRIPT"
  "$RTL_VM_SCRIPT"
  "$RTL_VM_PY"
  "$RTL_SYNTH_SCRIPT"
  "$RTL_SYNTH_TCL"

  # Python model
  "$PY_FXP_MODEL_FXP"
  "$PY_FXP_MODEL_CFXP"
  "$PY_FXP_MODEL_CFXPTENSOR"

  # Generation
  "$PY_GEN_PHANTOM"
  "$PY_GEN_SMAPS"
  "$PY_GEN_COILS"
  "$PY_GEN_KSPACE"
  "$PY_GEN_ALIASED_KSPACE"
  "$PY_GEN_COIL_ALIASED"

  # Quantizer
  "$PY_QUANTIZER_MAIN"
  "$PY_QUANTIZER_COMPLEX_3D"
  "$PY_QUANTIZER_HELPERS"
  "$PY_QUANTIZER_DAT_SAVER"

  # IFFT2D FXP
  "$PY_IFFT2D_FXP_RUNNER"
  "$PY_FFT1D_FXP"
  "$PY_FFT2D_FXP"
  "$PY_FFT2D_FXP_DAT_SAVER"

  # SENSE FP
  "$PY_SENSE_FP_RUNNER"
  "$PY_SENSE_FP_COMPUTE_A"
  "$PY_SENSE_FP_COMPUTE_B"
  "$PY_SENSE_FP_COMPUTE_I"
  "$PY_SENSE_FP_COMPUTE_M_HAT"

  # SENSE FXP
  "$PY_SENSE_FXP_RUNNER"
  "$PY_SENSE_FXP_COMPUTE_A"
  "$PY_SENSE_FXP_COMPUTE_B"
  "$PY_SENSE_FXP_COMPUTE_D"
  "$PY_SENSE_FXP_COMPUTE_I"
  "$PY_SENSE_FXP_COMPUTE_L"
  "$PY_SENSE_FXP_COMPUTE_X"
  "$PY_SENSE_FXP_COMPUTE_Z"
  "$PY_SENSE_FXP_COMPUTE_M_HAT"
  "$PY_SENSE_FXP_MP_COMPUTE_A"
  "$PY_SENSE_FXP_MP_COMPUTE_B"
  "$PY_SENSE_FXP_MP_COMPUTE_D"
  "$PY_SENSE_FXP_MP_COMPUTE_L"
  "$PY_SENSE_FXP_MP_COMPUTE_X"
  "$PY_SENSE_FXP_MP_COMPUTE_Z"
  "$PY_SENSE_FXP_MP_COMPUTE_M_HAT"
  "$PY_SENSE_FXP_DAT_SAVER"
  "$PY_SENSE_FXP_RPT_WRITER"
  "$PY_SENSE_FXP_STATS"
  "$PY_SENSE_FXP_SAVE_TENSOR_PNG"

  # Reporter
  "$PY_SENSE_REPORTER_RUNNER"
  "$PY_SENSE_CSV_SNR"
  "$PY_SENSE_PLOT_SNR"

  # RTL source
  "$RTL_CAST_SV"
  "$RTL_CMUL_SV"
  "$RTL_CSUM_SV"
  "$RTL_CSUB_SV"
  "$RTL_DIV_RESTORING_SV"
  "$RTL_COMPUTE_AIJ_SV"
  "$RTL_COMPUTE_BI_SV"

  # RTL testbenches
  "$RTL_TB_COMPUTE_AIJ_SV"
  "$RTL_TB_COMPUTE_BI_SV"
  "$RTL_TB_DIV_RESTORING_SV"

  # RTL constraints
  "$RTL_CLOCK_AIJ_XDC"
  "$RTL_CLOCK_BI_XDC"

  "$TRACK_MANIFEST_HELPER_SH"
)

# ==============================================================================
# Run checks
# ==============================================================================
printf "\n"
printf "[project_paths.sh] ${CYAN}Checking required directories...${NC}\n"
for d in "${PROJECT_REQUIRED_DIRS[@]}"; do
  _check_dir "$d"
done

printf "\n"
printf "[project_paths.sh] ${CYAN}Checking required files...${NC}\n"
for f in "${PROJECT_REQUIRED_FILES[@]}"; do
  _check_file "$f"
done

printf "\n"

if [[ "$_PROJECT_PATHS_ERRORS" -ne 0 ]]; then
  printf "[project_paths.sh] ${RED}ERROR:${NC} %d path check(s) failed.\n" "$_PROJECT_PATHS_ERRORS"
  return 1
fi

printf "[project_paths.sh] ${GREEN}All project paths loaded and verified successfully.${NC}\n"
printf "\n"
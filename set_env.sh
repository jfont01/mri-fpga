# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
printf "[set_env.sh]    ${GREEN}Loading enviorment variables...${NC}\n"
echo ""

: "${FPGA_MRI_ROOT:?Enviroment variable FPGA_MRI_ROOT must be defined}"

###########################################################################
# Funciones auxiliares
###########################################################################
check_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf "[set_env.sh]    ${RED}ERROR:${NC} directory not found: $path\n"
    return 1
  fi
  echo "[set_env.sh]    OK dir : $path"
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf "[set_env.sh]    ${RED}ERROR:${NC} file not found: $path\n"
    return 1
  fi
  echo "[set_env.sh]    OK file: $path"
}

check_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    printf "[set_env.sh]    ${RED}ERROR:${NC} variable '$name' is empty or undefined\n"
    return 1
  fi
  echo "[set_env.sh]    OK var : $name=$value"
}




###################################### Global Config Path ######################################
export GLOBAL_CONF_PATH="$FPGA_MRI_ROOT/global_config.conf"             #global_config.conf

###################################### Tracks Paths ######################################
export TRACK_ROOT="$FPGA_MRI_ROOT/tracks"
export TRACK_CONF="$TRACK_ROOT/track.conf"

######################################### Vivado Paths #########################################
export VIVADO_ROOT="$FPGA_MRI_ROOT/amd/vivado_sense"
export VIVADO_SIM_DIR="$VIVADO_ROOT/vivado_sense.sim/sim_1/behav/xsim"

######################################### RTL Paths #########################################
# Root
export RTL_ROOT="$FPGA_MRI_ROOT/rtl"                                    #rtl
# Dirs
export RTL_CONSTRAINTS_DIR="$RTL_ROOT/constraints"                      #rtl/constraints
export RTL_SCRIPTS_DIR="$RTL_ROOT/scripts"                              #rtl/scripts
export RTL_SRC_DIR="$RTL_ROOT/src"                                      #rtl/src
export RTL_TESTBENCH_DIR="$RTL_ROOT/testbench"                          #rtl/testbench
export RTL_SCRIPTS_SYNTH_DIR="$RTL_SCRIPTS_DIR/synth"                   #rtl/scripts/synth
export RTL_SCRIPTS_TB_DIR="$RTL_SCRIPTS_DIR/synth"                      #rtl/scripts/tb

# Scripts
# export RTL_SCRIPTS_SYNTH_RUNNER="$RTL_SCRIPTS_DIR/run_synth.sh"         #rtl/scripts/run_synth.sh
# export RTL_SCRIPTS_VM_RUNNER="$RTL_SCRIPTS_DIR/run_vm.py"               #rtl/scripts/run_vm.py
# export RTL_SCRIPTS_XSIM_RUNNER="$RTL_SCRIPTS_DIR/run_xsim.sh"           #rtl/scripts/run_xsim.sh

######################################### Python Paths #########################################
# Roots
export PY_ROOT="$FPGA_MRI_ROOT/py"                                      #py/
export PY_RUNNER="$PY_ROOT/runner"                                      #py/runner
export PY_FXP_MODEL_ROOT="$PY_ROOT/fxp_model"                           #py/fxp_model
export PY_NPY_DATA_ROOT="$PY_ROOT/npy_data"                             #py/npy_data
export PY_FFT2D_ROOT="$PY_ROOT/fft2d"                                   #py/fft2d
export PY_SENSE_ROOT="$PY_ROOT/sense"                                   #py/sense
export PY_GEN_ROOT="$PY_ROOT/gen"                                       #py/gen
export PY_QUANTIZER_ROOT="$PY_ROOT/quantizer"                           #py/quantizer

# Sense Directories
export PY_SENSE_FP_DIR="$PY_SENSE_ROOT/fp"                              #py/sense/fp
export PY_SENSE_FXP_DIR="$PY_SENSE_ROOT/fxp"                            #py/sense/fxp
export PY_SENSE_REPORTER_DIR="$PY_SENSE_ROOT/reporter"                  #py/sense/reporter

# FFT2D Directories
# export PY_FFT2D_FP_DIR="$PY_FFT2D_ROOT/fp"                              #py/fft2d/fp
export PY_FFT2D_FXP_DIR="$PY_FFT2D_ROOT/fxp"                            #py/fft2d/fxp
# export PY_FFT2D_REPORTER_DIR="$PY_FFT2D_ROOT/reporter"                  #py/fft2d/reporter


export PY_FXP_MODEL_TEST_DIR="$PY_FXP_MODEL_ROOT/test"


######################################### Scripts paths #########################################
export SCRIPTS_ROOT="$FPGA_MRI_ROOT/scripts"

export RTL_SCRIPTS_PATH="$SCRIPTS_ROOT/rtl"

export PY_SCRIPTS_PATH="$SCRIPTS_ROOT/py"
export PY_GEN_SCRIPT="$PY_SCRIPTS_PATH/run_gen.sh"
export PY_QUANTIZER_SCRIPT="$PY_SCRIPTS_PATH/run_quantizer.sh"
export PY_SENSE_FXP_SCRIPT="$PY_SCRIPTS_PATH/run_sense_fxp.sh"
export PY_SENSE_FP_SCRIPT="$PY_SCRIPTS_PATH/run_sense_fp.sh"
export PY_SENSE_REPORTER_SCRIPT="$PY_SCRIPTS_PATH/run_sense_reporter.sh"
export PY_IFFT2D_FXP_SCRIPT="$PY_SCRIPTS_PATH/run_ifft2d_fxp.sh"
export PY_RUNNER_SCRIPT="$PY_SCRIPTS_PATH/run.sh"

###########################################################################
# Verificación de variables
###########################################################################
check_var GLOBAL_CONF_PATH

check_var TRACK_ROOT
check_var TRACK_CONF

check_var FPGA_MRI_ROOT

# check_var VIVADO_ROOT
# check_var VIVADO_SIM_DIR

check_var RTL_ROOT
check_var RTL_CONSTRAINTS_DIR
check_var RTL_SCRIPTS_DIR
check_var RTL_SRC_DIR
check_var RTL_TESTBENCH_DIR
check_var RTL_SCRIPTS_SYNTH_DIR
check_var RTL_SCRIPTS_TB_DIR

check_var PY_ROOT
check_var PY_FXP_MODEL_ROOT
check_var PY_NPY_DATA_ROOT
check_var PY_FFT2D_ROOT
check_var PY_SENSE_ROOT
check_var PY_GEN_ROOT
check_var PY_QUANTIZER_ROOT

check_var PY_SENSE_FP_DIR
check_var PY_SENSE_FXP_DIR
check_var PY_SENSE_REPORTER_DIR

# check_var PY_FFT2D_FP_DIR
check_var PY_FFT2D_FXP_DIR
# check_var PY_FFT2D_REPORTER_DIR


check_var PY_FXP_MODEL_TEST_DIR
echo ""
###########################################################################
# Verificación de directorios
###########################################################################
check_dir "$FPGA_MRI_ROOT"
check_dir "$TRACK_ROOT"

check_dir "$PY_ROOT"

check_dir "$RTL_ROOT"
check_dir "$RTL_CONSTRAINTS_DIR"
check_dir "$RTL_SCRIPTS_DIR"
check_dir "$RTL_SRC_DIR"
check_dir "$RTL_TESTBENCH_DIR"
check_dir "$RTL_SCRIPTS_SYNTH_DIR"
check_dir "$RTL_SCRIPTS_TB_DIR"

# check_dir "$VIVADO_ROOT"
# check_dir "$VIVADO_SIM_DIR"
check_dir "$PY_RUNNER"
check_dir "$PY_FXP_MODEL_ROOT"
check_dir "$PY_NPY_DATA_ROOT"
check_dir "$PY_FFT2D_ROOT"
check_dir "$PY_SENSE_ROOT"
check_dir "$PY_GEN_ROOT"
check_dir "$PY_QUANTIZER_ROOT"
check_dir "$PY_SENSE_FP_DIR"
check_dir "$PY_SENSE_FXP_DIR"
check_dir "$PY_SENSE_REPORTER_DIR"
# check_dir "$PY_FFT2D_FP_DIR"
check_dir "$PY_FFT2D_FXP_DIR"
# check_dir "$PY_FFT2D_REPORTER_DIR"
check_dir "$PY_FXP_MODEL_TEST_DIR"
echo ""
###########################################################################
# Verificación de archivos
###########################################################################
check_file "$GLOBAL_CONF_PATH"
check_file "$TRACK_CONF"
check_file "$PY_GEN_SCRIPT"
check_file "$PY_QUANTIZER_SCRIPT"
check_file "$PY_SENSE_FXP_SCRIPT"
check_file "$PY_SENSE_FP_SCRIPT"
check_file "$PY_SENSE_REPORTER_SCRIPT"
check_file "$PY_IFFT2D_FXP_SCRIPT"
check_file "$PY_RUNNER_SCRIPT"




# check_file "$RTL_SCRIPTS_SYNTH_RUNNER"
# check_file "$RTL_SCRIPTS_VM_RUNNER"
# check_file "$RTL_SCRIPTS_XSIM_RUNNER"
echo ""



printf "[set_env.sh]    Sourcing .venv/bin/activate ...\n"
source py/.venv/bin/activate

echo ""
printf "[set_env.sh]    ${GREEN}Environment loaded successfully.${NC}\n"


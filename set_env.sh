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
# Roots
export VM_ROOT="$FPGA_MRI_ROOT/vm"

######################################### Vivado Paths #########################################
export VIVADO_ROOT="$FPGA_MRI_ROOT/amd/vivado_sense"
export VIVADO_SIM_DIR="$VIVADO_ROOT/vivado_sense.sim/sim_1/behav/xsim"
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

# Quantizer Runner
export PY_QUANTIZER_RUN="$PY_QUANTIZER_ROOT/run_quantizer.sh"           #py/quantizer/run_quantizer.sh

# Gen Runner
export PY_GEN_RUN="$PY_GEN_ROOT/run_gen.sh"                             #py/gen/run_gen.sh

# Sense Directories
export PY_SENSE_FP_DIR="$PY_SENSE_ROOT/fp"                              #py/sense/fp
export PY_SENSE_FXP_DIR="$PY_SENSE_ROOT/fxp"                            #py/sense/fxp
export PY_SENSE_REPORTER_DIR="$PY_SENSE_ROOT/reporter"                  #py/sense/reporter

# FFT2D Directories
export PY_FFT2D_FP_DIR="$PY_FFT2D_ROOT/fp"                              #py/fft2d/fp
export PY_FFT2D_FXP_DIR="$PY_FFT2D_ROOT/fxp"                            #py/fft2d/fxp
export PY_FFT2D_REPORTER_DIR="$PY_FFT2D_ROOT/reporter"                  #py/fft2d/reporter

# Sense Runners
export PY_SENSE_REPORTER_RUN="$PY_SENSE_REPORTER_DIR/run_reporter.sh"   #py/sense/reporter/run_reporter.sh
export PY_SENSE_FP_RUN="$PY_SENSE_FP_DIR/run_sense_fp.sh"               #py/sense/fp/run_sense_fp.sh
export PY_SENSE_FXP_RUN="$PY_SENSE_FXP_DIR/run_sense_fxp.sh"            #py/sense/fxp/run_sense_fxp.sh

# FFT2D Runners
export PY_FFT2D_FXP_RUN="$PY_FFT2D_FXP_DIR/run_ifft2d_fxp.sh"           #py/fft2d/fxp/run_ifft2d_fxp.sh

# Global runner path
export PY_GLOBAL_RUN="$PY_RUNNER/run.sh"                                #py/run.sh

export PY_FXP_MODEL_TEST_DIR="$PY_FXP_MODEL_ROOT/test"

######################################### Cpp Paths #########################################


###########################################################################
# Verificación de variables
###########################################################################
check_var GLOBAL_CONF_PATH

check_var FPGA_MRI_ROOT

check_var VM_ROOT

check_var VIVADO_ROOT
check_var VIVADO_SIM_DIR

check_var PY_ROOT
check_var PY_RUNNER
check_var PY_FXP_MODEL_ROOT
check_var PY_NPY_DATA_ROOT
check_var PY_FFT2D_ROOT
check_var PY_SENSE_ROOT
check_var PY_GEN_ROOT
check_var PY_QUANTIZER_ROOT

check_var PY_QUANTIZER_RUN

check_var PY_GEN_RUN

check_var PY_SENSE_FP_DIR
check_var PY_SENSE_FXP_DIR
check_var PY_SENSE_REPORTER_DIR

check_var PY_FFT2D_FP_DIR
check_var PY_FFT2D_FXP_DIR
check_var PY_FFT2D_REPORTER_DIR

check_var PY_SENSE_REPORTER_RUN
check_var PY_SENSE_FP_RUN
check_var PY_SENSE_FXP_RUN

check_var PY_GLOBAL_RUN

check_var PY_FXP_MODEL_TEST_DIR
echo ""
###########################################################################
# Verificación de directorios
###########################################################################
check_dir "$FPGA_MRI_ROOT"
check_dir "$PY_ROOT"
check_dir "$VM_ROOT"
check_dir "$VIVADO_ROOT"
check_dir "$VIVADO_SIM_DIR"
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
check_dir "$PY_FFT2D_FP_DIR"
check_dir "$PY_FFT2D_FXP_DIR"
check_dir "$PY_FFT2D_REPORTER_DIR"
check_dir "$PY_FXP_MODEL_TEST_DIR"
echo ""
###########################################################################
# Verificación de archivos
###########################################################################
check_file "$GLOBAL_CONF_PATH"
check_file "$PY_QUANTIZER_RUN"
check_file "$PY_GEN_RUN"
check_file "$PY_SENSE_REPORTER_RUN"
check_file "$PY_SENSE_FP_RUN"
check_file "$PY_SENSE_FXP_RUN"
check_file "$PY_GLOBAL_RUN"
echo ""

printf "[set_env.sh]    Running dos2unix on scripts and config files...\n"

dos2unix $GLOBAL_CONF_PATH
dos2unix $PY_QUANTIZER_RUN
dos2unix $PY_GEN_RUN
dos2unix $PY_SENSE_REPORTER_RUN
dos2unix $PY_SENSE_FP_RUN
dos2unix $PY_SENSE_FXP_RUN
dos2unix $PY_GLOBAL_RUN
echo ""

printf "[set_env.sh]    Sourcing .venv/bin/activate ...\n"
source .venv/bin/activate

echo ""
printf "[set_env.sh]    ${GREEN}Environment loaded successfully.${NC}\n"


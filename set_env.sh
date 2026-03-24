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
    printf "[set_env.sh]    ${RED}ERROR:${NC} directory not found: $path"
    return 1
  fi
  echo "[set_env.sh]    OK dir : $path"
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf "[set_env.sh]    ${RED}ERROR:${NC} file not found: $path"
    return 1
  fi
  echo "[set_env.sh]    OK file: $path"
}

check_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    printf "[set_env.sh]    ${RED}ERROR:${NC} variable '$name' is empty or undefined"
    return 1
  fi
  echo "[set_env.sh]    OK var : $name=$value"
}

###########################################################################
# Variables base
###########################################################################
export PY_ROOT="$FPGA_MRI_ROOT/py"
export FXP_MODEL_ROOT="$PY_ROOT/fxp_model"
export NPY_DATA_ROOT="$PY_ROOT/npy_data"
export FFT2D_ROOT="$PY_ROOT/fft2d"
export SENSE_ROOT="$PY_ROOT/sense"
export GLOBAL_CONF_PATH="$PY_ROOT/global_config.conf"
export GLOBAL_RUN_PATH="$PY_ROOT/run.sh"
export SENSE_GEN_DIR="$SENSE_ROOT/gen"
export SENSE_GEN_RUN="$SENSE_GEN_DIR/run_gen.sh"
export SENSE_FP_DIR="$SENSE_ROOT/fp"
export SENSE_FP_RUN="$SENSE_FP_DIR/run_sense_fp.sh"
export SENSE_FXP_DIR="$SENSE_ROOT/fxp"
export SENSE_FXP_QUANTIZER_DIR="$SENSE_FXP_DIR/quantizer"
export SENSE_QUANTIZER_RUN="$SENSE_FXP_QUANTIZER_DIR/run_quantizer.sh"
export FXP_MODEL_TEST_DIR="$FXP_MODEL_ROOT/test"
export SENSE_FXP_RUN="$SENSE_FXP_DIR/run_sense_fxp.sh"

###########################################################################
# Verificación de variables
###########################################################################
check_var FPGA_MRI_ROOT
check_var PY_ROOT
check_var FXP_MODEL_ROOT
check_var NPY_DATA_ROOT
check_var FFT2D_ROOT
check_var SENSE_ROOT
check_var SENSE_GEN_DIR
check_var SENSE_FP_DIR
check_var FXP_MODEL_TEST_DIR
check_var SENSE_FP_RUN
check_var SENSE_GEN_RUN
check_var SENSE_FXP_DIR
check_var SENSE_FXP_QUANTIZER_DIR
check_var GLOBAL_CONF_PATH
check_var GLOBAL_RUN_PATH
check_var SENSE_QUANTIZER_RUN
check_var SENSE_FXP_RUN
echo ""
###########################################################################
# Verificación de directorios
###########################################################################
check_dir "$FPGA_MRI_ROOT"
check_dir "$PY_ROOT"
check_dir "$FXP_MODEL_ROOT"
check_dir "$NPY_DATA_ROOT"
check_dir "$FFT2D_ROOT"
check_dir "$SENSE_ROOT"
check_dir "$SENSE_GEN_DIR"
check_dir "$SENSE_FP_DIR"
check_dir "$FXP_MODEL_TEST_DIR"
check_dir "$SENSE_FXP_DIR"
check_dir "$SENSE_FXP_QUANTIZER_DIR"

echo ""
###########################################################################
# Verificación de archivos
###########################################################################
check_file "$SENSE_GEN_RUN"
check_file "$SENSE_FP_RUN"
check_file "$GLOBAL_CONF_PATH"
check_file "$GLOBAL_RUN_PATH"
check_file "$SENSE_QUANTIZER_RUN"
check_file "$SENSE_FXP_RUN"
echo ""

printf "[set_env.sh]    Running dos2unix on scripts and config files...\n"

dos2unix $SENSE_GEN_RUN
dos2unix $SENSE_FP_RUN
dos2unix $GLOBAL_CONF_PATH
dos2unix $GLOBAL_RUN_PATH
dos2unix $SENSE_QUANTIZER_RUN
dos2unix $SENSE_FXP_RUN
echo ""

printf "[set_env.sh]    Sourcing .venv/bin/activate ...\n"
source .venv/bin/activate

echo ""
printf "[set_env.sh]    ${GREEN}Environment loaded successfully.${NC}\n"


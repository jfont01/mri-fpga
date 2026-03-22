echo ""
echo "Loading enviorment variables..."
echo ""

: "${FPGA_MRI_ROOT:?Enviroment variable FPGA_MRI_ROOT must be defined}"

###########################################################################
# Funciones auxiliares
###########################################################################
check_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[set_env.sh] ERROR: directory not found: $path"
    return 1
  fi
  echo "[set_env.sh] OK dir : $path"
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[set_env.sh] ERROR: file not found: $path"
    return 1
  fi
  echo "[set_env.sh] OK file: $path"
}

check_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "[set_env.sh] ERROR: variable '$name' is empty or undefined"
    return 1
  fi
  echo "[set_env.sh] OK var : $name=$value"
}

###########################################################################
# Variables base
###########################################################################
export PY_ROOT="$FPGA_MRI_ROOT/py"
export FXP_MODEL_ROOT="$PY_ROOT/fxp_model"
export NPY_DATA_ROOT="$PY_ROOT/npy_data"
export FFT2D_ROOT="$PY_ROOT/fft2d"
export SENSE_ROOT="$PY_ROOT/sense"

export SENSE_GEN_DIR="$SENSE_ROOT/gen"
export SENSE_GEN_CONF="$SENSE_GEN_DIR/config.conf"
export SENSE_GEN_RUN="$SENSE_GEN_DIR/run_gen.sh"

export SENSE_FP_DIR="$SENSE_ROOT/fp"
export SENSE_FP_CONF="$SENSE_FP_DIR/config.conf"
export SENSE_FP_RUN="$SENSE_FP_DIR/run_recon.sh"

export SENSE_FXP_DIR="$SENSE_ROOT/fxp"
export SENSE_FXP_QUANTIZER_DIR="$SENSE_FXP_DIR/quantizer"
export SENSE_FXP_QUANTIZER_CONF="$SENSE_FXP_QUANTIZER_DIR/config.conf"

export FXP_MODEL_TEST_DIR="$FXP_MODEL_ROOT/test"

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
check_var SENSE_GEN_CONF
check_var FXP_MODEL_TEST_DIR
check_var SENSE_FP_CONF
check_var SENSE_FP_RUN
check_var SENSE_GEN_RUN
check_var SENSE_FXP_DIR
check_var SENSE_FXP_QUANTIZER_DIR
check_var SENSE_FXP_QUANTIZER_CONF
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
check_dir  "$SENSE_FXP_QUANTIZER_DIR"
echo ""
###########################################################################
# Verificación de archivos
###########################################################################
check_file "$SENSE_GEN_CONF"
check_file "$SENSE_FP_CONF"
check_file "$SENSE_GEN_RUN"
check_file "$SENSE_FP_RUN"
check_file "$SENSE_FXP_QUANTIZER_CONF"
echo ""

dos2unix $SENSE_GEN_CONF
dos2unix $SENSE_FP_CONF
dos2unix $SENSE_GEN_RUN
dos2unix $SENSE_FP_RUN
dos2unix $SENSE_FXP_QUANTIZER_CONF
echo ""

echo "[set_env.sh] Environment loaded successfully."
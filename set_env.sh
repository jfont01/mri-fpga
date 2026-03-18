#!/usr/bin/env bash
set -euo pipefail

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
PY_ROOT="$FPGA_MRI_ROOT/py"
FXP_MODEL_ROOT="$PY_ROOT/fxp_model"
NPY_DATA_ROOT="$PY_ROOT/npy_data"
FFT2D_ROOT="$PY_ROOT/fft2d"
SENSE_ROOT="$PY_ROOT/sense"

SENSE_GEN_DIR="$SENSE_ROOT/gen"
SENSE_GEN_CONF="$SENSE_GEN_DIR/config.conf"
SENSE_FP_DIR="$SENSE_ROOT/fp"
SENSE_FP_CONF="$SENSE_FP_DIR/config.conf"


FXP_MODEL_TEST_DIR="$FXP_MODEL_ROOT/test"

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
###########################################################################
# Verificación de archivos
###########################################################################
check_file "$SENSE_GEN_CONF"
check_file "$SENSE_FP_CONF"
echo "[set_env.sh] Environment loaded successfully."
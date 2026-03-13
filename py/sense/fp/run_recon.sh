#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_MRI_ROOT:? Enviroment variable FPGA_MRI_ROOT must be defined}"

echo "==================================================================="
echo " Iniciando reconstrucción SENSE (floating point)"
echo "FPGA_MRI_ROOT = ${FPGA_MRI_ROOT}"
echo "==================================================================="

###########################################################################
#  Paths de entrada
###########################################################################
GEN_DIR="${FPGA_MRI_ROOT%/}/py/sense/gen"
CONF_PATH="$GEN_DIR/config.conf"
CONF="$CONF_PATH"

echo "GEN_DIR      = $GEN_DIR"
echo "CONF_PATH    = $CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: config file not found: $CONF"
  exit 1
fi

echo "Loading config from $CONF"
# shellcheck disable=SC1090
source "$CONF"

###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo " Parámetros de configuración:"
echo "  N       = $N"
echo "  AF      = $AF"
echo "  L       = $L"
echo "  AXIS    = $AXIS"
echo "  PHANTOM = $PHANTOM"
echo ""

###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
OUTPUT_DIR="${FPGA_MRI_ROOT%/}/py/sense/fp/recon"
PHANTOM_DIR="$GEN_DIR/pipes/N${N}_Af${AF}_L${L}_axis${AXIS}_phantom${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap_N${N}.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased_Af${AF}_axis${AXIS}.npy"

echo "PHANTOM_DIR            = $PHANTOM_DIR"
echo "SENS_MAPS_NPY_PATH     = $SENS_MAPS_NPY_PATH"
echo "ALIASED_COILS_NPY_PATH = $ALIASED_COILS_NPY_PATH"
echo "OUTPUT_DIR             = $OUTPUT_DIR"
echo ""

# Checks básicos de existencia
if [[ ! -f "$SENS_MAPS_NPY_PATH" ]]; then
  echo "ERROR: No se encontró smaps .npy en:"
  echo "  $SENS_MAPS_NPY_PATH"
  exit 1
fi

if [[ ! -f "$ALIASED_COILS_NPY_PATH" ]]; then
  echo "ERROR: No se encontró coils aliasadas .npy en:"
  echo "  $ALIASED_COILS_NPY_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo " OUTPUT_DIR creado."
echo ""

###########################################################################
#  Llamada a compute_m_hat.py
###########################################################################
echo " Ejecutando compute_m_hat.py con:"
echo "  --smaps-npy-path        \"$SENS_MAPS_NPY_PATH\""
echo "  --aliased-coils-npy-path \"$ALIASED_COILS_NPY_PATH\""
echo "  --output-path           \"$OUTPUT_DIR\""
echo "-------------------------------------------------------------------"

python3 compute_m_hat.py \
  --smaps-npy-path="$SENS_MAPS_NPY_PATH" \
  --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH" \
  --output-path="$OUTPUT_DIR"

echo "-------------------------------------------------------------------"
echo "Reconstrucción SENSE (floating point) finalizada."
echo "Resultados en: $OUTPUT_DIR"
echo "==================================================================="
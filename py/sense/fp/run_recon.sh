#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_MRI_ROOT:? Enviroment variable FPGA_MRI_ROOT must be defined}"


echo ""
echo "==================================================================="
echo " Running fp SENSE reconstruction"
echo "==================================================================="
echo ""
###########################################################################
#  Paths de entrada
###########################################################################
PY_ROOT="$FPGA_MRI_ROOT/py"
SENSE_DIR="$PY_ROOT/sense"
GEN_DIR="$SENSE_DIR/gen"
FP_DIR="$SENSE_DIR/fp"
CONF_PATH="$FP_DIR/config.conf"
CONF="$CONF_PATH"

echo "Paths:"
echo "FPGA_MRI_ROOT   = $FPGA_MRI_ROOT"
echo "PY_ROOT         = $PY_ROOT"
echo "SENSE_DIR       = $SENSE_DIR"
echo "GEN_DIR         = $GEN_DIR"
echo "CONF_PATH       = $CONF_PATH"
echo ""

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: config file not found: $CONF"
  exit 1
fi

source "$CONF"

###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo "Parámetros a reconstruir:"
echo "  N             = $N"
echo "  AF            = $AF"
echo "  L             = $L"
echo "  AXIS          = $AXIS"
echo "  PHANTOM       = $PHANTOM"
echo "  COMPUTE_TYPE  = $COMPUTE_TYPE"
echo ""

###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
OUTPUT_DIR="$FP_DIR/recon/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
PHANTOM_DIR="$GEN_DIR/pipes/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap_N${N}.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased_Af${AF}_axis${AXIS}.npy"

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

echo "Estímulos a reconstruir:"
echo "SENS_MAPS_NPY_PATH     = $SENS_MAPS_NPY_PATH"
echo "ALIASED_COILS_NPY_PATH = $ALIASED_COILS_NPY_PATH"
echo ""





###########################################################################
#  Llamada a compute_m_hat.py
###########################################################################
echo " Ejecutando sense.py"
echo ""
python3 sense.py                                        \
  --smaps-npy-path="$SENS_MAPS_NPY_PATH"                \
  --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH"    \
  --output-path="$OUTPUT_DIR"                           \
  --compute-type=$COMPUTE_TYPE
echo ""
echo "Done"
echo "Output path:"
echo "OUTPUT_DIR             = $OUTPUT_DIR"
echo ""
echo "==================================================================="

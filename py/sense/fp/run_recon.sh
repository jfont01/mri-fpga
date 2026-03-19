echo ""
echo "==================================================================="
echo " Running fp SENSE reconstruction"
echo "==================================================================="
echo ""

###########################################################################
#  Requiere entorno cargado por set_env.sh
###########################################################################
: "${FPGA_MRI_ROOT:?Env var FPGA_MRI_ROOT must be defined}"
: "${PY_ROOT:?Env var PY_ROOT must be defined (source set_env.sh)}"
: "${SENSE_ROOT:?Env var SENSE_ROOT must be defined (source set_env.sh)}"
: "${SENSE_GEN_DIR:?Env var SENSE_GEN_DIR must be defined (source set_env.sh)}"
: "${SENSE_FP_DIR:?Env var SENSE_FP_DIR must be defined (source set_env.sh)}"
: "${SENSE_FP_CONF:?Env var SENSE_FP_CONF must be defined (source set_env.sh)}"

: "${SENSE_FP_RUN:?Env var SENSE_FP_RUN must be defined (source set_env.sh)}"

CONF_PATH="$SENSE_FP_CONF"
CONF="$CONF_PATH"

echo "Paths (from environment):"
echo "FPGA_MRI_ROOT   = $FPGA_MRI_ROOT"
echo "PY_ROOT         = $PY_ROOT"
echo "SENSE_ROOT      = $SENSE_ROOT"
echo "SENSE_GEN_DIR   = $SENSE_GEN_DIR"
echo "SENSE_FP_DIR    = $SENSE_FP_DIR"
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
OUTPUT_DIR="$SENSE_FP_DIR/recon/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
PHANTOM_DIR="$SENSE_GEN_DIR/pipes/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap_N${N}.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased_Af${AF}_axis${AXIS}.npy"


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
echo "OUTPUT_DIR creado:"
echo "  $OUTPUT_DIR"
echo ""

echo "Estímulos a reconstruir:"
echo "SENS_MAPS_NPY_PATH     = $SENS_MAPS_NPY_PATH"
echo "ALIASED_COILS_NPY_PATH = $ALIASED_COILS_NPY_PATH"
echo ""

###########################################################################
#  Llamada a sense.py
###########################################################################
echo "Ejecutando sense.py"
echo ""

python3 "$SENSE_FP_DIR/sense.py" \
  --smaps-npy-path="$SENS_MAPS_NPY_PATH" \
  --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH" \
  --output-path="$OUTPUT_DIR" \
  --compute-type="$COMPUTE_TYPE"

echo ""
echo "Done"
echo "Output path:"
echo "  OUTPUT_DIR = $OUTPUT_DIR"
echo ""
echo "==================================================================="
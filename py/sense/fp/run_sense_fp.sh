# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_sense_fp.sh]${GREEN}    Running fp SENSE reconstruction${NC}\n"
printf "\n"

CONF="$GLOBAL_CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: config file not found: $CONF"
  exit 1
fi

source "$CONF"

###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo "[run_sense_fp.sh]    Parámetros a reconstruir:"
echo "  N             = $N"
echo "  AF            = $AF"
echo "  L             = $L"
echo "  AXIS          = $AXIS"
echo "  PHANTOM       = $PHANTOM"
echo ""

###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
OUTPUT_DIR="$SENSE_FP_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
PHANTOM_DIR="$SENSE_GEN_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased.npy"


if [[ ! -f "$SENS_MAPS_NPY_PATH" ]]; then
  printf "[run_sense_fp.sh] ${RED}ERROR: smaps .npy not found in: $SENS_MAPS_NPY_PATH${NC}\n"
  exit 1
fi

if [[ ! -f "$ALIASED_COILS_NPY_PATH" ]]; then
  printf "[run_sense_fp.sh]   ${RED}ERROR: aliased coils .npy not found in: $ALIASED_COILS_NPY_PATH${NC}\n"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[run_sense_fp.sh]   Created output dir: $OUTPUT_DIR"
echo ""

echo "[run_sense_fp.sh]   Reconstruction stimulus .npy:"
echo "SENS_MAPS_NPY_PATH     = $SENS_MAPS_NPY_PATH"
echo "ALIASED_COILS_NPY_PATH = $ALIASED_COILS_NPY_PATH"
echo ""

echo "[run_sense_fp.sh]   Running fp_sense.py"
echo ""

python3 "$SENSE_FP_DIR/fp_sense.py"                       \
  --smaps-npy-path="$SENS_MAPS_NPY_PATH"                  \
  --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH"      \
  --output-path="$OUTPUT_DIR"                             

echo ""
printf "[run_sense_fp.sh]   ${GREEN}Reconstruction finished successfully${NC}\n"
printf "\n"
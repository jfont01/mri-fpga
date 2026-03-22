# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_cuantizer.sh] ${CYAN}Running run_cuantizer.sh${NC}\n"
printf "\n"

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

CONF_PATH="$SENSE_FXP_QUANTIZER_CONF"
CONF="$CONF_PATH"


if [[ ! -f "$CONF" ]]; then
  printf "[run_cuantizer.sh]    ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"


###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
PHANTOM_DIR="$SENSE_GEN_DIR/pipes/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap_N${N}.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased_Af${AF}_axis${AXIS}.npy"

OUTPUT_DIR="$SENSE_FXP_QUANTIZER_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB${NB}_NBF${NBF}"

mkdir -p $OUTPUT_DIR

OUTPUT_NAME_S="S_q_NB${NB}_NBF${NBF}"
OUTPUT_NAME_y="y_q_NB${NB}_NBF${NBF}"

OUTPUT_SMAPS_PATH="$OUTPUT_DIR/$OUTPUT_NAME_S"
OUTPUT_Y_PATH="$OUTPUT_DIR/$OUTPUT_NAME_y"

if [[ ! -f "$SENS_MAPS_NPY_PATH" ]]; then
  printf "[run_cuantizer.sh] ${RED}ERROR: smaps .npy not found: $SENS_MAPS_NPY_PATH${NC}\n"
  exit 1
fi

if [[ ! -f "$ALIASED_COILS_NPY_PATH" ]]; then
  printf "[run_cuantizer.sh] ${RED}ERROR: aliased coils not found: $ALIASED_COILS_NPY_PATH${NC}\n"
  exit 1
fi

###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo "[run_cuantizer.sh]  Files to quantize:"
echo "[run_cuantizer.sh]      Sensitivity Maps  : $SENS_MAPS_NPY_PATH"
echo "[run_cuantizer.sh]      Aliased coils     : $ALIASED_COILS_NPY_PATH"
echo ""

echo "[run_cuantizer.sh]  Conf file readed      : $CONF"
echo ""

echo "[run_cuantizer.sh]  Parameters for quantization:"
echo "[run_cuantizer.sh]      NB             = $NB"
echo "[run_cuantizer.sh]      NBF            = $NBF"
echo "[run_cuantizer.sh]      signed         = $signed"
echo "[run_cuantizer.sh]      mode           = $mode"
echo ""



##########################################################################
# Llamada a quantizer.py
##########################################################################

if [[ ! -f "$CONF" ]]; then
  printf "[run_cuantizer.sh]  ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

printf "[run_cuantizer.sh]  ${YELLOW}Running quantizer.py${NC}\n"
printf "\n"

python3 "$SENSE_FXP_QUANTIZER_DIR/quantizer.py" \
 --smaps-npy-path="$SENS_MAPS_NPY_PATH" \
 --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH" \
 --output-smaps-path="$OUTPUT_SMAPS_PATH" \
 --output-y-path="$OUTPUT_Y_PATH" \
 --NB=$NB \
 --NBF=$NBF \
 --signed="$signed" \
 --mode="$mode"


printf "\n"
printf "[run_cuantizer.sh]  ${GREEN}Done${NC}\n"
printf "\n"
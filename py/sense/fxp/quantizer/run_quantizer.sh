# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_quantizer.sh] ${GREEN}Running quantizer${NC}\n"
printf "\n"

CONF="$GLOBAL_CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run_quantizer.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"

###########################################################################
# Validaciones de listas
###########################################################################
if [[ ${#NB_LIST[@]} -ne ${#NBF_LIST[@]} ]]; then
  printf "[run_quantizer.sh] ${RED}ERROR: NB_LIST and NBF_LIST must have the same length${NC}\n"
  exit 1
fi

###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
PHANTOM_DIR="$SENSE_GEN_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased.npy"
###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo "[run_quantizer.sh] Files to quantize:"
echo "[run_quantizer.sh]     Sensitivity Maps  : $SENS_MAPS_NPY_PATH"
echo "[run_quantizer.sh]     Aliased coils     : $ALIASED_COILS_NPY_PATH"
echo ""
echo "[run_quantizer.sh] Conf file read       : $CONF"
echo ""

##########################################################################
# Loop de regresión
##########################################################################
for idx in "${!NB_LIST[@]}"; do
  NB="${NB_LIST[$idx]}"
  NBF="${NBF_LIST[$idx]}"

  OUTPUT_DIR="$SENSE_FXP_QUANTIZER_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB${NB}_NBF${NBF}"
  mkdir -p "$OUTPUT_DIR"

  OUTPUT_NAME_S="S_q_NB${NB}_NBF${NBF}"
  OUTPUT_NAME_y="y_q_NB${NB}_NBF${NBF}"

  OUTPUT_SMAPS_PATH="$OUTPUT_DIR/$OUTPUT_NAME_S"
  OUTPUT_Y_PATH="$OUTPUT_DIR/$OUTPUT_NAME_y"

  printf "[run_quantizer.sh] ${YELLOW}Running quantizer.py for NB=${NB}, NBF=${NBF}${NC}\n"
  echo "[run_quantizer.sh] Parameters for quantization:"
  echo "[run_quantizer.sh]     NB             = $NB"
  echo "[run_quantizer.sh]     NBF            = $NBF"
  echo "[run_quantizer.sh]     signed         = $signed"
  echo "[run_quantizer.sh]     mode           = $mode"
  echo ""

  python3 "$SENSE_FXP_QUANTIZER_DIR/quantizer.py" \
    --smaps-npy-path="$SENS_MAPS_NPY_PATH" \
    --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH" \
    --output-smaps-path="$OUTPUT_SMAPS_PATH" \
    --output-y-path="$OUTPUT_Y_PATH" \
    --NB="$NB" \
    --NBF="$NBF" \
    --signed="$signed" \
    --mode="$mode"

  if [[ $? -ne 0 ]]; then
    printf "[run_quantizer.sh] ${RED}ERROR running quantizer.py for NB=${NB}, NBF=${NBF}${NC}\n"
    exit 1
  fi

  printf "[run_quantizer.sh] ${YELLOW}Done for NB=${NB}, NBF=${NBF}${NC}\n"
  printf "\n"
done


printf "[run_quantizer.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
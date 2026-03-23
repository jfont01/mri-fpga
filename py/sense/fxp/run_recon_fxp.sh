# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_recon_fxp.sh] ${GREEN}Running run_recon_fxp.sh${NC}\n"
printf "\n"


CONF_PATH="$SENSE_FXP_CONF"
CONF="$CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run_recon_fxp.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"

###########################################################################
# Validaciones de listas
###########################################################################
if [[ ${#NB_LIST[@]} -ne ${#NBF_LIST[@]} ]]; then
  printf "[run_recon_fxp.sh] ${RED}ERROR: NB_LIST and NBF_LIST must have the same length${NC}\n"
  exit 1
fi

###########################################################################
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
PHANTOM_DIR="$SENSE_GEN_DIR/pipes/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap_N${N}.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased_Af${AF}_axis${AXIS}.npy"

if [[ ! -f "$SENS_MAPS_NPY_PATH" ]]; then
  printf "[run_recon_fxp.sh] ${RED}ERROR: smaps .npy not found: $SENS_MAPS_NPY_PATH${NC}\n"
  exit 1
fi

if [[ ! -f "$ALIASED_COILS_NPY_PATH" ]]; then
  printf "[run_recon_fxp.sh] ${RED}ERROR: aliased coils not found: $ALIASED_COILS_NPY_PATH${NC}\n"
  exit 1
fi
###########################################################################
#  Parámetros cargados desde config.conf
###########################################################################
echo "[run_recon_fxp.sh] Files to quantize:"
echo "[run_recon_fxp.sh]     Sensitivity Maps  : $SENS_MAPS_NPY_PATH"
echo "[run_recon_fxp.sh]     Aliased coils     : $ALIASED_COILS_NPY_PATH"
echo ""
echo "[run_recon_fxp.sh] Conf file read       : $CONF"
echo ""

##########################################################################
# Loop de regresión
##########################################################################
SENSE_FXP_RECON_DIR="$SENSE_FXP_DIR/recon"
for idx in "${!NB_LIST[@]}"; do
  NB="${NB_LIST[$idx]}"
  NBF="${NBF_LIST[$idx]}"

  RECON_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
  CASE_NAME="NB${NB}_NBF${NBF}"

  OUTPUT_DIR="$SENSE_FXP_RECON_DIR/$RECON_NAME/$CASE_NAME"
  mkdir -p "$OUTPUT_DIR"

  NPZ_DIR="$SENSE_FXP_QUANTIZER_DIR/$RECON_NAME/$CASE_NAME"
  
  S_NPZ_PATH="$NPZ_DIR/S_q_NB${NB}_NBF${NBF}.npz"
  Y_NPZ_PATH="$NPZ_DIR/y_q_NB${NB}_NBF${NBF}.npz"
  

  printf "[run_recon_fxp.sh] ${YELLOW}Running fxp_sense.py ${NC}\n"
  echo ""

  python3 "$SENSE_FXP_DIR/fxp_sense.py" \
    --smaps-npy-path="$SENS_MAPS_NPY_PATH" \
    --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH" \
    --smaps-npz-path="$S_NPZ_PATH" \
    --aliased-coils-npz-path="$Y_NPZ_PATH" \
    --output-dir="$OUTPUT_DIR"              \
    --NB="$NB" \
    --NBF="$NBF" \
    --signed="$signed" \
    --mode="$mode"

  if [[ $? -ne 0 ]]; then
    printf "[run_recon_fxp.sh] ${RED}ERROR running quantizer.py for NB=${NB}, NBF=${NBF}${NC}\n"
    exit 1
  fi

  printf "[run_recon_fxp.sh] ${YELLOW}Done for NB=${NB}, NBF=${NBF}${NC}\n"
  printf "\n"
done


printf "[run_recon_fxp.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
set -Eeuo pipefail
# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_sense_fxp.sh] ${GREEN}Running run_sense_fxp.sh${NC}\n"
printf "\n"


CONF_PATH="$GLOBAL_CONF_PATH"
CONF="$CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run_sense_fxp.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"


##########################################################################
# Loop de regresión
##########################################################################
SENSE_FXP_RECON_DIR="$PY_SENSE_FXP_DIR/output"
for i in "${!NB_K_LIST[@]}"; do
  NB_K="${NB_K_LIST[$i]}"
  NBF_K="${NBF_K_LIST[$i]}"
  for j in "${!NB_A_LIST[@]}"; do
    NB_A="${NB_A_LIST[$j]}"
    NBF_A="${NBF_A_LIST[$j]}"
      for k in "${!NB_B_LIST[@]}"; do
        NB_B="${NB_B_LIST[$k]}"
        NBF_B="${NBF_B_LIST[$k]}"
        for l in "${!NB_S_LIST[@]}"; do
          NB_S="${NB_S_LIST[$l]}"
          NBF_S="${NBF_S_LIST[$l]}"

          S_NPZ_PATH="$PY_QUANTIZER_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/S/NB${NB_S}_NBF${NBF_S}/S.npz"
          Y_NPZ_PATH="$PY_FFT2D_FXP_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB${NB_K}_NBF${NBF_K}/coils_aliased.npz"

          OUTPUT_DIR="$SENSE_FXP_RECON_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB_Y${NB_K}_NBF_Y${NBF_K}NB_S${NB_S}_NBF_S${NBF_S}_NB_A${NB_A}_NBF_A${NBF_A}NB_B${NB_B}_NBF_B${NBF_B}"
          mkdir -p "$OUTPUT_DIR"

          printf "[run_sense_fxp.sh] ${YELLOW}Running fxp_sense.py with NB_Y=${NB_K} NBF_Y=${NBF_K} NB_S=${NB_S} NBF_S=${NBF_S} NB_A=${NB_A} NBF_A=${NBF_A} NB_B=${NB_B} NBF_B=${NBF_B}${NC}\n"

          python3 "$PY_SENSE_FXP_DIR/fxp_sense_runner.py"       \
          --smaps-npz-path="$S_NPZ_PATH"                        \
          --aliased-coils-npz-path="$Y_NPZ_PATH"                \
          --output-dir="$OUTPUT_DIR"                            \
          --NB-A="$NB_A"                                        \
          --NBF-A="$NBF_A"                                      \
          --NB-B="$NB_B"                                        \
          --NBF-B="$NBF_B"                                      \
          --max-workers=$MAX_WORKERS                            \
          --chunksize=$CHUNKSIZE                                \
          --save-images="$SAVE_COMPARISION_IMAGES"              

          printf "[run_sense_fxp.sh] ${YELLOW}Done for NB_Y=${NB_K} NBF_Y=${NBF_K} NB_S=${NB_S} NBF_S=${NBF_S} NB_A=${NB_A} NBF_A=${NBF_A} NB_B=${NB_B} NBF_B=${NBF_B}${NC}\n"
          printf "\n"
      done
    done
  done
done


printf "[run_sense_fxp.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
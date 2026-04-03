set -Eeuo pipefail
# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_reporter.sh] ${GREEN}Running run_reporter.sh${NC}\n"
printf "\n"

CONF_PATH="$GLOBAL_CONF_PATH"
CONF="$CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run_reporter.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi
source "$CONF"

printf "[run_reporter.sh] SNR_DB_THRESHOLD:       $SNR_DB_THRESHOLD\n"
printf "\n"

PHANTOM_DIR="$PY_GEN_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased.npy"


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

          OUTPUT_DIR="$PY_SENSE_REPORTER_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB_Y${NB_K}_NBF_Y${NBF_K}NB_S${NB_S}_NBF_S${NBF_S}_NB_A${NB_A}_NBF_A${NBF_A}NB_B${NB_B}_NBF_B${NBF_B}"
          mkdir -p "$OUTPUT_DIR"

          FP_DIR="$PY_SENSE_ROOT/fp/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
          FXP_DIR="$PY_SENSE_ROOT/fxp/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB_Y${NB_K}_NBF_Y${NBF_K}NB_S${NB_S}_NBF_S${NBF_S}_NB_A${NB_A}_NBF_A${NBF_A}NB_B${NB_B}_NBF_B${NBF_B}"
          S_NPZ_PATH="$PY_QUANTIZER_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/S/NB${NB_S}_NBF${NBF_S}/S.npz"
          Y_NPZ_PATH="$PY_FFT2D_FXP_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/NB${NB_K}_NBF${NBF_K}/coils_aliased.npz"
          printf "[run_reporter.sh] ${YELLOW}Running sense_reporter_runner.py with NB_Y=${NB_K} NBF_Y=${NBF_K} NB_S=${NB_S} NBF_S=${NBF_S} NB_A=${NB_A} NBF_A=${NBF_A} NB_B=${NB_B} NBF_B=${NBF_B}${NC}\n"


          python3 "$PY_SENSE_ROOT/reporter/sense_reporter_runner.py"      \
          --smaps-npz-path="$S_NPZ_PATH"                                  \
          --aliased-coils-npz-path="$Y_NPZ_PATH"                          \
          --smaps-npy-path="$SENS_MAPS_NPY_PATH"                          \
          --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH"              \
          --snr-db-threshold="$SNR_DB_THRESHOLD"                          \
          --fp-dir="$FP_DIR"                                              \
          --fxp-dir="$FXP_DIR"                                            \
          --output-dir="$OUTPUT_DIR"                                      \
          --save-images="$SAVE_COMPARISION_IMAGES" 
          printf "[run_reporter.sh] ${YELLOW}Done for NB_Y=${NB_K} NBF_Y=${NBF_K} NB_S=${NB_S} NBF_S=${NBF_S} NB_A=${NB_A} NBF_A=${NBF_A} NB_B=${NB_B} NBF_B=${NBF_B}${NC}\n"
          printf "\n"
      done
    done
  done
done


printf "[run_reporter.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
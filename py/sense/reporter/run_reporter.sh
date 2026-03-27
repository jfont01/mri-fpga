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

PHANTOM_DIR="$SENSE_GEN_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap.npy"
ALIASED_COILS_NPY_PATH="$PHANTOM_DIR/coils-aliased/coil_aliased.npy"


SENSE_FXP_RECON_DIR="$SENSE_FXP_DIR/output"
for idx in "${!NB_LIST[@]}"; do
  NB="${NB_LIST[$idx]}"
  NBF="${NBF_LIST[$idx]}"

  RECON_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
  CASE_NAME="NB${NB}_NBF${NBF}"

  OUTPUT_DIR="$SENSE_ROOT/reporter/$RECON_NAME/$CASE_NAME"
  mkdir -p "$OUTPUT_DIR"

  NPZ_DIR="$SENSE_FXP_QUANTIZER_DIR/output/$RECON_NAME/$CASE_NAME"
  
  S_NPZ_PATH="$NPZ_DIR/S_q_NB${NB}_NBF${NBF}.npz"
  Y_NPZ_PATH="$NPZ_DIR/y_q_NB${NB}_NBF${NBF}.npz"
    
  FP_DIR="$SENSE_ROOT/fp/output/$RECON_NAME"
  FXP_DIR="$SENSE_ROOT/fxp/output/$RECON_NAME/$CASE_NAME"

  printf "[run_reporter.sh] ${YELLOW}Running sense_reporter_runner.py with NB=${NB} NBF=${NBF}${NC}\n"


    python3 "$SENSE_ROOT/reporter/sense_reporter_runner.py"         \
    --smaps-npz-path="$S_NPZ_PATH"                                  \
    --aliased-coils-npz-path="$Y_NPZ_PATH"                          \
    --smaps-npy-path="$SENS_MAPS_NPY_PATH"                          \
    --aliased-coils-npy-path="$ALIASED_COILS_NPY_PATH"              \
    --snr-db-threshold="$SNR_DB_THRESHOLD"                          \
    --fp-dir="$FP_DIR"                                              \
    --fxp-dir="$FXP_DIR"                                            \
    --output-dir="$OUTPUT_DIR"                                      \
    --save-images="$SAVE_COMPARISION_IMAGES" 


  printf "[run_reporter.sh] ${YELLOW}Done for NB=${NB}, NBF=${NBF}${NC}\n"
  printf "\n"
done


printf "[run_reporter.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
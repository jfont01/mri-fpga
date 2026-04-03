set -Eeuo pipefail

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_iff2d_fxp.sh] ${GREEN}Running iff2d${NC}\n"
printf "\n"

CONF="$GLOBAL_CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run_iff2d_fxp.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"


##########################################################################
# Loop de regresión
##########################################################################
INPUT_ROOT="$PY_QUANTIZER_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}/k"
OUTPUT_DIR="$PY_FFT2D_FXP_DIR/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
for j in "${!NB_K_LIST[@]}"; do
  NB_K="${NB_K_LIST[$j]}"
  NBF_K="${NBF_K_LIST[$j]}"

  CASE_NAME="NB${NB_K}_NBF${NBF_K}"

  K_SPACE_NPZ_PATH="$INPUT_ROOT/$CASE_NAME/k.npz"

  python3 "$PY_FFT2D_FXP_DIR/fxp_ifft2d_runner.py"                        \
    --stimulus-npz-path="$K_SPACE_NPZ_PATH"                               \
    --NB=$NB_K                                                            \
    --NBF=$NBF_K                                                          \
    --output-dir="$OUTPUT_DIR/NB${NB_K}_NBF${NBF_K}"                      \
    --save-images="$SAVE_COMPARISION_IMAGES"

  printf "[run_iff2d_fxp.sh] ${YELLOW}Done for NB_K=${NB_K}, NBF_K=${NBF_K}${NC}\n"
  printf "\n"
done



printf "[run_iff2d_fxp.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
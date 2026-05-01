set -Eeuo pipefail

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
#  Construcción de paths de phantom / smaps / coils aliasadas
###########################################################################
PHANTOM_DIR="$PY_GEN_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
SENS_MAPS_NPY_PATH="$PHANTOM_DIR/sens-maps/smap.npy"
ALIASED_COILS_K_SPACE_NPY_PATH="$PHANTOM_DIR/coils-kspace-aliased/kspace_undersampled.npy"



##########################################################################
# Loop de regresión
##########################################################################
OUTPUT_ROOT="$PY_QUANTIZER_ROOT/output/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
mkdir -p "$OUTPUT_ROOT"
for i in "${!NB_S_LIST[@]}"; do
  NB_S="${NB_S_LIST[$i]}"
  NBF_S="${NBF_S_LIST[$i]}"
  for j in "${!NB_K_LIST[@]}"; do
    NB_K="${NB_K_LIST[$j]}"
    NBF_K="${NBF_K_LIST[$j]}"


    python3 "$PY_QUANTIZER_ROOT/quantizer.py"                               \
      --smaps-npy-path="$SENS_MAPS_NPY_PATH"                                \
      --aliased-coils-k-space-npy-path="$ALIASED_COILS_K_SPACE_NPY_PATH"    \
      --output-root="$OUTPUT_ROOT"                                          \
      --NB_S="$NB_S"                                                        \
      --NBF_S="$NBF_S"                                                      \
      --NB_K="$NB_K"                                                        \
      --NBF_K="$NBF_K"                                                      \
      --signed="$signed"                                                    \
      --mode="$mode"

    printf "[run_quantizer.sh] ${YELLOW}Done for NB_S=${NB_S}, NBF_S=${NBF_S}, NB_K=${NB_K}, NBF_K=${NBF_K}${NC}\n"
    printf "\n"
  done
done


printf "[run_quantizer.sh] ${GREEN}Regression finished successfully${NC}\n"
printf "\n"
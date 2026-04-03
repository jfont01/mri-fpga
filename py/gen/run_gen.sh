# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[run_gen.sh] ${GREEN}Running stimulus generator${NC}\n"
printf "\n"

CONF_PATH="$GLOBAL_CONF_PATH"
CONF="$CONF_PATH"

source "$CONF"

# Opcionales (evita que -u reviente si no están)
USE_FFTSHIFT="${USE_FFTSHIFT:-False}"
NORM="${NORM:-None}"

if [[ "$AXIS" != "x" && "$AXIS" != "y" ]]; then
  echo "Error: AXIS must be 'x' or 'y'"
  exit 1
fi

echo "[run_gen.sh]    Running sense pipe generator with:"
echo "  N       = ${N}"
echo "  AF      = ${AF}"
echo "  L       = ${L}"
echo "  AXIS    = ${AXIS}"
echo "  PHANTOM = ${PHANTOM}"
echo ""

###########################################################################
# Paths de salida
###########################################################################
OUTPUT_DIR="$PY_GEN_ROOT/output"
OUTPUT_CASE_DIR="$OUTPUT_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

COILS_ABS_DIR="$OUTPUT_CASE_DIR/coils-abs"
COILS_ALIASED_ZPADDED_DIR="$OUTPUT_CASE_DIR/coils-aliased-zpadded"
COILS_ALIASED_DIR="$OUTPUT_CASE_DIR/coils-aliased"
COILS_KSPACE_DIR="$OUTPUT_CASE_DIR/coils-kspace"
COILS_KSPACE_ALIASED_DIR="$OUTPUT_CASE_DIR/coils-kspace-aliased"
COILS_KSPACE_ALIASED_ZPADDED_DIR="$OUTPUT_CASE_DIR/coils-kspace-aliased-zpadded"
SENS_MAPS_DIR="$OUTPUT_CASE_DIR/sens-maps"
TARGET_DIR="$OUTPUT_CASE_DIR/target"

mkdir -p                                \
  "$COILS_ABS_DIR"                      \
  "$COILS_ALIASED_DIR"                  \
  "$COILS_ALIASED_ZPADDED_DIR"          \
  "$COILS_KSPACE_DIR"                   \
  "$COILS_KSPACE_ALIASED_DIR"           \
  "$COILS_KSPACE_ALIASED_ZPADDED_DIR"   \
  "$SENS_MAPS_DIR"                      \
  "$TARGET_DIR"


###########################################################################
# 1) Phantom
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_phantom.py..."
python3 "$PY_GEN_ROOT/gen_phantom.py" \
  -N="$N" -A="$AMP" \
  --rings-period="$RINGS_PERIOD" \
  --phase0="$PHASE0" \
  --phantom-type="$PHANTOM" \
  --output-name="$TARGET_DIR/phantom" \
  --input-npy="$PY_NPY_DATA_ROOT/$PHANTOM.npy" \
  --cmap="$PHANTOM_CMAP"

###########################################################################
# 2) Mapas de sensibilidad
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_smaps.py..."
python3 "$PY_GEN_ROOT/gen_smaps.py" \
  -N="$N" \
  -L="$L" \
  --radius-factor="$RADIUS_FACTOR" \
  --sigma-factor="$SIGMA_FACTOR" \
  --phase-scale="$PHASE_SCALE" \
  --output-name="$SENS_MAPS_DIR/smap" \
  --cmap="$SMAPS_CMAP"
echo ""

###########################################################################
# 3) Imágenes de bobina
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_coils.py..."
python3 "$PY_GEN_ROOT/gen_coils.py" \
  --phantom="$TARGET_DIR/phantom.npy" \
  --sens-maps="$SENS_MAPS_DIR/smap.npy" \
  --output-npy="$COILS_ABS_DIR/coil_images.npy" \
  --png-prefix="$COILS_ABS_DIR/coil_image"
echo ""

###########################################################################
# 4) K-space de bobinas
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_kspace.py..."
KSPACE_ARGS=()
if [[ "${USE_FFTSHIFT}" == "True" || "${USE_FFTSHIFT}" == "true" ]]; then
  KSPACE_ARGS+=(--fftshift)
fi
if [[ -n "${NORM}" && "${NORM}" != "None" ]]; then
  KSPACE_ARGS+=(--norm "$NORM")
fi

python3 "$PY_GEN_ROOT/gen_kspace.py" \
  --input="$COILS_ABS_DIR/coil_images.npy" \
  --output="$COILS_KSPACE_DIR/kspace.npy" \
  "${KSPACE_ARGS[@]}"

###########################################################################
# 5) K-space submuestreado (reducido)
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_aliased_kspace.py ..."
python3 "$PY_GEN_ROOT/gen_aliased_kspace.py" \
  --input-npy="$COILS_KSPACE_DIR/kspace.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled" \
  --cmap="$KSPACE_CMAP"
echo ""

###########################################################################
# 6) K-space submuestreado zero-padded (full size)
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_aliased_kspace.py (zpadded) ..."
python3 "$PY_GEN_ROOT/gen_aliased_kspace.py" \
  --input-npy="$COILS_KSPACE_DIR/kspace.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd" \
  --cmap="$KSPACE_CMAP" \
  --full
echo ""


###########################################################################
# 7) Imágenes aliasadas de bobina (reducido)
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_coil_aliased.py ..."
python3 "$PY_GEN_ROOT/gen_coil_aliased.py" \
  --input-npy="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled.npy" \
  --output-name="$COILS_ALIASED_DIR/coil_aliased" \
  --cmap="$ALIASED_CMAP"
echo ""

###########################################################################
# 8) Imágenes aliasadas de bobina (zpadded)
###########################################################################
echo ""
echo "[run_gen.sh]    Running gen_coil_aliased.py ..."
python3 "$PY_GEN_ROOT/gen_coil_aliased.py" \
  --input-npy="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd.npy" \
  --output-name="$COILS_ALIASED_ZPADDED_DIR/coil_aliased_zpadd" \
  --cmap="$ALIASED_CMAP"
echo ""


printf "[run_gen.sh]    ${GREEN}Stimulus generated successfully${NC}\n"
printf "\n"
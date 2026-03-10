#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_MRI_ROOT:? Enviroment variable FPGA_MRI_ROOT must be defined}"

# Uso: ./run_gen.sh config.conf
if [[ $# -ne 1 ]]; then
  echo "Uso: $0 <config.conf>"
  echo "  config.conf"
  exit 1
fi

CONF="$1"

if [[ ! -f "$CONF" ]]; then
  echo "Error: $CONF missing"
  exit 1
fi

echo "Loading config from $CONF"
# shellcheck disable=SC1090
source "$CONF"

###########################################################################
#  Parámetros requeridos
###########################################################################

: "${N:? N missing in .conf}"
: "${AF:? Af missing in .conf}"
: "${L:? L missing in .conf}"
: "${AXIS:? AXIS missing in .conf}"
: "${PHANTOM:? PHANTOM missing in .conf}"
: "${AMP:? AMP is missing in .conf}"
: "${RINGS_PERIOD:? RINGS_PERIOD is missing in .conf}"
: "${PHASE0:? PHASE0 is missing in .conf}"
: "${RADIUS_FACTOR:? RADIUS_FACTOR is missing in .conf}"
: "${SIGMA_FACTOR:? SIGMA_FACTOR is missing in .conf}"
: "${PHASE_SCALE:? PHASE_SCALE is missing in .conf}"
: "${PHANTOM_CMAP:? PHANTOM_CMAP is missing in .conf}"
: "${SMAPS_CMAP:? SMAPS_CMAP is missing in .conf}"
: "${KSPACE_CMAP:? KSPACE_CMAP is missing in .conf}"
: "${ALIASED_CMAP:? ALIASED_CMAP is missing in .conf}"


if [[ "$AXIS" != "x" && "$AXIS" != "y" ]]; then
  echo "Error: AXIS must be 'x' or 'y'"
  exit 1
fi

case "$PHANTOM" in
  two-disks|rings|two-gaussian-dots|shepp-logan)
    ;;
  *)
    echo "Error: PHANTOM debe ser uno de {two-disks, rings, two-gaussian-dots, shepp-logan}, recibido: '$PHANTOM'"
    exit 1
    ;;
esac


###########################################################################
#  Paths de salida
###########################################################################

echo "Running sense pipe generator with:"
echo "  N     = ${N}"
echo "  AF    = ${AF}"
echo "  L     = ${L}"
echo "  AXIS  = ${AXIS}"
echo "  PHANTOM = ${PHANTOM}"
echo ""

BASE="${FPGA_MRI_ROOT%/}/py/sense/gen/pipes"

RUN_DIR="$BASE/N${N}_Af${AF}_L${L}_axis${AXIS}_phantom${PHANTOM}"

COILS_ABS_DIR="$RUN_DIR/coils-abs"
COILS_ALIASED_DIR="$RUN_DIR/coils-aliased"
COILS_KSPACE_DIR="$RUN_DIR/coils-kspace"
COILS_KSPACE_ALIASED_DIR="$RUN_DIR/coils-kspace-aliased-${AXIS}"
COILS_KSPACE_ALIASED_ZPADDED_DIR="$RUN_DIR/coils-kspace-aliased-zpadded-${AXIS}"
SENS_MAPS_DIR="$RUN_DIR/sens-maps"
TARGET_DIR="$RUN_DIR/target"

mkdir -p \
  "$COILS_ABS_DIR" \
  "$COILS_ALIASED_DIR" \
  "$COILS_KSPACE_DIR" \
  "$COILS_KSPACE_ALIASED_DIR" \
  "$COILS_KSPACE_ALIASED_ZPADDED_DIR" \
  "$SENS_MAPS_DIR" \
  "$TARGET_DIR"

echo "RUN_DIR: $RUN_DIR"
echo "Created/checked:"
echo "  $COILS_ABS_DIR"
echo "  $COILS_ALIASED_DIR"
echo "  $COILS_KSPACE_DIR"
echo "  $COILS_KSPACE_ALIASED_DIR"
echo "  $COILS_KSPACE_ALIASED_ZPADDED_DIR"
echo "  $SENS_MAPS_DIR"
echo "  $TARGET_DIR"
echo ""

###########################################################################
# 1) Phantom
###########################################################################

echo "Running gen_phantom.py"
python3 gen_phantom.py \
  -N="$N" -A="$AMP" \
  --rings-period="$RINGS_PERIOD" \
  --phase0="$PHASE0" \
  --phantom-type="$PHANTOM" \
  --output-name="$TARGET_DIR/phantom" \
  --cmap="$PHANTOM_CMAP"

###########################################################################
# 2) Mapas de sensibilidad
###########################################################################

echo "Running gen_smaps.py"
python3 gen_smaps.py \
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

echo "Running gen_coils.py"
python3 gen_coils.py \
  --phantom="$TARGET_DIR/phantom_N$N.npy" \
  --sens-maps="$SENS_MAPS_DIR/smap_N$N.npy" \
  --output-npy="$COILS_ABS_DIR/coil_images_N$N.npy" \
  --png-prefix="$COILS_ABS_DIR/coil_image_N$N"
echo ""

###########################################################################
# 4) K-space de bobinas
###########################################################################

echo "Running gen_kspace.py"
KSPACE_ARGS=()
if [[ "${USE_FFTSHIFT}" == "True" ]]; then
  KSPACE_ARGS+=(--fftshift)
fi
if [[ -n "${NORM:-}" && "${NORM}" != "None" ]]; then
  KSPACE_ARGS+=(--norm "$NORM")
fi

python3 gen_kspace.py \
  --input="$COILS_ABS_DIR/coil_images_N$N.npy" \
  --output="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  "${KSPACE_ARGS[@]}"

###########################################################################
# 5) K-space submuestreado
###########################################################################

echo "Running gen_aliased_kspace.py (reducido)"
python3 gen_aliased_kspace.py \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled_axis${AXIS}" \
  --cmap="$KSPACE_CMAP"
echo ""

###########################################################################
# 6) K-space submuestreado zero-padded
###########################################################################

echo "Running gen_aliased_kspace.py (zero padded, full size)"
python3 gen_aliased_kspace.py \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}" \
  --cmap="$KSPACE_CMAP" \
  --full
echo ""

# Sufijo correcto según eje 
FULL_SUFFIX="fullNy"
if [[ "$AXIS" == "x" ]]; then
  FULL_SUFFIX="fullNx"
fi

ALIased_INPUT_NPY="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}_Af${AF}_${FULL_SUFFIX}.npy"

###########################################################################
# 7) Imágenes aliasadas de bobina
###########################################################################

echo "Running gen_coil_aliased.py"
python3 gen_coil_aliased.py \
  --input-npy="$ALIased_INPUT_NPY" \
  --output-name="$COILS_ALIASED_DIR/coil_aliased_Af${AF}_axis${AXIS}" \
  --cmap="$ALIASED_CMAP"
echo ""
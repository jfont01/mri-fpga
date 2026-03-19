#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_MRI_ROOT:?Env var FPGA_MRI_ROOT must be defined}"
: "${PY_ROOT:?Env var PY_ROOT must be defined (source set_env.sh)}"
: "${SENSE_ROOT:?Env var SENSE_ROOT must be defined (source set_env.sh)}"
: "${SENSE_GEN_DIR:?Env var SENSE_GEN_DIR must be defined (source set_env.sh)}"
: "${SENSE_FP_DIR:?Env var SENSE_FP_DIR must be defined (source set_env.sh)}"
: "${SENSE_GEN_CONF:?Env var SENSE_GEN_CONF must be defined (source set_env.sh)}"
: "${NPY_DATA_ROOT:?Env var NPY_DATA_ROOT must be defined (source set_env.sh)}"

echo ""
echo "==================================================================="
echo " Running fp stimulus generator"
echo "==================================================================="
echo ""

CONF_PATH="$SENSE_GEN_CONF"
CONF="$CONF_PATH"

echo "Paths (from environment):"
echo "FPGA_MRI_ROOT   = $FPGA_MRI_ROOT"
echo "PY_ROOT         = $PY_ROOT"
echo "SENSE_ROOT      = $SENSE_ROOT"
echo "SENSE_GEN_DIR   = $SENSE_GEN_DIR"
echo "SENSE_FP_DIR    = $SENSE_FP_DIR"
echo "NPY_DATA_ROOT   = $NPY_DATA_ROOT"
echo "CONF_PATH       = $CONF_PATH"
echo ""

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: config file not found: $CONF"
  exit 1
fi

source "$CONF"

###########################################################################
# Parámetros requeridos desde config.conf
###########################################################################
: "${N:? N missing in .conf}"
: "${AF:? Af missing in .conf}"
: "${L:? L missing in .conf}"
: "${AXIS:? AXIS missing in .conf}"
: "${PHANTOM:? PHANTOM missing in .conf}"
: "${AMP:? AMP missing in .conf}"
: "${RINGS_PERIOD:? RINGS_PERIOD missing in .conf}"
: "${PHASE0:? PHASE0 missing in .conf}"
: "${RADIUS_FACTOR:? RADIUS_FACTOR missing in .conf}"
: "${SIGMA_FACTOR:? SIGMA_FACTOR missing in .conf}"
: "${PHASE_SCALE:? PHASE_SCALE missing in .conf}"
: "${PHANTOM_CMAP:? PHANTOM_CMAP missing in .conf}"
: "${SMAPS_CMAP:? SMAPS_CMAP missing in .conf}"
: "${KSPACE_CMAP:? KSPACE_CMAP missing in .conf}"
: "${ALIASED_CMAP:? ALIASED_CMAP missing in .conf}"

# Opcionales (evita que -u reviente si no están)
USE_FFTSHIFT="${USE_FFTSHIFT:-False}"
NORM="${NORM:-None}"

if [[ "$AXIS" != "x" && "$AXIS" != "y" ]]; then
  echo "Error: AXIS must be 'x' or 'y'"
  exit 1
fi

echo "Running sense pipe generator with:"
echo "  N       = ${N}"
echo "  AF      = ${AF}"
echo "  L       = ${L}"
echo "  AXIS    = ${AXIS}"
echo "  PHANTOM = ${PHANTOM}"
echo ""

###########################################################################
# Paths de salida
###########################################################################
BASE="$SENSE_GEN_DIR/pipes"
RUN_DIR="$BASE/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

COILS_ABS_DIR="$RUN_DIR/coils-abs"
COILS_ALIASED_ZPADDED_DIR="$RUN_DIR/coils-aliased-zpadded"
COILS_ALIASED_DIR="$RUN_DIR/coils-aliased"
COILS_KSPACE_DIR="$RUN_DIR/coils-kspace"
COILS_KSPACE_ALIASED_DIR="$RUN_DIR/coils-kspace-aliased-${AXIS}"
COILS_KSPACE_ALIASED_ZPADDED_DIR="$RUN_DIR/coils-kspace-aliased-zpadded-${AXIS}"
SENS_MAPS_DIR="$RUN_DIR/sens-maps"
TARGET_DIR="$RUN_DIR/target"

mkdir -p \
  "$COILS_ABS_DIR" \
  "$COILS_ALIASED_DIR" \
  "$COILS_ALIASED_ZPADDED_DIR" \
  "$COILS_KSPACE_DIR" \
  "$COILS_KSPACE_ALIASED_DIR" \
  "$COILS_KSPACE_ALIASED_ZPADDED_DIR" \
  "$SENS_MAPS_DIR" \
  "$TARGET_DIR"

echo "RUN_DIR: $RUN_DIR"
echo ""

###########################################################################
# Scripts python (rutas absolutas)
###########################################################################
GEN_PHANTOM_PY="$SENSE_GEN_DIR/gen_phantom.py"
GEN_SMAPS_PY="$SENSE_GEN_DIR/gen_smaps.py"
GEN_COILS_PY="$SENSE_GEN_DIR/gen_coils.py"
GEN_KSPACE_PY="$SENSE_GEN_DIR/gen_kspace.py"
GEN_ALIASED_KSPACE_PY="$SENSE_GEN_DIR/gen_aliased_kspace.py"
GEN_COIL_ALIASED_PY="$SENSE_GEN_DIR/gen_coil_aliased.py"

for f in "$GEN_PHANTOM_PY" "$GEN_SMAPS_PY" "$GEN_COILS_PY" "$GEN_KSPACE_PY" "$GEN_ALIASED_KSPACE_PY" "$GEN_COIL_ALIASED_PY"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing python script: $f"
    exit 1
  fi
done

###########################################################################
# 1) Phantom
###########################################################################
echo "Running gen_phantom.py"
python3 "$GEN_PHANTOM_PY" \
  -N="$N" -A="$AMP" \
  --rings-period="$RINGS_PERIOD" \
  --phase0="$PHASE0" \
  --phantom-type="$PHANTOM" \
  --output-name="$TARGET_DIR/phantom" \
  --input-npy="$NPY_DATA_ROOT/$PHANTOM.npy" \
  --cmap="$PHANTOM_CMAP"

###########################################################################
# 2) Mapas de sensibilidad
###########################################################################
echo "Running gen_smaps.py"
python3 "$GEN_SMAPS_PY" \
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
python3 "$GEN_COILS_PY" \
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
if [[ "${USE_FFTSHIFT}" == "True" || "${USE_FFTSHIFT}" == "true" ]]; then
  KSPACE_ARGS+=(--fftshift)
fi
if [[ -n "${NORM}" && "${NORM}" != "None" ]]; then
  KSPACE_ARGS+=(--norm "$NORM")
fi

python3 "$GEN_KSPACE_PY" \
  --input="$COILS_ABS_DIR/coil_images_N$N.npy" \
  --output="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  "${KSPACE_ARGS[@]}"

###########################################################################
# 5) K-space submuestreado (reducido)
###########################################################################
echo "Running gen_aliased_kspace.py (reducido)"
python3 "$GEN_ALIASED_KSPACE_PY" \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled_axis${AXIS}" \
  --cmap="$KSPACE_CMAP"
echo ""

###########################################################################
# 6) K-space submuestreado zero-padded (full size)
###########################################################################
echo "Running gen_aliased_kspace.py (zero padded, full size)"
python3 "$GEN_ALIASED_KSPACE_PY" \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$AF" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}" \
  --cmap="$KSPACE_CMAP" \
  --full
echo ""

# Sufijo correcto según eje (asumiendo tu naming actual)
FULL_SUFFIX="fullNy"
if [[ "$AXIS" == "x" ]]; then
  FULL_SUFFIX="fullNx"
fi

ALIASED_ZPADD_INPUT_NPY="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}_Af${AF}_${FULL_SUFFIX}.npy"
ALIASED_INPUT_NPY="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled_axis${AXIS}_Af${AF}.npy"

###########################################################################
# 7) Imágenes aliasadas de bobina (reducido)
###########################################################################
echo "Running gen_coil_aliased.py (reducido)"
python3 "$GEN_COIL_ALIASED_PY" \
  --input-npy="$ALIASED_INPUT_NPY" \
  --output-name="$COILS_ALIASED_DIR/coil_aliased_Af${AF}_axis${AXIS}" \
  --cmap="$ALIASED_CMAP"
echo ""

###########################################################################
# 8) Imágenes aliasadas de bobina (zpadded)
###########################################################################
echo "Running gen_coil_aliased.py (zpadded)"
python3 "$GEN_COIL_ALIASED_PY" \
  --input-npy="$ALIASED_ZPADD_INPUT_NPY" \
  --output-name="$COILS_ALIASED_ZPADDED_DIR/coil_aliased_Af${AF}_axis${AXIS}_zpadd" \
  --cmap="$ALIASED_CMAP"
echo ""
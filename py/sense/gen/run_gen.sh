set -euo pipefail

: "${FPGA_MRI_ROOT}"


if [[ $# -ne 5 ]]; then
  echo "Uso: $0 <N> <Af> <L> <axis> <phantom>"
  echo "  N       : tamaño de imagen (p.ej. 256)"
  echo "  Af      : factor de aceleración (p.ej. 2, 4)"
  echo "  L       : número de bobinas"
  echo "  axis    : eje de undersampling en k-space: x o y"
  echo "  phantom : tipo de phantom: two-disks o rings"
  echo "Ejemplo: $0 256 2 4 y two-disks"
  exit 1
fi

N="$1"
Af="$2"
L="$3"
AXIS="$4"
PHANTOM="$5"

if [[ "$AXIS" != "x" && "$AXIS" != "y" ]]; then
  echo "Error: axis debe ser 'x' o 'y', recibido: '$AXIS'"
  exit 1
fi



echo "Running sense pipe generator with N = ${N}, Af = ${Af}, L = ${L}, axis = ${AXIS}"
echo ""

BASE="${FPGA_MRI_ROOT%/}/py/sense/gen"

RUN_DIR="$BASE/N${N}_Af${Af}_L${L}_axis${AXIS}_phantom${PHANTOM}"

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
echo "Created: $COILS_ABS_DIR"
echo "Created: $COILS_ALIASED_DIR"
echo "Created: $COILS_KSPACE_DIR"
echo "Created: $COILS_KSPACE_ALIASED_DIR"
echo "Created: $COILS_KSPACE_ALIASED_ZPADDED_DIR"
echo "Created: $SENS_MAPS_DIR"
echo "Created: $TARGET_DIR"
echo ""

echo "Running gen_phantom.py"
python3 gen_phantom.py \
  -N="$N" -A=1.0 \
  --rings-period=16.0 \
  --phase0=0.0 \
  --phantom-type="$PHANTOM" \
  --output-name="$TARGET_DIR/concentric_rings" \
  --cmap=gray
echo 

echo "Running gen_smaps.py"
python3 gen_smaps.py \
  -N="$N" \
  -L="$L" \
  --radius-factor=1.0 \
  --sigma-factor=0.8 \
  --phase-scale=1.0 \
  --output-name="$SENS_MAPS_DIR/smap" \
  --cmap=gray
echo ""

echo "Running gen_coils.py"
python3 gen_coils.py \
  --phantom="$TARGET_DIR/concentric_rings_N$N.npy" \
  --sens-maps="$SENS_MAPS_DIR/smap_N$N.npy" \
  --output-npy="$COILS_ABS_DIR/coil_images_N$N.npy" \
  --png-prefix="$COILS_ABS_DIR/coil_image_N$N"
echo ""

echo "Running gen_kspace.py"
python3 gen_kspace.py \
  --input="$COILS_ABS_DIR/coil_images_N$N.npy" \
  --output="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --fftshift
echo ""

echo "Running gen_aliased_kspace.py (reducido)"
python3 gen_aliased_kspace.py \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$Af" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_DIR/kspace_undersampled_axis${AXIS}" \
  --cmap=gray
echo ""

echo "Running gen_aliased_kspace.py (zero padded, full size)"
python3 gen_aliased_kspace.py \
  --input-npy="$COILS_KSPACE_DIR/kspace_N$N.npy" \
  --acc-factor="$Af" \
  --axis="$AXIS" \
  --output-name="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}" \
  --cmap=gray \
  --full
echo ""

FULL_SUFFIX="fullNy"
if [[ "$AXIS" == "x" ]]; then
  FULL_SUFFIX="fullNx"
fi

ALIased_INPUT_NPY="$COILS_KSPACE_ALIASED_ZPADDED_DIR/kspace_undersampled_zpadd_axis${AXIS}_Af${Af}_${FULL_SUFFIX}.npy"

echo "Running gen_coil_aliased.py"
python3 gen_coil_aliased.py \
  --input-npy="$ALIased_INPUT_NPY" \
  --output-name="$COILS_ALIASED_DIR/coil_aliased_Af${Af}_axis${AXIS}" \
  --cmap=gray
echo ""
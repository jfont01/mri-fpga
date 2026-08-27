#!/usr/bin/env bash
set -euo pipefail

: "${BOOTGEN:?BOOTGEN is required}"
: "${FSBL:?FSBL is required}"
: "${PMUFW:?PMUFW is required}"
: "${BITSTREAM:?BITSTREAM is required}"
: "${ELF:?ELF is required}"
: "${BIF_OUT:?BIF_OUT is required}"
: "${BOOT_BIN:?BOOT_BIN is required}"

for f in "$FSBL" "$PMUFW" "$BITSTREAM" "$ELF"; do
  [[ -s "$f" ]] || { echo "ERROR: boot input missing/empty: $f" >&2; exit 1; }
done

command -v "$BOOTGEN" >/dev/null 2>&1 || { echo "ERROR: bootgen not found: $BOOTGEN" >&2; exit 1; }
mkdir -p "$(dirname "$BIF_OUT")" "$(dirname "$BOOT_BIN")"

cat > "$BIF_OUT" <<BIF
v0_boot:
{
    [pmufw_image] ${PMUFW}
    [bootloader,destination_cpu=a53-0] ${FSBL}
    [destination_device=pl] ${BITSTREAM}
    [destination_cpu=a53-0,exception_level=el-3] ${ELF}
}
BIF

echo "BIF: $BIF_OUT"
cat "$BIF_OUT"
"$BOOTGEN" -arch zynqmp -image "$BIF_OUT" -w on -o "$BOOT_BIN"
echo "BOOT.BIN: $BOOT_BIN"

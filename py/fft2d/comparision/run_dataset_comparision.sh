#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=python3

op="ifft"
coil=0
MODE="round"      # o el modo que uses para los twiddles
N=512

# --- valores a barrer ---
NB_LIST=(21 22 23 24 25 26 27 28)
NBF_LIST=(20 21 22 23 24 25 26 27)
# NB_LIST y NBF_LIST deben tener la misma longitud

# Flags lógicos (como strings para el bash)
CAST="True"
SHIFT_RIGHT_STAGE="True"
DEBUG="False"

NPY_INPUT="$PY_NPY_DATA/MTR_030_2d_kspace.npy"

for i in "${!NB_LIST[@]}"; do
  NB=${NB_LIST[$i]}
  NBF=${NBF_LIST[$i]}

  # normalmente redondeás al mismo formato
  NB_ROUND=$NB
  NBF_ROUND=$NBF

  # outdir distinto por configuración para no pisar PNGs
  OUTDIR="$SCRIPT_DIR/fft2d_skmtea_sweep"
  mkdir -p "$OUTDIR"

  echo ">>> Ejecutando op=$op coil=$coil NB=$NB NBF=$NBF OUTDIR=$OUTDIR"

  ARGS=(
    --coil "$coil"
    --op "$op"
    --N "$N"
    --NB "$NB"
    --NBF "$NBF"
    --mode "$MODE"
    --NB-round "$NB_ROUND"
    --NBF-round "$NBF_ROUND"
    --npy-input "$NPY_INPUT"
    --outdir "$OUTDIR"
  )

  # Solo añadimos los flags si están a "True"
  if [ "$CAST" = "True" ]; then
    ARGS+=(--cast)
  fi

  if [ "$SHIFT_RIGHT_STAGE" = "True" ]; then
    ARGS+=(--shift-right-stage)
  fi

  if [ "$DEBUG" = "True" ]; then
    ARGS+=(--debug)
  fi

  "$PYTHON" "$SCRIPT_DIR/dataset_comparision.py" "${ARGS[@]}"
done
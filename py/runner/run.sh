#!/usr/bin/env bash
set -Eeuo pipefail

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

START_TS_EPOCH=$(date +%s)
DELETE_CASE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete-case)
      DELETE_CASE=1
      shift
      ;;
    *)
      printf "[run.sh] ${RED}ERROR: unknown argument: %s${NC}\n" "$1"
      printf "[run.sh] Usage: %s [--delete-case]\n" "$0"
      exit 1
      ;;
  esac
done

LOG_DIR="./_log"
mkdir -p "$LOG_DIR"

RUN_TS="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_FILE="$LOG_DIR/run_$RUN_TS.log"

exec > >(tee -a "$LOG_FILE") 2>&1

printf "\n"
printf "[run.sh] Log file: %s\n" "$LOG_FILE"
printf "[run.sh] ${GREEN}=======================================================================================${NC}\n"
printf "[run.sh] ${GREEN}Running GLOBAL: stimulus gen, quantize, fp and fxp SENSE reconstruction and comparision${NC}\n"
printf "[run.sh] ${GREEN}=======================================================================================${NC}\n"
printf "\n"

CONF_PATH="$GLOBAL_CONF_PATH"
CONF="$CONF_PATH"

if [[ ! -f "$CONF" ]]; then
  printf "[run.sh] ${RED}ERROR: config file not found: $CONF${NC}\n"
  exit 1
fi

source "$CONF"

######################################### CASE_DIRS #########################################
CASE_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

OUTPUT_STIMULUS_GEN_CASE_DIR="$PY_GEN_ROOT/output/$CASE_NAME"
OUTPUT_QUANTIZER_CASE_DIR="$PY_QUANTIZER_ROOT/output/$CASE_NAME"
OUTPUT_FFT2D_FXP_CASE_DIR="$PY_FFT2D_FXP_DIR/output/$CASE_NAME"
OUTPUT_SENSE_FP_CASE_DIR="$PY_SENSE_FP_DIR/output/$CASE_NAME"
OUTPUT_SENSE_FXP_CASE_DIR="$PY_SENSE_FXP_DIR/output/$CASE_NAME"
OUTPUT_REPORTER_CASE_DIR="$PY_SENSE_REPORTER_DIR/output/$CASE_NAME"

RUNNER_OUTPUT_CASE_DIR="$PY_RUNNER/output/$CASE_NAME"

######################################## DELETE CASE ########################################
if [[ "$DELETE_CASE" -eq 1 ]]; then
  printf "[run.sh] ${YELLOW}--delete-case detected. Deleting previous case directories...${NC}\n"

  for dir in \
    "$OUTPUT_STIMULUS_GEN_CASE_DIR" \
    "$OUTPUT_QUANTIZER_CASE_DIR"    \
    "$OUTPUT_FFT2D_FXP_CASE_DIR"    \
    "$OUTPUT_SENSE_FP_CASE_DIR"     \
    "$OUTPUT_SENSE_FXP_CASE_DIR"    \
    "$OUTPUT_REPORTER_CASE_DIR"     \
    "$RUNNER_OUTPUT_CASE_DIR"
  do
    if [[ -d "$dir" ]]; then
      printf "[run.sh]    Deleting: %s\n" "$dir"
      rm -rf "$dir"
    else
      printf "[run.sh]    Not found, skipping delete: %s\n" "$dir"
    fi
  done

  printf "[run.sh] ${GREEN}Delete finished. Exiting.${NC}\n"
  exit 0
fi

########################################### run_gen.sh ###########################################
if [[ -d "$OUTPUT_STIMULUS_GEN_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_STIMULUS_GEN_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_gen.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_STIMULUS_GEN_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_gen.sh...\n"
  source "$PY_GEN_RUN"
fi
echo ""

######################################## run_quantizer.sh ########################################
if [[ -d "$OUTPUT_QUANTIZER_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_QUANTIZER_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_quantizer.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_QUANTIZER_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_quantizer.sh...\n"
  source "$PY_QUANTIZER_RUN"
fi
echo ""

######################################## run_fft2d_fxp.sh ########################################
if [[ -d "$OUTPUT_FFT2D_FXP_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_FFT2D_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_fft2d_fxp.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_FFT2D_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_fft2d_fxp.sh...\n"
  source "$PY_FFT2D_FXP_RUN"
fi
echo ""

######################################## run_sense_fp.sh ########################################
if [[ -d "$OUTPUT_SENSE_FP_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_SENSE_FP_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_sense_fp.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_SENSE_FP_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_sense_fp.sh...\n"
  source "$PY_SENSE_FP_RUN"
fi
echo ""

######################################## run_sense_fxp.sh ########################################
if [[ -d "$OUTPUT_SENSE_FXP_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_SENSE_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_sense_fxp.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_SENSE_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_sense_fxp.sh...\n"
  source "$PY_SENSE_FXP_RUN"
fi
echo ""

######################################## run_reporter.sh #########################################
if [[ -d "$OUTPUT_REPORTER_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_REPORTER_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_reporter.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_REPORTER_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_reporter.sh...\n"
  source "$PY_SENSE_REPORTER_RUN"
fi
echo ""

######################################## COPY RESULTS TO runner/output #########################################
printf "[run.sh]    Copying generated outputs to: %s\n" "$RUNNER_OUTPUT_CASE_DIR"
mkdir -p "$RUNNER_OUTPUT_CASE_DIR"

copy() {
  local src="$1"
  local dst="$2"

  if [[ -d "$src" ]]; then
    printf "[run.sh]    Copying directory: %s -> %s\n" "$src" "$dst"
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  else
    printf "[run.sh]    Directory not found, skipping copy: %s\n" "$src"
  fi
}

copy "$OUTPUT_STIMULUS_GEN_CASE_DIR" "$RUNNER_OUTPUT_CASE_DIR/gen"
copy "$OUTPUT_QUANTIZER_CASE_DIR"    "$RUNNER_OUTPUT_CASE_DIR/quantizer"
copy "$OUTPUT_FFT2D_FXP_CASE_DIR"    "$RUNNER_OUTPUT_CASE_DIR/fft2d_fxp"
copy "$OUTPUT_SENSE_FP_CASE_DIR"     "$RUNNER_OUTPUT_CASE_DIR/sense_fp"
copy "$OUTPUT_SENSE_FXP_CASE_DIR"    "$RUNNER_OUTPUT_CASE_DIR/sense_fxp"
copy "$OUTPUT_REPORTER_CASE_DIR"     "$RUNNER_OUTPUT_CASE_DIR/reporter"

cp -f "$LOG_FILE" "$RUNNER_OUTPUT_CASE_DIR/run.log"

######################################## EXECUTION TIME #########################################
END_TS_EPOCH=$(date +%s)
ELAPSED_SEC=$((END_TS_EPOCH - START_TS_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

printf "\n"
printf "[run.sh] ${GREEN}Finished successfully${NC}\n"
printf "[run.sh] Total execution time: %02d:%02d:%02d\n" "$ELAPSED_H" "$ELAPSED_M" "$ELAPSED_S"
printf "[run.sh] Snapshot saved in: %s\n" "$RUNNER_OUTPUT_CASE_DIR"
printf "\n"
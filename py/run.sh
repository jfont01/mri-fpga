#!/usr/bin/env bash
set -Eeuo pipefail

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_DIR="./log"
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

######################################## run_gen.sh ########################################
OUTPUT_STIMULUS_GEN_DIR="$SENSE_GEN_DIR/output"
OUTPUT_STIMULUS_GEN_CASE_DIR="$OUTPUT_STIMULUS_GEN_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

if [[ -d "$OUTPUT_STIMULUS_GEN_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_STIMULUS_GEN_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_gen.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_STIMULUS_GEN_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_gen.sh...\n"
  source $SENSE_GEN_RUN
fi


SENS_MAPS_NPY_PATH="$OUTPUT_STIMULUS_GEN_CASE_DIR/sens-maps/smap.npy"
ALIASED_COILS_NPY_PATH="$OUTPUT_STIMULUS_GEN_CASE_DIR/coils-aliased/coil_aliased.npy"

if [[ ! -f "$SENS_MAPS_NPY_PATH" ]]; then
  printf "[run.sh]    ${RED}ERROR:${NC} file not found: $SENS_MAPS_NPY_PATH"
  exit 1
fi

if [[ ! -f "$ALIASED_COILS_NPY_PATH" ]]; then
  printf "[run.sh]    ${RED}ERROR:${NC} file not found: $ALIASED_COILS_NPY_PATH"
  exit 1
fi
echo ""

######################################## run_quantizer.sh ########################################
OUTPUT_QUANTIZER_DIR="$SENSE_FXP_QUANTIZER_DIR/output"
OUTPUT_QUANTIZER_CASE_DIR="$OUTPUT_QUANTIZER_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

if [[ -d "$OUTPUT_QUANTIZER_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_QUANTIZER_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_quantizer.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_QUANTIZER_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_quantizer.sh...\n"
  source $SENSE_QUANTIZER_RUN
fi

for idx in "${!NB_LIST[@]}"; do
  NB="${NB_LIST[$idx]}"
  NBF="${NBF_LIST[$idx]}"

  OUTPUT_QUANTIZER_CASE_BITS_DIR="$OUTPUT_QUANTIZER_CASE_DIR/NB${NB}_NBF${NBF}"

  if [[ ! -f "$OUTPUT_QUANTIZER_CASE_BITS_DIR/S_q_NB${NB}_NBF${NBF}.npz" ]]; then
    printf "[run.sh]    ${RED}ERROR:${NC} file not found: $OUTPUT_QUANTIZER_CASE_BITS_DIR/S_q_NB${NB}_NBF${NBF}.npz"
    exit 1
  fi

  if [[ ! -f "$OUTPUT_QUANTIZER_CASE_BITS_DIR/y_q_NB${NB}_NBF${NBF}.npz" ]]; then
    printf "[run.sh]    ${RED}ERROR:${NC} file not found: $OUTPUT_QUANTIZER_CASE_BITS_DIR/y_q_NB${NB}_NBF${NBF}.npz"
    exit 1
  fi
done
echo ""

######################################## run_sense_fp.sh ########################################
OUTPUT_SENSE_FP_DIR="$SENSE_FP_DIR/output"
OUTPUT_SENSE_FP_CASE_DIR="$OUTPUT_SENSE_FP_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

if [[ -d "$OUTPUT_SENSE_FP_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_SENSE_FP_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_sense_fp.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_SENSE_FP_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_sense_fp.sh...\n"
  source $SENSE_FP_RUN
fi

if [[ ! -f "$OUTPUT_SENSE_FP_CASE_DIR/I/I.npy" ]]; then
  printf "[run.sh]    ${RED}ERROR:${NC} file not found: $OUTPUT_SENSE_FP_CASE_DIR/I/I.npy"
  exit 1
fi
echo ""
######################################## run_sense_fxp.sh ########################################
OUTPUT_SENSE_FXP_DIR="$SENSE_FXP_DIR/output"
OUTPUT_SENSE_FXP_CASE_DIR="$OUTPUT_SENSE_FXP_DIR/N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"

if [[ -d "$OUTPUT_SENSE_FXP_CASE_DIR" ]]; then
  printf "[run.sh]    Directory exists: $OUTPUT_SENSE_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Skipping run_sense_fp.sh...\n"
else
  printf "[run.sh]    Directory does not exist: $OUTPUT_SENSE_FXP_CASE_DIR${NC}\n"
  printf "[run.sh]    Running run_sense_fp.sh...\n"
  source $SENSE_FXP_RUN
fi


for idx in "${!NB_LIST[@]}"; do
  NB="${NB_LIST[$idx]}"
  NBF="${NBF_LIST[$idx]}"

  OUTPUT_SENSE_FXP_CASE_BITS_DIR="$OUTPUT_SENSE_FXP_CASE_DIR/NB${NB}_NBF${NBF}"

  if [[ ! -f "$OUTPUT_SENSE_FXP_CASE_BITS_DIR/global_fxp_report.rpt" ]]; then
    printf "[run.sh]    ${RED}ERROR:${NC} file not found: $OUTPUT_SENSE_FXP_CASE_BITS_DIR/global_fxp_report.rpt"
    exit 1
  fi

done


######################################## run_reporter.sh ########################################
printf "[run.sh]    Running run_reporter.sh...\n"
source $SENSE_REPORTER_RUN




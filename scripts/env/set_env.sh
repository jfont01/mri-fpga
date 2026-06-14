# set_env.sh

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "\n"
printf "[set_env.sh] ${GREEN}Loading environment...${NC}\n"
printf "\n"

if [[ -z "${FPGA_MRI_ROOT:-}" ]]; then
  printf "[set_env.sh] ${RED}ERROR:${NC} FPGA_MRI_ROOT is not defined\n"
  return 1
fi

PROJECT_PATHS_SCRIPT="$FPGA_MRI_ROOT/scripts/env/project_paths.sh"

if [[ ! -f "$PROJECT_PATHS_SCRIPT" ]]; then
  printf "[set_env.sh] ${RED}ERROR:${NC} project paths script not found: %s\n" "$PROJECT_PATHS_SCRIPT"
  return 1
fi

source "$PROJECT_PATHS_SCRIPT"

printf "[set_env.sh] Sourcing track manifest helper...\n"
source "$TRACK_MANIFEST_HELPER_SH"

printf "[set_env.sh] Sourcing Python virtual environment...\n"
source "$PY_VENV_ACTIVATE"

printf "[set_env.sh] Sourcing Vivado settings...\n"
source "$VIVADO_SETTINGS"
# ==============================================================================
# Convenience functions
# ==============================================================================
run_py_model() {
  bash "$PY_RUNNER_SCRIPT" "$@"
}

delete_py_model() {
  bash "$PY_RUNNER_SCRIPT" --delete-case
}

create_release() {
  bash "$RTL_CREATE_RELEASE_SCRIPT" "$@"
}

run_synthesis() {
  bash "$RTL_SYNTH_SCRIPT" "$@"
}

run_xsim() {
  bash "$RTL_XSIM_SCRIPT" "$@"
}

run_vm() {
  bash "$RTL_VM_SCRIPT" "$@"
}

printf "\n"
printf "[set_env.sh] ${GREEN}Environment loaded successfully.${NC}\n"
printf "\n"
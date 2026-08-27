SHELL := /bin/bash
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# =============================================================================
# Shared KV260 / K26 build infrastructure
#
# This file lives in:
#   bringup/scripts/Makefile
#
# Each version/project (v0, v1, new, ...) includes it from a tiny wrapper
# Makefile located in the project directory.
# =============================================================================


# -----------------------------------------------------------------------------
# Project / infrastructure roots
# -----------------------------------------------------------------------------

# Directory of the project/version currently being built.
PROJECT_ROOT ?= $(CURDIR)
PROJECT_ROOT := $(abspath $(PROJECT_ROOT))

# Directory containing this shared Makefile and all shared scripts.
INFRA_ROOT ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
INFRA_ROOT := $(abspath $(INFRA_ROOT))

include $(INFRA_ROOT)/config.mk


# -----------------------------------------------------------------------------
# Output colorizer
# -----------------------------------------------------------------------------

COLORIZER := $(INFRA_ROOT)/tools/colorize.sh


# -----------------------------------------------------------------------------
# Common Vivado command-line options
# -----------------------------------------------------------------------------

VIVADO_FLAGS := \
	-mode batch \
	-nojournal \
	-nolog \
	-notrace


# -----------------------------------------------------------------------------
# Generated directories
# -----------------------------------------------------------------------------

BUILD_DIR       := $(PROJECT_ROOT)/build
VIVADO_BUILD    := $(BUILD_DIR)/vivado
PLATFORM_WS     := $(BUILD_DIR)/vitis/platform_ws
APP_WS          := $(BUILD_DIR)/vitis/app_ws
PLATFORM_REPO_DIR   := $(PLATFORM_WS)/$(PLATFORM_NAME)/export
PLATFORM_EXPORT_DIR := $(PLATFORM_REPO_DIR)/$(PLATFORM_NAME)
PLATFORM_XPFM       := $(PLATFORM_EXPORT_DIR)/$(PLATFORM_NAME).xpfm
STAMP_DIR       := $(BUILD_DIR)/.stamps
RESULTS_DIR     := $(PROJECT_ROOT)/results


# -----------------------------------------------------------------------------
# Final artifacts
# -----------------------------------------------------------------------------

PROJECT_XPR     := $(VIVADO_BUILD)/$(PROJECT)/$(PROJECT).xpr

BIT_RESULT      := $(RESULTS_DIR)/$(PROJECT).bit
XSA_RESULT      := $(RESULTS_DIR)/$(PROJECT).xsa

XPFM_RESULT     := $(RESULTS_DIR)/$(PLATFORM_NAME).xpfm
ELF_RESULT      := $(RESULTS_DIR)/$(APP_NAME).elf

PSU_INIT_RESULT := $(RESULTS_DIR)/psu_init.tcl

FSBL_RESULT     := $(RESULTS_DIR)/boot/fsbl.elf
PMUFW_RESULT    := $(RESULTS_DIR)/boot/pmufw.elf

BOOT_BIF        := $(RESULTS_DIR)/boot/boot.bif
BOOT_BIN        := $(RESULTS_DIR)/boot/BOOT.BIN


# -----------------------------------------------------------------------------
# Incremental-build stamps
# -----------------------------------------------------------------------------

PROJECT_STAMP   := $(STAMP_DIR)/project
BD_STAMP        := $(STAMP_DIR)/block_design
SYNTH_STAMP     := $(STAMP_DIR)/synthesis
IMPL_STAMP      := $(STAMP_DIR)/implementation


# -----------------------------------------------------------------------------
# Infrastructure files
# -----------------------------------------------------------------------------

SHARED_MAKEFILE  := $(INFRA_ROOT)/Makefile
SHARED_CONFIG    := $(INFRA_ROOT)/config.mk
PROJECT_MAKEFILE := $(PROJECT_ROOT)/Makefile


# -----------------------------------------------------------------------------
# Environment passed to Vivado Tcl scripts
# -----------------------------------------------------------------------------

VIVADO_ENV = \
  PROJECT_NAME='$(PROJECT)' \
  PROJECT_DIR='$(VIVADO_BUILD)/$(PROJECT)' \
  PROJECT_XPR='$(PROJECT_XPR)' \
  PART='$(PART)' \
  BOARD_PART='$(BOARD_PART)' \
  BOARD_CONNECTIONS='$(BOARD_CONNECTIONS)' \
  BD_TCL='$(BD_TCL)' \
  RESULTS_DIR='$(RESULTS_DIR)' \
  JOBS='$(JOBS)' \
  THREADS='$(THREADS)'


# -----------------------------------------------------------------------------
# Firmware source dependency list
# -----------------------------------------------------------------------------

FIRMWARE_SOURCES := $(shell find "$(FIRMWARE_DIR)" -type f \
  \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' \
     -o -name '*.h' -o -name '*.hpp' \
     -o -name '*.S' -o -name '*.s' \) 2>/dev/null)


.PHONY: help all \
        project block_design synthesis implementation bit bitstream xsa \
        xpfm elf hardware software build \
        check uart jtag program_pl init_pspl load_elf reset run \
        xsdb hw_server \
        bootbin qspi \
        reports status print-vars \
        clean distclean rebuild test


# =============================================================================
# Help
# =============================================================================

help:
	@printf '%s\n' \
	  'KV260 / K26 shared build infrastructure' \
	  '' \
	  'Hardware (Vivado):' \
	  '  make project          Create the Vivado project' \
	  '  make block_design     Recreate/validate BD and wrapper' \
	  '  make synthesis        Run synth_1' \
	  '  make implementation   Run implementation through route_design' \
	  '  make bit              Generate results/$(PROJECT).bit' \
	  '  make bitstream        Alias of make bit' \
	  '  make xsa              Generate results/$(PROJECT).xsa with bit included' \
	  '  make hardware         Build everything through XSA' \
	  '' \
	  'Software (Vitis):' \
	  '  make xpfm             Create/build standalone Vitis platform from XSA' \
	  '  make elf              Create/build application from firmware/' \
	  '  make software         Build everything through ELF' \
	  '  make build            Full hardware + software build; DOES NOT touch board' \
	  '' \
	  'Board / debug:' \
	  '  make check            Check local tools, sources and configuration' \
	  '  make jtag             Verify PL/PSU/APU/A53 targets over JTAG' \
	  '  make uart             Open UART with minicom' \
	  '  make program_pl       Program only results/$(PROJECT).bit into PL' \
	  '  make init_pspl        Run psu_init + isolation removal + PL reset config' \
	  '  make load_elf         Reset A53#0 and download ELF; do not continue' \
	  '  make reset            Reset Cortex-A53 #0' \
	  '  make run              Full JTAG flow: BIT + PS/PL init + ELF + run' \
	  '  make xsdb             Open interactive XSDB shell' \
	  '  make hw_server        Start hw_server in foreground' \
	  '' \
	  'Boot / persistent flash:' \
	  '  make bootbin          Generate results/boot/BOOT.BIN' \
	  '  make qspi             Program QSPI (guarded; explicit confirmation required)' \
	  '' \
	  'Utilities:' \
	  '  make reports          Print implementation report paths' \
	  '  make status           Show which artifacts currently exist' \
	  '  make print-vars       Print all resolved paths/configuration' \
	  '  make clean            Remove build/, preserve results/' \
	  '  make distclean        Remove build/ and results/' \
	  '  make rebuild          distclean + build' \
	  '  make test             Run shared infrastructure static tests (if installed)' \
	  '' \
	  'Colors:' \
	  '  NO_COLOR=1 make ...   Disable colored output' \
	  '  FORCE_COLOR=1 ...     Force ANSI colors through pipes' \
	  '' \
	  'Useful overrides:' \
	  '  JOBS=1 THREADS=2 UART=/dev/ttyUSB1 BAUD=115200' \
	  '  VIVADO=/path/vivado VITIS=/path/vitis XSDB=/path/xsdb'


all: build


# =============================================================================
# Vivado hardware flow
# =============================================================================

project: $(PROJECT_STAMP)


$(PROJECT_STAMP): \
	$(SHARED_MAKEFILE) \
	$(SHARED_CONFIG) \
	$(PROJECT_MAKEFILE) \
	$(COLORIZER) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/project.tcl

	@mkdir -p "$(STAMP_DIR)" "$(RESULTS_DIR)"

	@command -v "$(VIVADO)" >/dev/null || { \
		echo 'ERROR: vivado not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/project.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@test -f "$(PROJECT_XPR)"

	@touch "$@"


block_design: $(BD_STAMP)


$(BD_STAMP): \
	$(PROJECT_STAMP) \
	$(BD_TCL) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/block_design.tcl

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/block_design.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@touch "$@"


synthesis: $(SYNTH_STAMP)


$(SYNTH_STAMP): \
	$(BD_STAMP) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/synthesis.tcl

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/synthesis.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@touch "$@"


implementation: $(IMPL_STAMP)


$(IMPL_STAMP): \
	$(SYNTH_STAMP) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/implementation.tcl

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/implementation.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@touch "$@"


bit bitstream: $(BIT_RESULT)


$(BIT_RESULT): \
	$(IMPL_STAMP) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/bit.tcl

	@mkdir -p "$(RESULTS_DIR)"

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/bit.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@test -s "$@"


xsa hardware: $(XSA_RESULT)


$(XSA_RESULT): \
	$(BIT_RESULT) \
	$(INFRA_ROOT)/vivado/common.tcl \
	$(INFRA_ROOT)/vivado/xsa.tcl

	@set -o pipefail; \
	$(VIVADO_ENV) \
	"$(VIVADO)" \
		$(VIVADO_FLAGS) \
		-source "$(INFRA_ROOT)/vivado/xsa.tcl" \
		2>&1 | bash "$(COLORIZER)"

	@test -s "$@"


# =============================================================================
# Vitis software flow
# =============================================================================

xpfm: $(XPFM_RESULT)


$(XPFM_RESULT): \
	$(XSA_RESULT) \
	$(INFRA_ROOT)/vitis/create_platform.py

	@command -v "$(VITIS)" >/dev/null || { \
		echo 'ERROR: vitis not found'; \
		exit 1; \
	}

	@mkdir -p "$(RESULTS_DIR)/boot"

	@set -o pipefail; \
	VITIS_WORKSPACE="$(PLATFORM_WS)" \
	PLATFORM_NAME="$(PLATFORM_NAME)" \
	XSA="$(XSA_RESULT)" \
	XPFM_OUT="$(XPFM_RESULT)" \
	RESULTS_DIR="$(RESULTS_DIR)" \
	"$(VITIS)" \
		-s "$(INFRA_ROOT)/vitis/create_platform.py" \
		2>&1 | bash "$(COLORIZER)"

	@test -s "$@"


# Generated as side effects of the Vitis platform build.

$(PSU_INIT_RESULT): $(XPFM_RESULT)
	@test -s "$@" || { \
		echo 'ERROR: psu_init.tcl was not exported by Vitis platform build'; \
		exit 1; \
	}


$(FSBL_RESULT): $(XPFM_RESULT)
	@test -s "$@" || { \
		echo 'ERROR: FSBL was not exported by Vitis platform build'; \
		exit 1; \
	}


$(PMUFW_RESULT): $(XPFM_RESULT)
	@test -s "$@" || { \
		echo 'ERROR: PMUFW was not exported by Vitis platform build'; \
		exit 1; \
	}


elf software build: $(ELF_RESULT)


$(ELF_RESULT): \
	$(XPFM_RESULT) \
	$(FIRMWARE_SOURCES) \
	$(INFRA_ROOT)/vitis/create_application.py

	@command -v "$(VITIS)" >/dev/null || { \
		echo 'ERROR: vitis not found'; \
		exit 1; \
	}

	@test -d "$(FIRMWARE_DIR)" || { \
		echo 'ERROR: firmware directory not found: $(FIRMWARE_DIR)'; \
		exit 1; \
	}

	@test -s "$(PLATFORM_XPFM)" || { \
		echo 'ERROR: internal Vitis platform XPFM not found:'; \
		echo '  $(PLATFORM_XPFM)'; \
		exit 1; \
	}

	@set -o pipefail; \
	PLATFORM_REPO_PATHS="$(PLATFORM_REPO_DIR)" \
	PLATFORM_REPO="$(PLATFORM_REPO_DIR)" \
	VITIS_WORKSPACE="$(APP_WS)" \
	APP_NAME="$(APP_NAME)" \
	XPFM="$(PLATFORM_XPFM)" \
	DOMAIN_NAME="$(DOMAIN_NAME)" \
	FIRMWARE_DIR="$(FIRMWARE_DIR)" \
	ELF_OUT="$(ELF_RESULT)" \
	"$(VITIS)" \
		-s "$(INFRA_ROOT)/vitis/create_application.py" \
		2>&1 | bash "$(COLORIZER)"

	@test -s "$@"

# =============================================================================
# Local checks / board interaction
# =============================================================================

check:
	@set -o pipefail; \
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	INFRA_ROOT="$(INFRA_ROOT)" \
	VIVADO="$(VIVADO)" \
	VITIS="$(VITIS)" \
	XSDB="$(XSDB)" \
	BOOTGEN="$(BOOTGEN)" \
	PROGRAM_FLASH="$(PROGRAM_FLASH)" \
	MINICOM="$(MINICOM)" \
	BD_TCL="$(BD_TCL)" \
	FIRMWARE_DIR="$(FIRMWARE_DIR)" \
	"$(INFRA_ROOT)/tools/check.sh" \
		2>&1 | bash "$(COLORIZER)"


jtag:
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/check_target.tcl" \
		2>&1 | bash "$(COLORIZER)"


uart:
	@command -v "$(MINICOM)" >/dev/null || { \
		echo 'ERROR: minicom not found'; \
		exit 1; \
	}

	@test -e "$(UART)" || { \
		echo 'ERROR: UART device not found: $(UART)'; \
		exit 1; \
	}

	@"$(MINICOM)" \
		-D "$(UART)" \
		-b "$(BAUD)"


program_pl: $(BIT_RESULT)
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	BITSTREAM="$(BIT_RESULT)" \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/program_pl.tcl" \
		2>&1 | bash "$(COLORIZER)"


init_pspl: $(PSU_INIT_RESULT)
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	PSU_INIT="$(PSU_INIT_RESULT)" \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/init_pspl.tcl" \
		2>&1 | bash "$(COLORIZER)"


load_elf: $(ELF_RESULT)
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	ELF="$(ELF_RESULT)" \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/load_elf.tcl" \
		2>&1 | bash "$(COLORIZER)"


reset:
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@set -o pipefail; \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/reset_a53.tcl" \
		2>&1 | bash "$(COLORIZER)"


run: \
	$(BIT_RESULT) \
	$(ELF_RESULT) \
	$(PSU_INIT_RESULT)

	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@echo 'TIP: keep "make uart" open in another terminal.'

	@set -o pipefail; \
	BITSTREAM="$(BIT_RESULT)" \
	PSU_INIT="$(PSU_INIT_RESULT)" \
	ELF="$(ELF_RESULT)" \
	DMA_BASE="$(DMA_BASE)" \
	"$(XSDB)" \
		"$(INFRA_ROOT)/jtag/run.tcl" \
		2>&1 | bash "$(COLORIZER)"


# Interactive XSDB: do NOT pipe through the colorizer because that would
# interfere with the interactive terminal.
xsdb:
	@command -v "$(XSDB)" >/dev/null || { \
		echo 'ERROR: xsdb not found'; \
		exit 1; \
	}

	@"$(XSDB)"


# hw_server stays attached directly to the terminal.
hw_server:
	@command -v "$(HW_SERVER)" >/dev/null || { \
		echo 'ERROR: hw_server not found'; \
		exit 1; \
	}

	@"$(HW_SERVER)"


# =============================================================================
# Boot image / persistent QSPI
# =============================================================================

bootbin: $(BOOT_BIN)


$(BOOT_BIN): \
	$(ELF_RESULT) \
	$(BIT_RESULT) \
	$(FSBL_RESULT) \
	$(PMUFW_RESULT) \
	$(INFRA_ROOT)/boot/create_bootbin.sh

	@set -o pipefail; \
	BOOTGEN="$(BOOTGEN)" \
	FSBL="$(FSBL_RESULT)" \
	PMUFW="$(PMUFW_RESULT)" \
	BITSTREAM="$(BIT_RESULT)" \
	ELF="$(ELF_RESULT)" \
	BIF_OUT="$(BOOT_BIF)" \
	BOOT_BIN="$(BOOT_BIN)" \
	"$(INFRA_ROOT)/boot/create_bootbin.sh" \
		2>&1 | bash "$(COLORIZER)"


qspi: $(BOOT_BIN)

	@if [[ "$(CONFIRM_QSPI)" != "YES" ]]; then \
		echo 'ERROR: QSPI programming is persistent and can replace KV260 SOM boot firmware.'; \
		echo 'Re-run only after verifying the intended flash/update flow:'; \
		echo '  make qspi CONFIRM_QSPI=YES FLASH_TYPE=<verified_type>'; \
		exit 1; \
	fi

	@if [[ -z "$(FLASH_TYPE)" ]]; then \
		echo 'ERROR: FLASH_TYPE is intentionally unset; do not guess it.'; \
		exit 1; \
	fi

	@command -v "$(PROGRAM_FLASH)" >/dev/null || { \
		echo 'ERROR: program_flash not found'; \
		exit 1; \
	}

	@echo '*** PERSISTENT QSPI PROGRAMMING ***'
	@echo "Image      : $(BOOT_BIN)"
	@echo "Flash type : $(FLASH_TYPE)"

	@set -o pipefail; \
	"$(PROGRAM_FLASH)" \
		-f "$(BOOT_BIN)" \
		-flash_type "$(FLASH_TYPE)" \
		-blank_check \
		-verify \
		-cable type xilinx_tcf url tcp:localhost:3121 \
		2>&1 | bash "$(COLORIZER)"


# =============================================================================
# Utility targets
# =============================================================================

reports: implementation
	@echo "Timing      : $(RESULTS_DIR)/$(PROJECT)_timing.rpt"
	@echo "Utilization : $(RESULTS_DIR)/$(PROJECT)_util.rpt"
	@echo "Routed DCP  : $(RESULTS_DIR)/$(PROJECT)_routed.dcp"


status:
	@set -o pipefail; \
	PROJECT_XPR="$(PROJECT_XPR)" \
	BIT="$(BIT_RESULT)" \
	XSA="$(XSA_RESULT)" \
	XPFM="$(XPFM_RESULT)" \
	ELF="$(ELF_RESULT)" \
	PSU_INIT="$(PSU_INIT_RESULT)" \
	FSBL="$(FSBL_RESULT)" \
	PMUFW="$(PMUFW_RESULT)" \
	BOOT_BIN="$(BOOT_BIN)" \
	"$(INFRA_ROOT)/tools/status.sh" \
		2>&1 | bash "$(COLORIZER)"


print-vars:
	@printf '%-24s %s\n' \
	  PROJECT_ROOT "$(PROJECT_ROOT)" \
	  INFRA_ROOT "$(INFRA_ROOT)" \
	  PROJECT "$(PROJECT)" \
	  APP_NAME "$(APP_NAME)" \
	  PLATFORM_NAME "$(PLATFORM_NAME)" \
	  PLATFORM_WS "$(PLATFORM_WS)" \
	  PLATFORM_REPO_DIR "$(PLATFORM_REPO_DIR)" \
	  PLATFORM_EXPORT_DIR "$(PLATFORM_EXPORT_DIR)" \
	  PLATFORM_XPFM "$(PLATFORM_XPFM)" \
	  DOMAIN_NAME "$(DOMAIN_NAME)" \
	  PART "$(PART)" \
	  BOARD_PART "$(BOARD_PART)" \
	  BD_TCL "$(BD_TCL)" \
	  FIRMWARE_DIR "$(FIRMWARE_DIR)" \
	  BUILD_DIR "$(BUILD_DIR)" \
	  RESULTS_DIR "$(RESULTS_DIR)" \
	  JOBS "$(JOBS)" \
	  THREADS "$(THREADS)" \
	  BIT "$(BIT_RESULT)" \
	  XSA "$(XSA_RESULT)" \
	  XPFM "$(XPFM_RESULT)" \
	  ELF "$(ELF_RESULT)" \
	  UART "$(UART)" \
	  DMA_BASE "$(DMA_BASE)" \
	  FLASH_TYPE "$(FLASH_TYPE)"


clean:
	@rm -rf "$(BUILD_DIR)"

	@echo "Removed: $(BUILD_DIR)"
	@echo "Preserved: $(RESULTS_DIR)"


distclean:
	@rm -rf \
		"$(BUILD_DIR)" \
		"$(RESULTS_DIR)"

	@echo "Removed: $(BUILD_DIR)"
	@echo "Removed: $(RESULTS_DIR)"


rebuild: distclean build


test:
	@if [[ -x "$(INFRA_ROOT)/tests/verify_infra.sh" ]]; then \
		PROJECT_ROOT="$(PROJECT_ROOT)" \
		INFRA_ROOT="$(INFRA_ROOT)" \
		"$(INFRA_ROOT)/tests/verify_infra.sh"; \
	else \
		echo 'ERROR: shared infrastructure test not installed:'; \
		echo '  $(INFRA_ROOT)/tests/verify_infra.sh'; \
		exit 1; \
	fi
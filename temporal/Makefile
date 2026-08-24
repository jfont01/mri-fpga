SHELL := /bin/bash

VIVADO ?= vivado
PROJECT ?= v0
JOBS ?= 8

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WORK_DIR := $(ROOT)/work
RESULTS_DIR := $(ROOT)/results
STAMP_DIR := $(ROOT)/.stamps
TCL_DIR := $(ROOT)/tcl
BD_TCL := $(ROOT)/build_block_design.tcl

PROJECT_STAMP := $(STAMP_DIR)/project.stamp
BD_STAMP := $(STAMP_DIR)/block_design.stamp
SYNTH_STAMP := $(STAMP_DIR)/synthesis.stamp
IMPL_STAMP := $(STAMP_DIR)/implementation.stamp
BIT_STAMP := $(STAMP_DIR)/bitstream.stamp
PLATFORM_STAMP := $(STAMP_DIR)/platform.stamp

.PHONY: help project block_design synthesis implementation bitstream platform build clean

help:
	@echo "Vivado build flow for $(PROJECT)"
	@echo
	@echo "  make project         Create a clean Vivado project"
	@echo "  make block_design    Recreate BD + wrapper + top"
	@echo "  make synthesis       Run synthesis"
	@echo "  make implementation  Run implementation through route_design"
	@echo "  make bitstream       Generate bitstream"
	@echo "  make platform        Export XSA with bitstream included"
	@echo "  make build           Run the complete flow"
	@echo "  make clean           Remove generated work/results/stamps"
	@echo
	@echo "Optional variables: VIVADO=/path/to/vivado JOBS=8 PROJECT=v0"

project: $(PROJECT_STAMP)

block_design: $(BD_STAMP)

synthesis: $(SYNTH_STAMP)

implementation: $(IMPL_STAMP)

bitstream: $(BIT_STAMP)

platform: $(PLATFORM_STAMP)

build: platform

# build_block_design.tcl is treated as source-of-truth. If it changes,
# rebuild the Vivado project from a clean work/ directory.
$(PROJECT_STAMP): $(TCL_DIR)/create_project.tcl $(TCL_DIR)/create_block_design.tcl $(TCL_DIR)/common.tcl $(BD_TCL) Makefile
	@mkdir -p $(STAMP_DIR) $(RESULTS_DIR)
	@echo "[PROJECT] Creating $(PROJECT)"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/create_project.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)"
	@touch $@

$(BD_STAMP): $(PROJECT_STAMP) $(TCL_DIR)/create_block_design.tcl $(TCL_DIR)/common.tcl $(BD_TCL)
	@echo "[BLOCK DESIGN] Recreating $(PROJECT) block design"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/create_block_design.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)"
	@touch $@

$(SYNTH_STAMP): $(BD_STAMP) $(TCL_DIR)/run_synthesis.tcl $(TCL_DIR)/common.tcl
	@echo "[SYNTHESIS] Running synth_1"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/run_synthesis.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)" "$(JOBS)"
	@touch $@

$(IMPL_STAMP): $(SYNTH_STAMP) $(TCL_DIR)/run_implementation.tcl $(TCL_DIR)/common.tcl
	@echo "[IMPLEMENTATION] Running impl_1 through route_design"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/run_implementation.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)" "$(JOBS)"
	@touch $@

$(BIT_STAMP): $(IMPL_STAMP) $(TCL_DIR)/run_bitstream.tcl $(TCL_DIR)/common.tcl
	@echo "[BITSTREAM] Generating bitstream"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/run_bitstream.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)" "$(JOBS)"
	@touch $@

$(PLATFORM_STAMP): $(BIT_STAMP) $(TCL_DIR)/export_platform.tcl $(TCL_DIR)/common.tcl
	@echo "[PLATFORM] Exporting $(PROJECT).xsa with bitstream"
	@$(VIVADO) -mode batch -nolog -nojournal \
		-source $(TCL_DIR)/export_platform.tcl \
		-tclargs "$(ROOT)" "$(PROJECT)"
	@touch $@

clean:
	@echo "[CLEAN] Removing generated files"
	@rm -rf "$(WORK_DIR)" "$(RESULTS_DIR)" "$(STAMP_DIR)" "$(ROOT)/.Xil"
	@rm -f "$(ROOT)"/*.jou "$(ROOT)"/*.log "$(ROOT)"/*.str

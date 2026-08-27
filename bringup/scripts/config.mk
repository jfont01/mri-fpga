# =============================================================================
# Shared defaults for KV260 / K26 projects
#
# This file lives in:
#   bringup/scripts/config.mk
#
# A per-version wrapper Makefile may override any of these variables BEFORE
# including $(INFRA_ROOT)/Makefile.
# =============================================================================


# -----------------------------------------------------------------------------
# Project identity
# -----------------------------------------------------------------------------

PROJECT       ?= v0

APP_NAME      ?= $(PROJECT)_application
PLATFORM_NAME ?= $(PROJECT)_platform

DOMAIN_NAME   ?= standalone_psu_cortexa53_0


# -----------------------------------------------------------------------------
# Device / board
# -----------------------------------------------------------------------------

PART ?= xck26-sfvc784-2LV-c

BOARD_PART ?= xilinx.com:kv260_som:part0:1.4

BOARD_CONNECTIONS ?= \
	som240_1_connector \
	xilinx.com:kv260_carrier:som240_1_connector:1.3


# -----------------------------------------------------------------------------
# Resource limits
#
# Conservative defaults for a machine with 16 GB RAM.
# -----------------------------------------------------------------------------

JOBS    ?= 1
THREADS ?= 2


# -----------------------------------------------------------------------------
# Tool commands
#
# Can be overridden from command line, for example:
#
# make build VIVADO=/tools/Xilinx/Vivado/2024.2/bin/vivado
# -----------------------------------------------------------------------------

VIVADO        ?= vivado
VITIS         ?= vitis

XSDB          ?= xsdb
HW_SERVER     ?= hw_server

BOOTGEN       ?= bootgen
PROGRAM_FLASH ?= program_flash

MINICOM       ?= minicom


# -----------------------------------------------------------------------------
# UART / debug
# -----------------------------------------------------------------------------

UART ?= /dev/ttyUSB1

BAUD ?= 115200

DMA_BASE ?= 0x80000000

HW_SERVER_URL ?= TCP:127.0.0.1:3121


# -----------------------------------------------------------------------------
# Persistent QSPI programming
#
# Intentionally disabled until the exact flash/update flow is verified.
# -----------------------------------------------------------------------------

FLASH_TYPE ?=

CONFIRM_QSPI ?= NO


# -----------------------------------------------------------------------------
# Project-local source locations
# -----------------------------------------------------------------------------

BD_TCL ?= \
	$(PROJECT_ROOT)/hardware/build_block_design.tcl

FIRMWARE_DIR ?= \
	$(PROJECT_ROOT)/firmware
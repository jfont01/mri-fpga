#!/usr/bin/env bash

set -u

# -----------------------------------------------------------------------------
# Terminal color filter
#
# Disable colors when:
#   - stdout is not a terminal
#   - NO_COLOR is set
#   - TERM=dumb
#
# FORCE_COLOR=1 can be used to force ANSI colors even through pipes.
# -----------------------------------------------------------------------------

if [[ -n "${NO_COLOR:-}" ]]; then
    cat
    exit 0
fi

if [[ "${TERM:-}" == "dumb" ]]; then
    cat
    exit 0
fi

if [[ ! -t 1 && -z "${FORCE_COLOR:-}" ]]; then
    cat
    exit 0
fi


RESET=$'\033[0m'

BOLD=$'\033[1m'

RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BLUE=$'\033[34m'
MAGENTA=$'\033[35m'
CYAN=$'\033[36m'


while IFS= read -r line || [[ -n "$line" ]]; do

    case "$line" in

        # ---------------------------------------------------------------------
        # Errors
        # ---------------------------------------------------------------------

        Traceback*|\
        *"ERROR:"*|\
        *"Error:"*|\
        *"TypeError:"*|\
        *"RuntimeError:"*|\
        *"ValueError:"*|\
        *"Exception:"*|\
        make:\ \*\*\**)
            printf '%s%s%s%s\n' \
                "$BOLD" "$RED" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Critical warnings
        # ---------------------------------------------------------------------

        *"CRITICAL WARNING"*|\
        *"Critical Warning"*|\
        *"Critical violations"*)
            printf '%s%s%s%s\n' \
                "$BOLD" "$MAGENTA" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Warnings
        # ---------------------------------------------------------------------

        WARNING:*|\
        WARNING::*|\
        *"WARNING:"*|\
        *"WARNING::"*)
            printf '%s%s%s\n' \
                "$YELLOW" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Tcl / Python scripts being executed
        # ---------------------------------------------------------------------

        source\ *)
            printf '%s%s%s%s\n' \
                "$BOLD" "$CYAN" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Important build actions
        # ---------------------------------------------------------------------

        "Creating Vivado project"*|\
        "Creating block design"*|\
        "Creating bare-metal Vitis platform"*|\
        "Creating Vitis application"*|\
        "Writing bitstream:"*|\
        "Writing hardware platform"*|\
        "Completing impl_1"*|\
        "Vivado implementation bitstream:"*|\
        "Copying bitstream to:"*)
            printf '%s%s%s\n' \
                "$CYAN" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Successful completion
        # ---------------------------------------------------------------------

        *"completed successfully"*|\
        *"Completed Successfully"*|\
        *"Complete!"*|\
        *"Successfully created"*|\
        *"generation complete"*|\
        *"Generation complete"*|\
        *"XPFM exported:"*|\
        *"ELF exported:"*|\
        *"CHECK: PASS"*|\
        READY\ *)
            printf '%s%s%s%s\n' \
                "$BOLD" "$GREEN" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Status output
        # ---------------------------------------------------------------------

        EMPTY\ *)
            printf '%s%s%s\n' \
                "$YELLOW" "$line" "$RESET"
            ;;

        "-----  "*)
            printf '%s%s%s\n' \
                "$RED" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Tool banners
        # ---------------------------------------------------------------------

        "****** Vivado"*|\
        "****** Vitis"*|\
        "****** XSDB"*|\
        "****** Xilinx"*|\
        "*** Running vivado"*)
            printf '%s%s%s%s\n' \
                "$BOLD" "$BLUE" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Interesting status lines
        # ---------------------------------------------------------------------

        "impl_1 status"*|\
        "impl_1 progress"*|\
        "synth_1 status"*|\
        "synth_1 progress"*|\
        "Current impl_1 state:"*)
            printf '%s%s%s\n' \
                "$CYAN" "$line" "$RESET"
            ;;


        # ---------------------------------------------------------------------
        # Everything else unchanged
        # ---------------------------------------------------------------------

        *)
            printf '%s\n' "$line"
            ;;

    esac

done
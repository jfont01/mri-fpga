#!/usr/bin/env bash

set -euo pipefail

if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    RESET=$'\033[0m'
else
    GREEN=''
    YELLOW=''
    RED=''
    RESET=''
fi

show() {
    local label="$1"
    local path="$2"

    if [[ -s "$path" ]]; then
        printf '%s%-6s%s %-12s %s\n' \
            "$GREEN" "READY" "$RESET" "$label" "$path"

    elif [[ -e "$path" ]]; then
        printf '%s%-6s%s %-12s %s\n' \
            "$YELLOW" "EMPTY" "$RESET" "$label" "$path"

    else
        printf '%s%-6s%s %-12s %s\n' \
            "$RED" "-----" "$RESET" "$label" "$path"
    fi
}

show XPR      "$PROJECT_XPR"
show BIT      "$BIT"
show XSA      "$XSA"
show XPFM     "$XPFM"
show ELF      "$ELF"
show PSU_INIT "$PSU_INIT"
show FSBL     "$FSBL"
show PMUFW    "$PMUFW"
show BOOT_BIN "$BOOT_BIN"
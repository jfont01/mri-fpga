#!/usr/bin/env bash
# platform_utils.sh -- detección de plataforma y selección de backend

detect_platform() {
    case "$(uname -s)" in
        Linux*)               echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "msys" ;;
        *)                    echo "unknown" ;;
    esac
}

# Backend de simulación según plataforma:
#   linux -> xsim (Vivado)
#   msys  -> iverilog
select_sim_backend() {
    local platform
    platform="$(detect_platform)"

    case "$platform" in
        linux) echo "xsim" ;;
        msys)  echo "iverilog" ;;
        *)     echo "" ;;
    esac
}

# ¿Está Vivado disponible en esta plataforma?
platform_has_vivado() {
    [[ "$(detect_platform)" == "linux" ]]
}
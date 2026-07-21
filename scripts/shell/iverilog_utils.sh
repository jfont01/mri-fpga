#!/usr/bin/env bash
# iverilog_utils.sh -- simulación RTL con Icarus Verilog (plataforma msys).
# Debe sourcearse desde set_env.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[iverilog_utils.sh] ERROR: this file must be sourced, not executed." >&2
    exit 1
fi

iverilog_log()   { echo "[iverilog_utils.sh] $*"; }
iverilog_error() { echo "[iverilog_utils.sh] ERROR: $*" >&2; }

# Detecta el módulo actual y devuelve el path a su flist de testbench,
# resolviendo ${PREFIX}_TB_FLIST igual que hace update_flist.
_iverilog_resolve_tb_flist() {
    local module_name prefix flist_var

    module_name="$(detect_current_module_for_flist)" || return 1

    if declare -F module_prefix >/dev/null 2>&1; then
        prefix="$(module_prefix "$module_name")"
    else
        prefix="$(echo "$module_name" | tr '[:lower:]' '[:upper:]')"
    fi

    flist_var="${prefix}_TB_FLIST"

    if [[ -z "${!flist_var:-}" ]]; then
        iverilog_error "variable no definida: $flist_var"
        iverilog_error "corré update_flist desde el módulo primero."
        return 1
    fi

    printf "%s\n" "${!flist_var}"
}

# Traduce un flist en formato XSIM (-i <dir>) a los argumentos que espera
# Icarus (-I<dir> pegado). Devuelve, por stdout, la lista de args ya lista
# para pasarle a iverilog: primero los -I<dir>, luego los archivos fuente.
_iverilog_flist_to_args() {
    local flist_file="$1"
    local line trimmed dir

    while IFS= read -r line || [[ -n "$line" ]]; do
        # saltear comentarios y líneas vacías
        trimmed="${line#"${line%%[![:space:]]*}"}"   # ltrim
        [[ -z "$trimmed" ]] && continue
        [[ "$trimmed" == \#* ]] && continue

        if [[ "$trimmed" == -i\ * ]]; then
            # "-i /ruta" (formato xsim) -> "-I/ruta" (formato icarus)
            dir="${trimmed#-i }"
            dir="${dir#"${dir%%[![:space:]]*}"}"      # ltrim del path
            printf -- "-I%s\n" "$dir"
        else
            # línea de archivo fuente: la pasamos tal cual
            printf "%s\n" "$trimmed"
        fi
    done < "$flist_file"
}

run_iverilog() {
    local case_dir="${1:-build}"
    local n_cycles="${2:-}"
    shift 2 2>/dev/null || shift $# # consume los dos primeros si existen

    local tb_flist iverilog_args vvp_out stamp sim_log_dir actual_dir

    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        iverilog_error "PROJECT_ROOT no está definido."
        return 1
    fi

    if ! command -v iverilog >/dev/null 2>&1; then
        iverilog_error "iverilog no está en el PATH."
        return 1
    fi

    if [[ -z "$n_cycles" ]]; then
        iverilog_error "falta N_CYCLES."
        iverilog_error "uso: run_iverilog [CASE_DIR] [N_CYCLES] [plusargs...]"
        return 1
    fi

    tb_flist="$(_iverilog_resolve_tb_flist)" || return 1

    if [[ ! -f "$tb_flist" ]]; then
        iverilog_error "flist inexistente: $tb_flist"
        iverilog_error "corré update_flist desde el módulo primero."
        return 1
    fi

    mapfile -t iverilog_args < <(_iverilog_flist_to_args "$tb_flist")

    sim_log_dir="$PROJECT_ROOT/build/iverilog_logs"
    mkdir -p "$sim_log_dir"
    mkdir -p "$case_dir"

    # El testbench escribe en <case_dir>/simulation/vectors/actual/out_ports.
    # Lo creamos acá (Icarus no tiene $system) para que el tb quede portable.
    actual_dir="$case_dir/simulation/vectors/actual/out_ports"
    mkdir -p "$actual_dir"

    stamp="$(date +%Y%m%d_%H%M%S)_$$"
    vvp_out="$case_dir/sim.vvp"

    iverilog_log "module flist : $tb_flist"
    iverilog_log "case_dir     : $case_dir"
    iverilog_log "n_cycles     : $n_cycles"
    iverilog_log "compilando   : $vvp_out"

    iverilog -g2012 -o "$vvp_out" "${iverilog_args[@]}" \
        2>&1 | tee "$sim_log_dir/compile_${stamp}.log"

    if [[ ! -f "$vvp_out" ]]; then
        iverilog_error "la compilación no generó $vvp_out"
        return 1
    fi

    iverilog_log "ejecutando vvp ..."
    vvp "$vvp_out" "+CASE_DIR=${case_dir}" "+N_CYCLES=${n_cycles}" "$@" \
        2>&1 | tee "$sim_log_dir/run_${stamp}.log"
}
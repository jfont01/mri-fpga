#!/usr/bin/env bash

# Este archivo debe ser sourced desde set_env.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[flist_utils.sh] ERROR: this file must be sourced, not executed." >&2
    exit 1
fi

flist_log() {
    echo "[flist_utils.sh] $*"
}

flist_error() {
    echo "[flist_utils.sh] ERROR: $*" >&2
}

flist_require_env() {
    local name="$1"

    if [[ -z "${!name:-}" ]]; then
        flist_error "environment variable not defined: $name"
        return 1
    fi
}

get_module_var() {
    local var_name="$1"

    if [[ -z "${!var_name:-}" ]]; then
        flist_error "variable not defined: $var_name"
        return 1
    fi

    printf "%s\n" "${!var_name}"
}

extract_verilog_includes() {
    local top_file="$1"

    {
        grep -E '^[[:space:]]*`include[[:space:]]+"[^"]+"' "$top_file" 2>/dev/null || true
    } \
        | sed -E 's/^[[:space:]]*`include[[:space:]]+"([^"]+)".*/\1/' \
        | awk '!seen[$0]++'
}

array_add_unique() {
    local array_name="$1"
    local item="$2"

    declare -n arr="$array_name"

    local existing
    for existing in "${arr[@]}"; do
        if [[ "$existing" == "$item" ]]; then
            return 0
        fi
    done

    arr+=("$item")
}

detect_current_module_for_flist() {
    flist_require_env MODULES_ROOT || return 1

    local pwd_real
    local modules_real
    local rel_path
    local module_name

    pwd_real="$(pwd -P)"
    modules_real="$(cd "$MODULES_ROOT" && pwd -P)" || return 1

    case "$pwd_real" in
        "$modules_real"/*)
            rel_path="${pwd_real#$modules_real/}"
            module_name="${rel_path%%/*}"
            ;;
        *)
            flist_error "current directory is not inside MODULES_ROOT."
            echo "[flist_utils.sh]        pwd=$pwd_real" >&2
            echo "[flist_utils.sh]        MODULES_ROOT=$modules_real" >&2
            return 1
            ;;
    esac

    if [[ -z "$module_name" ]]; then
        flist_error "could not detect current module."
        return 1
    fi

    if [[ ! -d "$MODULES_ROOT/$module_name" ]]; then
        flist_error "detected module directory does not exist: $MODULES_ROOT/$module_name"
        return 1
    fi

    printf "%s\n" "$module_name"
}

resolve_include_incdir() {
    local include_file="$1"
    local top_dir="$2"

    local include_path
    local include_dir

    if [[ "$include_file" = /* ]]; then
        include_path="$include_file"
    else
        include_path="$top_dir/$include_file"
    fi

    if [[ -f "$include_path" ]]; then
        include_dir="$(cd "$(dirname "$include_path")" && pwd -P)"
        printf "%s\n" "$include_dir"
        return 0
    fi

    local include_base
    local dep_module
    local dep_rtl_dir
    local candidate

    include_base="$(basename "$include_file")"
    dep_module="${include_base%.*}"

    dep_rtl_dir="$MODULES_ROOT/$dep_module/rtl"
    candidate="$dep_rtl_dir/$include_base"

    if [[ -f "$candidate" ]]; then
        dep_rtl_dir="$(cd "$dep_rtl_dir" && pwd -P)"
        printf "%s\n" "$dep_rtl_dir"
        return 0
    fi

    flist_error "include not found: $include_file"
    echo "[flist_utils.sh]        searched in:" >&2
    echo "[flist_utils.sh]          $top_dir/$include_file" >&2
    echo "[flist_utils.sh]          $candidate" >&2
    return 1
}

# Resuelve un `include a un PATH de archivo (no al directorio).
# Misma regla que resolve_include_incdir: primero junto al archivo que lo
# incluye, despues en $MODULES_ROOT/<modulo>/rtl/<archivo>.
resolve_include_path() {
    local include_file="$1"
    local top_dir="$2"

    local include_path
    if [[ "$include_file" = /* ]]; then
        include_path="$include_file"
    else
        include_path="$top_dir/$include_file"
    fi

    if [[ -f "$include_path" ]]; then
        printf "%s\n" "$include_path"
        return 0
    fi

    local include_base dep_module candidate
    include_base="$(basename "$include_file")"
    dep_module="${include_base%.*}"
    candidate="$MODULES_ROOT/$dep_module/rtl/$include_base"

    if [[ -f "$candidate" ]]; then
        printf "%s\n" "$candidate"
        return 0
    fi

    return 1
}

# RECURSIVO: sigue la cadena de includes (fft1d_r2.v -> cmul.v -> cast.v).
# Sin recursion, un modulo que depende de otro de forma indirecta queda sin
# su -i y el preprocesador falla.
collect_incdirs_for_file() {
    local file="$1"
    local incdirs_array_name="$2"
    local _seen_name="${3:-}"

    if [[ ! -f "$file" ]]; then
        flist_error "source file not found: $file"
        return 1
    fi

    # lista de archivos ya visitados (evita ciclos)
    local -a _local_seen
    if [[ -z "$_seen_name" ]]; then
        _local_seen=()
        _seen_name="_local_seen"
    fi
    declare -n _seen="$_seen_name"

    local file_real
    file_real="$(cd "$(dirname "$file")" && pwd -P)/$(basename "$file")"

    local s
    for s in "${_seen[@]}"; do
        [[ "$s" == "$file_real" ]] && return 0
    done
    _seen+=("$file_real")

    local file_dir
    file_dir="$(dirname "$file_real")"

    array_add_unique "$incdirs_array_name" "$file_dir"

    local include_file include_dir include_path

    while IFS= read -r include_file; do
        [[ -z "$include_file" ]] && continue

        include_dir="$(resolve_include_incdir "$include_file" "$file_dir")" || return 1
        array_add_unique "$incdirs_array_name" "$include_dir"

        # bajar un nivel: el archivo incluido puede incluir otros
        if include_path="$(resolve_include_path "$include_file" "$file_dir")"; then
            collect_incdirs_for_file "$include_path" "$incdirs_array_name" "$_seen_name" || return 1
        fi
    done < <(extract_verilog_includes "$file")
}

get_rtl_top_var_name() {
    local prefix="$1"

    local v_var="${prefix}_V"
    local sv_var="${prefix}_SV"

    if [[ -n "${!v_var:-}" ]]; then
        printf "%s\n" "$v_var"
        return 0
    fi

    if [[ -n "${!sv_var:-}" ]]; then
        printf "%s\n" "$sv_var"
        return 0
    fi

    flist_error "neither $v_var nor $sv_var is defined."
    return 1
}

write_flist() {
    local flist_file="$1"
    shift

    local incdirs=()
    local files=()
    local mode="incdirs"

    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--files" ]]; then
            mode="files"
            continue
        fi

        if [[ "$mode" == "incdirs" ]]; then
            incdirs+=("$arg")
        else
            files+=("$arg")
        fi
    done

    mkdir -p "$(dirname "$flist_file")"

    {
        echo "# Autogenerated by update_flist"
        echo "# Do not edit manually unless you know what you are doing."
        echo

        local incdir
        for incdir in "${incdirs[@]}"; do
            echo "-i $incdir"
        done

        echo

        local file
        for file in "${files[@]}"; do
            echo "$file"
        done
    } > "$flist_file"
}

update_flist() {
    if [[ "$#" -ne 0 ]]; then
        flist_error "usage: update_flist"
        echo "[flist_utils.sh]        run it from inside modules/<module_name>/..." >&2
        return 1
    fi

    flist_require_env MODULES_ROOT || return 1

    local module_name
    module_name="$(detect_current_module_for_flist)" || return 1

    if [[ ! "$module_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        flist_error "invalid detected module name: $module_name"
        return 1
    fi

    local prefix
    if declare -F module_prefix >/dev/null 2>&1; then
        prefix="$(module_prefix "$module_name")"
    else
        prefix="$(echo "$module_name" | tr '[:lower:]' '[:upper:]')"
    fi

    local module_dir="$MODULES_ROOT/$module_name"
    local vars_file="$module_dir/${module_name}_vars.sh"

    if [[ ! -f "$vars_file" ]]; then
        flist_error "missing vars file: $vars_file"
        return 1
    fi

    source "$vars_file"

    local top_var
    local top_file
    local tb_sv
    local synth_wrapper_v
    local impl_wrapper_v
    local tb_flist
    local synth_flist
    local impl_flist

    top_var="$(get_rtl_top_var_name "$prefix")" || return 1
    top_file="$(get_module_var "$top_var")" || return 1

    tb_sv="$(get_module_var "${prefix}_TB_SV")" || return 1
    synth_wrapper_v="$(get_module_var "${prefix}_SYNTH_WRAPPER_V")" || return 1
    impl_wrapper_v="$(get_module_var "${prefix}_IMPL_WRAPPER_V")" || return 1

    tb_flist="$(get_module_var "${prefix}_TB_FLIST")" || return 1
    synth_flist="$(get_module_var "${prefix}_SYNTH_FLIST")" || return 1
    impl_flist="$(get_module_var "${prefix}_IMPL_FLIST")" || return 1

    if [[ ! -f "$top_file" ]]; then
        flist_error "RTL top file not found: $top_file"
        return 1
    fi

    if [[ ! -f "$tb_sv" ]]; then
        flist_error "testbench file not found: $tb_sv"
        return 1
    fi

    if [[ ! -f "$synth_wrapper_v" ]]; then
        flist_error "synthesis wrapper not found: $synth_wrapper_v"
        return 1
    fi

    if [[ ! -f "$impl_wrapper_v" ]]; then
        flist_error "implementation wrapper not found: $impl_wrapper_v"
        return 1
    fi

    local tb_incdirs=()
    local synth_incdirs=()
    local impl_incdirs=()

    collect_incdirs_for_file "$top_file" tb_incdirs || return 1
    collect_incdirs_for_file "$tb_sv" tb_incdirs || return 1

    collect_incdirs_for_file "$top_file" synth_incdirs || return 1
    collect_incdirs_for_file "$synth_wrapper_v" synth_incdirs || return 1

    collect_incdirs_for_file "$top_file" impl_incdirs || return 1
    collect_incdirs_for_file "$impl_wrapper_v" impl_incdirs || return 1

    write_flist "$tb_flist" \
        "${tb_incdirs[@]}" \
        --files \
        "$top_file" \
        "$tb_sv"

    write_flist "$synth_flist" \
        "${synth_incdirs[@]}" \
        --files \
        "$top_file" \
        "$synth_wrapper_v"

    write_flist "$impl_flist" \
        "${impl_incdirs[@]}" \
        --files \
        "$top_file" \
        "$impl_wrapper_v"

    flist_log "module      : $module_name"
    flist_log "rtl         : $top_file"
    flist_log "tb          : $tb_sv"
    flist_log "synth wrap  : $synth_wrapper_v"
    flist_log "impl wrap   : $impl_wrapper_v"
    flist_log "updated     : $tb_flist"
    flist_log "updated     : $synth_flist"
    flist_log "updated     : $impl_flist"
}
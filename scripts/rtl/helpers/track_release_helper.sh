#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[track_release_helper.sh] ERROR: this script must be sourced, not executed." >&2
    exit 1
fi

validate_release_environment() {
    for v in \
        N AF L AXIS PHANTOM \
        TRACK_NB_S TRACK_NBF_S \
        TRACK_NB_Y TRACK_NBF_Y \
        TRACK_NB_A TRACK_NBF_A \
        TRACK_NB_B TRACK_NBF_B \
        CLOCK_AIJ_MHZ CLOCK_BI_MHZ \
        PY_RUNNER TRACK_ROOT RTL_ROOT PY_FXP_MODEL_ROOT \
        TRACK_MANIFEST_HELPER_SH \
        TRACK_DIV_METHOD TRACK_DIV_QUANT_MODE TRACK_DIV_N_CASES TRACK_DIV_SEED \
        TRACK_NB_DIV_NUM TRACK_NBF_DIV_NUM \
        TRACK_NB_DIV_DEN TRACK_NBF_DIV_DEN \
        TRACK_NB_DIV_QUOTIENT TRACK_NBF_DIV_QUOTIENT
    do
        require_var "$v"
    done

    require_file "$TRACK_MANIFEST_HELPER_SH"

    case "$TRACK_DIV_METHOD" in
        restoring|div_restoring)
            require_var CLOCK_DIV_RESTORING_MHZ
            ;;

        newton_raphson|nr|reciprocal_nr)
            die "TRACK_DIV_METHOD='$TRACK_DIV_METHOD' todavía no tiene RTL/testbench de track implementado. Usar TRACK_DIV_METHOD='restoring' por ahora."
            ;;

        *)
            die "TRACK_DIV_METHOD inválido: '$TRACK_DIV_METHOD'. Opciones actuales: restoring. Reservadas: newton_raphson."
            ;;
    esac

    case "$TRACK_DIV_QUANT_MODE" in
        trunc|round)
            ;;

        *)
            die "TRACK_DIV_QUANT_MODE inválido: '$TRACK_DIV_QUANT_MODE'. Usar trunc o round."
            ;;
    esac
}

map_track_formats() {
    NB_S="${TRACK_NB_S}"
    NBF_S="${TRACK_NBF_S}"

    NB_Y="${TRACK_NB_Y}"
    NBF_Y="${TRACK_NBF_Y}"

    NB_A="${TRACK_NB_A}"
    NBF_A="${TRACK_NBF_A}"

    NB_B="${TRACK_NB_B}"
    NBF_B="${TRACK_NBF_B}"

    # Alias temporal por compatibilidad con rutas existentes del flujo Python.
    # En el flujo Python, el formato de Y proviene de la cuantización de K/IFFT2D.
    NB_K="$NB_Y"
    NBF_K="$NBF_Y"

    # -------------------------------------------------------------------------
    # Configuración de división del track
    # -------------------------------------------------------------------------
    DIV_METHOD="${TRACK_DIV_METHOD}"
    DIV_QUANT_MODE="${TRACK_DIV_QUANT_MODE}"
    DIV_N_CASES="${TRACK_DIV_N_CASES}"
    DIV_SEED="${TRACK_DIV_SEED}"

    NB_DIV_NUM="${TRACK_NB_DIV_NUM}"
    NBF_DIV_NUM="${TRACK_NBF_DIV_NUM}"

    NB_DIV_DEN="${TRACK_NB_DIV_DEN}"
    NBF_DIV_DEN="${TRACK_NBF_DIV_DEN}"

    NB_DIV_QUOTIENT="${TRACK_NB_DIV_QUOTIENT}"
    NBF_DIV_QUOTIENT="${TRACK_NBF_DIV_QUOTIENT}"

    case "$DIV_METHOD" in
        restoring|div_restoring)
            DIV_METHOD="restoring"
            DIV_RTL_CASE="div_restoring"
            ;;

        newton_raphson|nr|reciprocal_nr)
            DIV_METHOD="newton_raphson"
            DIV_RTL_CASE="div_newton_raphson"
            ;;

        *)
            die "DIV_METHOD inválido: '$DIV_METHOD'"
            ;;
    esac
}

build_track_case_names() {
    # Nombre completo usado para buscar resultados Python existentes.
    STIM_DIR_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
    CASE_DIR_NAME="NB_Y${NB_Y}_NBF_Y${NBF_Y}-NB_S${NB_S}_NBF_S${NBF_S}-NB_A${NB_A}_NBF_A${NBF_A}-NB_B${NB_B}_NBF_B${NBF_B}"

    # Nombre reducido usado solamente para crear el track.
    TRACK_STIM_DIR_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}"
    TRACK_CASE_DIR_NAME="NB_Y${NB_Y}-NB_S${NB_S}-NB_A${NB_A}-NB_B${NB_B}"

    TRACK_BASE="$TRACK_ROOT/track.${TRACK_STIM_DIR_NAME}.${TRACK_CASE_DIR_NAME}"

    next_rev="$(get_next_revision "$TRACK_BASE")"
    TRACK_DIR="${TRACK_BASE}.rev${next_rev}"

    release_log "Source stimulus case : $STIM_DIR_NAME"
    release_log "Source format case   : $CASE_DIR_NAME"
    release_log "Track base           : $TRACK_BASE"
    release_log "Next revision        : rev${next_rev}"
    release_log "Track dir            : $TRACK_DIR"
    release_log "Division method      : $DIV_METHOD"
    release_log "Division RTL case    : $DIV_RTL_CASE"
    release_log "Division quant mode  : $DIV_QUANT_MODE"
}

set_source_report_paths() {
    FXP_IFFT2D_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/fft2d_fxp/NB${NB_K}_NBF${NBF_K}/coils_aliased.rpt"
    FXP_QUANTIZER_S_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/quantizer/S/NB${NB_S}_NBF${NBF_S}/S.rpt"
    FXP_QUANTIZER_K_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/quantizer/k/NB${NB_K}_NBF${NBF_K}/k.rpt"

    GLOBAL_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/reporter/$CASE_DIR_NAME/global_compare_report.rpt"
    GLOBAL_FXP_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/global_fxp_report.rpt"
    GLOBAL_FP_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fp/global_fp_report.rpt"
}

set_source_vector_paths() {
    PY_A_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/A/py_A.dat"
    PY_B_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/b/py_b.dat"
    PY_D_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/D/py_D.dat"
    PY_I_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/I/py_I.dat"
    PY_L_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/L/py_L.dat"
    PY_M_HAT_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/m_hat/py_m_hat.dat"
    PY_X_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/x/py_x.dat"
    PY_Z_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/z/py_z.dat"

    PY_S_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/quantizer/S/NB${NB_S}_NBF${NBF_S}/py_S.dat"
    PY_Y_DAT="$PY_RUNNER/output/$STIM_DIR_NAME/fft2d_fxp/NB${NB_K}_NBF${NBF_K}/py_y.dat"
}

preflight_release_sources() {
    release_log "Running preflight checks..."

    for f in \
        "$FXP_IFFT2D_RPT" \
        "$FXP_QUANTIZER_S_RPT" \
        "$FXP_QUANTIZER_K_RPT" \
        "$GLOBAL_RPT" \
        "$GLOBAL_FXP_RPT" \
        "$GLOBAL_FP_RPT" \
        "$PY_A_DAT" \
        "$PY_B_DAT" \
        "$PY_D_DAT" \
        "$PY_I_DAT" \
        "$PY_L_DAT" \
        "$PY_M_HAT_DAT" \
        "$PY_X_DAT" \
        "$PY_Z_DAT" \
        "$PY_S_DAT" \
        "$PY_Y_DAT"
    do
        require_file "$f"
    done

    release_log "Preflight OK."
}

create_track_directories() {
    mkdir -p "$TRACK_DIR"
    mark_track_created

    TRACK_VECTORS_RTL="$TRACK_DIR/vectors/rtl"
    TRACK_VECTORS_PY="$TRACK_DIR/vectors/py"
    TRACK_STIMULI="$TRACK_DIR/stimuli"

    SV_PKG_DIR="$TRACK_DIR/package"
    FLIST_DIR="$TRACK_DIR/flist"
    SYNTHESIS_DIR="$TRACK_DIR/synthesis"
    SIMULATION_DIR="$TRACK_DIR/simulation"
    CONSTRAINTS_DIR="$TRACK_DIR/constraints"

    mkdir -p \
        "$TRACK_VECTORS_RTL" \
        "$TRACK_VECTORS_PY" \
        "$TRACK_STIMULI" \
        "$SV_PKG_DIR" \
        "$FLIST_DIR" \
        "$SYNTHESIS_DIR" \
        "$SIMULATION_DIR" \
        "$CONSTRAINTS_DIR"
}

copy_release_reports() {
    copy_file_as "$FXP_IFFT2D_RPT"      "$TRACK_DIR/fxp_ifft2d.rpt"
    copy_file_as "$FXP_QUANTIZER_S_RPT" "$TRACK_DIR/fxp_quantize_S.rpt"
    copy_file_as "$FXP_QUANTIZER_K_RPT" "$TRACK_DIR/fxp_quantize_k.rpt"
    copy_file_as "$GLOBAL_RPT"          "$TRACK_DIR/snr_global.rpt"
    copy_file_as "$GLOBAL_FXP_RPT"      "$TRACK_DIR/fxp_global.rpt"
    copy_file_as "$GLOBAL_FP_RPT"       "$TRACK_DIR/fp_global.rpt"
}

copy_release_vectors() {
    copy_file_to_dir "$PY_A_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_B_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_D_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_I_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_L_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_M_HAT_DAT" "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_X_DAT"     "$TRACK_VECTORS_PY"
    copy_file_to_dir "$PY_Z_DAT"     "$TRACK_VECTORS_PY"

    copy_file_to_dir "$PY_S_DAT" "$TRACK_STIMULI"
    copy_file_to_dir "$PY_Y_DAT" "$TRACK_STIMULI"
}

generate_division_vectors() {
    case "$DIV_METHOD" in
        restoring)
            DIV_RTL_CASE="div_restoring"
            ;;

        newton_raphson)
            die "generate_division_vectors: TRACK_DIV_METHOD='newton_raphson' todavía no está implementado para RTL."
            ;;

        *)
            die "generate_division_vectors: método inválido '$DIV_METHOD'"
            ;;
    esac

    # Se conservan estos nombres porque run_xsim/run_vm actuales detectan:
    #   stimuli/div_restoring_*_in.dat
    #   vectors/py/py_div_restoring_*.dat
    DIVISION_IN_DAT="$TRACK_STIMULI/${DIV_RTL_CASE}_${DIV_QUANT_MODE}_in.dat"
    PY_DIVISION_DAT="$TRACK_VECTORS_PY/py_${DIV_RTL_CASE}_${DIV_QUANT_MODE}.dat"

    release_log "Generating division stimulus/reference..."
    release_log "  method: $DIV_METHOD"
    release_log "  case  : $DIV_RTL_CASE"
    release_log "  mode  : $DIV_QUANT_MODE"
    release_log "  input : $DIVISION_IN_DAT"
    release_log "  ref   : $PY_DIVISION_DAT"

    export PY_FXP_MODEL_ROOT

    export DIVISION_IN_DAT
    export PY_DIVISION_DAT

    export DIV_METHOD
    export DIV_QUANT_MODE
    export DIV_N_CASES
    export DIV_SEED

    export NB_DIV_NUM
    export NBF_DIV_NUM
    export NB_DIV_DEN
    export NBF_DIV_DEN
    export NB_DIV_QUOTIENT
    export NBF_DIV_QUOTIENT

    python3 <<'PY'
import os
import sys
import random
from pathlib import Path

PY_FXP_MODEL_ROOT = os.environ["PY_FXP_MODEL_ROOT"]
sys.path.insert(0, PY_FXP_MODEL_ROOT)

from fxp import Fxp
from fxp_division import divide


def get_range_from(NB: int, NBF: int, signed: bool) -> tuple[float, float]:
    if signed:
        max_val = (2 ** (NB - NBF - 1)) - (2 ** (-NBF))
        min_val = -(2 ** (NB - NBF - 1))
    else:
        max_val = (2 ** (NB - NBF)) - (2 ** (-NBF))
        min_val = 0.0

    return min_val, max_val


def gen_random_value(rng: random.Random, NB: int, NBF: int, signed: bool) -> float:
    min_val, max_val = get_range_from(NB, NBF, signed)
    return rng.uniform(min_val, max_val)


in_path = Path(os.environ["DIVISION_IN_DAT"])
out_path = Path(os.environ["PY_DIVISION_DAT"])

method = os.environ["DIV_METHOD"]
quant_mode = os.environ["DIV_QUANT_MODE"]

n_cases = int(os.environ["DIV_N_CASES"])
seed = int(os.environ["DIV_SEED"])

NB_NUM = int(os.environ["NB_DIV_NUM"])
NBF_NUM = int(os.environ["NBF_DIV_NUM"])

NB_DEN = int(os.environ["NB_DIV_DEN"])
NBF_DEN = int(os.environ["NBF_DIV_DEN"])

NB_QUOTIENT = int(os.environ["NB_DIV_QUOTIENT"])
NBF_QUOTIENT = int(os.environ["NBF_DIV_QUOTIENT"])

signed = True
rng = random.Random(seed)

in_path.parent.mkdir(parents=True, exist_ok=True)
out_path.parent.mkdir(parents=True, exist_ok=True)

with in_path.open("w", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    written = 0

    while written < n_cases:
        num_f = gen_random_value(rng, NB_NUM, NBF_NUM, signed)
        den_f = gen_random_value(rng, NB_DEN, NBF_DEN, signed)

        # Evitar denominadores demasiado cercanos a cero.
        if abs(den_f) < 2.0 ** (-max(NBF_DEN - 2, 0)):
            den_f = 0.5 if den_f >= 0 else -0.5

        num = Fxp.quantize(
            num_f,
            NB=NB_NUM,
            NBF=NBF_NUM,
            mode=quant_mode,
            signed=signed,
        )

        den = Fxp.quantize(
            den_f,
            NB=NB_DEN,
            NBF=NBF_DEN,
            mode=quant_mode,
            signed=signed,
        )

        if den.get_val() == 0.0:
            continue

        min_neg_num = -(1 << (NB_NUM - 1))
        min_neg_den = -(1 << (NB_DEN - 1))

        if signed and num.to_sint() == min_neg_num:
            continue

        if signed and den.to_sint() == min_neg_den:
            continue

        q = divide(
            num=num,
            den=den,
            NB_out=NB_QUOTIENT,
            NBF_out=NBF_QUOTIENT,
            mode=quant_mode,
            overflow="saturate",
            signed_out=signed,
            method=method,
        )

        fin.write(f"{num.to_hex().lower()} {den.to_hex().lower()}\n")
        fout.write(f"{q.to_hex().lower()}\n")

        written += 1

print(f"[create_release.sh] Generated division cases: {written}")
print(f"[create_release.sh] Method: {method}")
print(f"[create_release.sh] Mode  : {quant_mode}")
print(f"[create_release.sh] Input : {in_path}")
print(f"[create_release.sh] Ref   : {out_path}")
PY
}

# Alias temporal por compatibilidad si create_release.sh todavía llama a la función vieja.
generate_div_restoring_vectors() {
    generate_division_vectors
}

generate_track_params_pkg() {
    SV_PKG_FILE="$SV_PKG_DIR/track_params_pkg.sv"

    cat > "$SV_PKG_FILE" <<EOF
package track_params_pkg;

    parameter int NB_S  = ${NB_S};
    parameter int NBF_S = ${NBF_S};

    parameter int NB_Y  = ${NB_Y};
    parameter int NBF_Y = ${NBF_Y};

    parameter int NB_A  = ${NB_A};
    parameter int NBF_A = ${NBF_A};

    parameter int NB_B  = ${NB_B};
    parameter int NBF_B = ${NBF_B};

    parameter int L     = ${L};
    parameter int AF    = ${AF};
    parameter int N     = ${N};

    parameter int NB_DIV_NUM       = ${NB_DIV_NUM};
    parameter int NBF_DIV_NUM      = ${NBF_DIV_NUM};

    parameter int NB_DIV_DEN       = ${NB_DIV_DEN};
    parameter int NBF_DIV_DEN      = ${NBF_DIV_DEN};

    parameter int NB_DIV_QUOTIENT  = ${NB_DIV_QUOTIENT};
    parameter int NBF_DIV_QUOTIENT = ${NBF_DIV_QUOTIENT};

endpackage
EOF

    release_log "Generated SV package: $SV_PKG_FILE"
}

generate_track_flists() {
    cat > "$FLIST_DIR/tb_compute_Aij.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/cmul.sv
$RTL_ROOT/src/sense/compute_Aij.sv
$RTL_ROOT/testbench/sense/tb_compute_Aij.sv
EOF

    cat > "$FLIST_DIR/tb_compute_bi.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/cmul.sv
$RTL_ROOT/src/sense/compute_bi.sv
$RTL_ROOT/testbench/sense/tb_compute_bi.sv
EOF

    cat > "$FLIST_DIR/synth_compute_Aij.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/cmul.sv
$RTL_ROOT/src/sense/compute_Aij.sv
$RTL_ROOT/wrappers/wrapper_compute_Aij.sv
EOF

    cat > "$FLIST_DIR/synth_compute_bi.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/cmul.sv
$RTL_ROOT/src/sense/compute_bi.sv
$RTL_ROOT/wrappers/wrapper_compute_bi.sv
EOF

    case "$DIV_METHOD" in
        restoring)
            cat > "$FLIST_DIR/tb_div_restoring.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/div_restoring.sv
$RTL_ROOT/testbench/ops/tb_div_restoring.sv
EOF

            cat > "$FLIST_DIR/synth_div_restoring.flist" <<EOF
$TRACK_DIR/package/track_params_pkg.sv
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/div_restoring.sv
$RTL_ROOT/wrappers/wrapper_div_restoring.sv
EOF
            ;;

        newton_raphson)
            die "generate_track_flists: RTL newton_raphson todavía no implementado."
            ;;

        *)
            die "generate_track_flists: método de división inválido '$DIV_METHOD'"
            ;;
    esac

    release_log "Generated flists in: $FLIST_DIR"
}

generate_track_constraints() {
    CLOCK_AIJ_NS="$(awk "BEGIN {printf \"%.3f\", 1000.0 / $CLOCK_AIJ_MHZ}")"
    CLOCK_BI_NS="$(awk "BEGIN {printf \"%.3f\", 1000.0 / $CLOCK_BI_MHZ}")"

    Aij_XDC="$CONSTRAINTS_DIR/clock_Aij.xdc"
    BI_XDC="$CONSTRAINTS_DIR/clock_bi.xdc"

    cat > "$Aij_XDC" <<EOF
create_clock -name i_clock -period ${CLOCK_AIJ_NS} [get_ports i_clock]
EOF

    cat > "$BI_XDC" <<EOF
create_clock -name i_clock -period ${CLOCK_BI_NS} [get_ports i_clock]
EOF

    release_log "Generated XDC: $Aij_XDC"
    release_log "Generated XDC: $BI_XDC"

    case "$DIV_METHOD" in
        restoring)
            CLOCK_DIV_RESTORING_NS="$(awk "BEGIN {printf \"%.3f\", 1000.0 / $CLOCK_DIV_RESTORING_MHZ}")"
            DIV_RESTORING_XDC="$CONSTRAINTS_DIR/clock_div_restoring.xdc"

            cat > "$DIV_RESTORING_XDC" <<EOF
create_clock -name i_clock -period ${CLOCK_DIV_RESTORING_NS} [get_ports i_clock]
EOF

            release_log "Generated XDC: $DIV_RESTORING_XDC"
            ;;

        newton_raphson)
            die "generate_track_constraints: RTL newton_raphson todavía no implementado."
            ;;

        *)
            die "generate_track_constraints: método de división inválido '$DIV_METHOD'"
            ;;
    esac
}

generate_release_manifest() {
    TRACK_MANIFEST_JSON="$TRACK_DIR/track_manifest.json"
    TRACK_REV="$next_rev"
    TRACK_CREATED_AT="$(date -Iseconds)"

    export TRACK_MANIFEST_JSON
    export TRACK_REV
    export TRACK_CREATED_AT

    export STIM_DIR_NAME
    export CASE_DIR_NAME
    export TRACK_STIM_DIR_NAME
    export TRACK_CASE_DIR_NAME
    export TRACK_BASE
    export TRACK_DIR

    export NB_S
    export NBF_S
    export NB_Y
    export NBF_Y
    export NB_A
    export NBF_A
    export NB_B
    export NBF_B
    export NB_K
    export NBF_K

    export DIV_METHOD
    export DIV_RTL_CASE
    export DIV_QUANT_MODE
    export DIV_N_CASES
    export DIV_SEED

    export NB_DIV_NUM
    export NBF_DIV_NUM
    export NB_DIV_DEN
    export NBF_DIV_DEN
    export NB_DIV_QUOTIENT
    export NBF_DIV_QUOTIENT

    export FXP_IFFT2D_RPT
    export FXP_QUANTIZER_S_RPT
    export FXP_QUANTIZER_K_RPT
    export GLOBAL_RPT
    export GLOBAL_FXP_RPT
    export GLOBAL_FP_RPT

    export PY_A_DAT
    export PY_B_DAT
    export PY_D_DAT
    export PY_I_DAT
    export PY_L_DAT
    export PY_M_HAT_DAT
    export PY_X_DAT
    export PY_Z_DAT
    export PY_S_DAT
    export PY_Y_DAT

    export DIVISION_IN_DAT
    export PY_DIVISION_DAT

    generate_track_manifest_json "$TRACK_MANIFEST_JSON"
}
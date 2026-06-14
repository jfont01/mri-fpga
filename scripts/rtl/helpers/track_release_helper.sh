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
        PY_RUNNER TRACK_ROOT RTL_ROOT TRACK_MANIFEST_HELPER_SH
    do
        require_var "$v"
    done

    require_file "$TRACK_MANIFEST_HELPER_SH"
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
}

build_track_case_names() {
    # Nombre completo usado para buscar resultados Python existentes
    STIM_DIR_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
    CASE_DIR_NAME="NB_Y${NB_Y}_NBF_Y${NBF_Y}-NB_S${NB_S}_NBF_S${NBF_S}-NB_A${NB_A}_NBF_A${NBF_A}-NB_B${NB_B}_NBF_B${NBF_B}"

    # Nombre reducido usado solamente para crear el track
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
}

generate_release_manifest() {
    TRACK_MANIFEST_JSON="$TRACK_DIR/track_manifest.json"
    TRACK_REV="$next_rev"
    TRACK_CREATED_AT="$(date -Iseconds)"

    generate_track_manifest_json "$TRACK_MANIFEST_JSON"
}
source "$TRACK_CONF"

STIM_DIR_NAME="N${N}_Af${AF}_L${L}_axis${AXIS}_${PHANTOM}"
CASE_DIR_NAME="NB_Y${NB_K}_NBF_Y${NBF_K}-NB_S${NB_S}_NBF_S${NBF_S}-NB_A${NB_A}_NBF_A${NBF_A}-NB_B${NB_B}_NBF_B${NBF_B}"
TRACK_BASE="track.${STIM_DIR_NAME}.${CASE_DIR_NAME}"

last_rev=0

for d in "${TRACK_BASE}".rev*; do
    [ -e "$d" ] || continue

    rev_part="${d##*.rev}"

    if [[ "$rev_part" =~ ^[0-9]+$ ]]; then
        if (( rev_part > last_rev )); then
            last_rev=$rev_part
        fi
    fi
done

next_rev=$((last_rev + 1))
TRACK_DIR="${TRACK_BASE}.rev${next_rev}"

mkdir -p "$TRACK_DIR"

FXP_IFFT2D_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/fft2d_fxp/NB${NB_K}_NBF${NBF_K}/coils_aliased.rpt"
FXP_QUANTIZER_S_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/quantizer/S/NB${NB_S}_NBF${NBF_S}/S.rpt"
FXP_QUANTIZER_K_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/quantizer/k/NB${NB_K}_NBF${NBF_K}/k.rpt"
GLOBAL_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/reporter/$CASE_DIR_NAME/global_compare_report.rpt"
GLOBAL_FXP_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fxp/$CASE_DIR_NAME/global_fxp_report.rpt"
GLOBAL_FP_RPT="$PY_RUNNER/output/$STIM_DIR_NAME/sense_fp/global_fp_report.rpt"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Missing file: $path" >&2
        exit 1
    fi
}

require_file "$FXP_IFFT2D_RPT"
require_file "$FXP_QUANTIZER_S_RPT"
require_file "$FXP_QUANTIZER_K_RPT"
require_file "$GLOBAL_RPT"
require_file "$GLOBAL_FXP_RPT"
require_file "$GLOBAL_FP_RPT"

copy() {
  local src="$1"
  local dst="$2"

  if [[ -f "$src" ]]; then
    printf "[run.sh]    Copying file: %s to %s\n" "$src" "$dst"
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
  else
    printf "[run.sh]    File not found, skipping copy: %s\n" "$src"
  fi
}


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

require_file "$PY_A_DAT"
require_file "$PY_B_DAT"
require_file "$PY_D_DAT"
require_file "$PY_I_DAT"
require_file "$PY_L_DAT"
require_file "$PY_M_HAT_DAT"
require_file "$PY_X_DAT"
require_file "$PY_Z_DAT"
require_file "$PY_S_DAT"
require_file "$PY_Y_DAT"

for d in A b D I L S x y z m_hat; do
    mkdir -p "$TRACK_DIR/vm/$d"
done
mkdir -p "$TRACK_DIR/stimuli"

copy "$FXP_IFFT2D_RPT"      "$TRACK_DIR/fxp_ifft2d.rpt"
copy "$FXP_QUANTIZER_S_RPT" "$TRACK_DIR/fxp_quantize_S.rpt"
copy "$FXP_QUANTIZER_K_RPT" "$TRACK_DIR/fxp_quantize_k.rpt"
copy "$GLOBAL_RPT"          "$TRACK_DIR/snr_global.rpt"
copy "$GLOBAL_FXP_RPT"      "$TRACK_DIR/fxp_global.rpt"
copy "$GLOBAL_FP_RPT"       "$TRACK_DIR/fp_global.rpt"

copy "$PY_A_DAT"        "$TRACK_DIR/vm/A"
copy "$PY_B_DAT"        "$TRACK_DIR/vm/b"
copy "$PY_D_DAT"        "$TRACK_DIR/vm/D"
copy "$PY_I_DAT"        "$TRACK_DIR/vm/I"
copy "$PY_L_DAT"        "$TRACK_DIR/vm/L"
copy "$PY_M_HAT_DAT"    "$TRACK_DIR/vm/m_hat"
copy "$PY_X_DAT"        "$TRACK_DIR/vm/x"
copy "$PY_Z_DAT"        "$TRACK_DIR/vm/z"
copy "$PY_S_DAT"        "$TRACK_DIR/stimuli"
copy "$PY_Y_DAT"        "$TRACK_DIR/stimuli"


SV_PKG_DIR="$TRACK_DIR/package"
SV_PKG_FILE="$SV_PKG_DIR/track_params_pkg.svh"

mkdir -p "$SV_PKG_DIR"

cat > "$SV_PKG_FILE" <<EOF
package track_params_pkg;

    parameter int NB_S  = ${NB_S};
    parameter int NBF_S = ${NBF_S};

    parameter int NB_Y  = ${NB_K};
    parameter int NBF_Y = ${NBF_K};

    parameter int NB_A  = ${NB_A};
    parameter int NBF_A = ${NBF_A};

    parameter int NB_B  = ${NB_B};
    parameter int NBF_B = ${NBF_B};

    parameter int L     = ${L};
    parameter int AF    = ${AF};
    parameter int N     = ${N};

endpackage
EOF

echo "[run.sh]    Generated SVH package: $SV_PKG_FILE"

FLIST="$TRACK_DIR/flist/tb_flist.f"
mkdir -p "$(dirname "$FLIST")"

cat > "$FLIST" <<EOF
$TRACK_DIR/include/track_params_pkg.svh
$RTL_ROOT/src/ops/cast.sv
$RTL_ROOT/src/ops/cmul.sv
$RTL_ROOT/src/sense/compute_Aij.sv
$RTL_ROOT/src/sense/compute_bi.sv
$RTL_ROOT/tb/tb_compute_Aij.sv
$RTL_ROOT/tb/tb_compute_bi.sv
EOF

SCRIPTS_DIR="$TRACK_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"

SYNTHESIS_DIR="$TRACK_DIR/synthesis"
mkdir -p "$SYNTHESIS_DIR"

SIMULATION_DIR="$TRACK_DIR/simulation"
mkdir -p "$SIMULATION_DIR"

require_env() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "[ERROR] Environment variable $var_name is not defined" >&2
        exit 1
    fi
}

make_link() {
    local target="$1"
    local link_path="$2"

    mkdir -p "$(dirname "$link_path")"

    if [[ ! -e "$target" ]]; then
        echo "[ERROR] Link target does not exist: $target" >&2
        exit 1
    fi

    ln -sfn "$target" "$link_path"
    echo "[run.sh]    Symlink created: $link_path -> $target"
}

require_env RTL_SCRIPTS_SYNTH_RUNNER
require_env RTL_SCRIPTS_VM_RUNNER
require_env RTL_SCRIPTS_XSIM_RUNNER

make_link "$RTL_SCRIPTS_SYNTH_RUNNER" "$TRACK_DIR/scripts/run_synth.sh"
make_link "$RTL_SCRIPTS_XSIM_RUNNER"  "$TRACK_DIR/scripts/run_xsim.sh"
make_link "$RTL_SCRIPTS_VM_RUNNER"    "$TRACK_DIR/scripts/run_vm.sh"
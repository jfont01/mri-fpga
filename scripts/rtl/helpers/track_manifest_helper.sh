#!/usr/bin/env bash

generate_track_manifest_json() {
    local manifest_path="$1"

    if [[ -z "$manifest_path" ]]; then
        echo "[track_manifest_helper.sh] ERROR: missing manifest output path" >&2
        return 1
    fi

    export TRACK_MANIFEST_JSON="$manifest_path"

    export TRACK_CREATED_AT="${TRACK_CREATED_AT:-$(date -Iseconds)}"
    export TRACK_REV="${TRACK_REV:-${next_rev:-}}"

    export TRACK_DIR
    export TRACK_BASE
    export TRACK_STIM_DIR_NAME
    export TRACK_CASE_DIR_NAME
    export STIM_DIR_NAME
    export CASE_DIR_NAME

    export N
    export AF
    export L
    export AXIS
    export PHANTOM

    export NB_S
    export NBF_S
    export NB_K
    export NBF_K
    export NB_A
    export NBF_A
    export NB_B
    export NBF_B

    export CLOCK_AIJ_MHZ
    export CLOCK_BI_MHZ
    export CLOCK_AIJ_NS
    export CLOCK_BI_NS

    export TRACK_CONF
    export GLOBAL_CONF_FOR_MANIFEST="${GLOBAL_CONF_PATH:-${GLOBAL_CONFIG_CONF:-}}"

    export PY_RUNNER
    export RTL_ROOT

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

    export TRACK_VECTORS_RTL
    export TRACK_VECTORS_PY
    export TRACK_STIMULI

    export SV_PKG_FILE
    export FLIST_DIR
    export SYNTHESIS_DIR
    export SIMULATION_DIR
    export CONSTRAINTS_DIR
    export Aij_XDC
    export BI_XDC

    python3 <<'PY'
import json
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def int_env(name: str):
    value = env(name)
    return int(value) if value != "" else None


def float_env(name: str):
    value = env(name)
    return float(value) if value != "" else None


def abspath(path: str) -> str:
    if not path:
        return ""
    return str(Path(path).resolve())


def dst_file(dst_dir: str, src_file: str) -> str:
    if not dst_dir or not src_file:
        return ""
    return str((Path(dst_dir) / Path(src_file).name).resolve())


track_dir = env("TRACK_DIR")
track_vectors_py = env("TRACK_VECTORS_PY")
track_stimuli = env("TRACK_STIMULI")

manifest = {
    "schema_version": 1,

    "created_at": env("TRACK_CREATED_AT"),

    "track": {
        "dir": abspath(track_dir),
        "base": abspath(env("TRACK_BASE")),
        "revision": int_env("TRACK_REV"),
        "stim_dir_name": env("TRACK_STIM_DIR_NAME"),
        "case_dir_name": env("TRACK_CASE_DIR_NAME"),
    },

    "python_source_case": {
        "stim_dir_name": env("STIM_DIR_NAME"),
        "case_dir_name": env("CASE_DIR_NAME"),
        "runner_output_root": abspath(env("PY_RUNNER")),
    },

    "stimulus": {
        "N": int_env("N"),
        "AF": int_env("AF"),
        "L": int_env("L"),
        "axis": env("AXIS"),
        "phantom": env("PHANTOM"),
    },

    "formats": {
        "S": {
            "NB": int_env("NB_S"),
            "NBF": int_env("NBF_S"),
            "signed": True,
        },
        "Y": {
            "NB": int_env("NB_K"),
            "NBF": int_env("NBF_K"),
            "signed": True,
            "note": "NB_K/NBF_K are used as NB_Y/NBF_Y in the RTL track package.",
        },
        "A": {
            "NB": int_env("NB_A"),
            "NBF": int_env("NBF_A"),
            "signed": True,
        },
        "B": {
            "NB": int_env("NB_B"),
            "NBF": int_env("NBF_B"),
            "signed": True,
        },
    },

    "clocks": {
        "Aij": {
            "frequency_mhz": float_env("CLOCK_AIJ_MHZ"),
            "period_ns": float_env("CLOCK_AIJ_NS"),
            "xdc": abspath(env("Aij_XDC")),
        },
        "bi": {
            "frequency_mhz": float_env("CLOCK_BI_MHZ"),
            "period_ns": float_env("CLOCK_BI_NS"),
            "xdc": abspath(env("BI_XDC")),
        },
    },

    "configs": {
        "track_conf": abspath(env("TRACK_CONF")),
        "global_conf": abspath(env("GLOBAL_CONF_FOR_MANIFEST")),
    },

    "source_reports": {
        "fxp_ifft2d": abspath(env("FXP_IFFT2D_RPT")),
        "quantizer_S": abspath(env("FXP_QUANTIZER_S_RPT")),
        "quantizer_k": abspath(env("FXP_QUANTIZER_K_RPT")),
        "global_compare": abspath(env("GLOBAL_RPT")),
        "global_fxp": abspath(env("GLOBAL_FXP_RPT")),
        "global_fp": abspath(env("GLOBAL_FP_RPT")),
    },

    "track_reports": {
        "fxp_ifft2d": abspath(f"{track_dir}/fxp_ifft2d.rpt"),
        "quantizer_S": abspath(f"{track_dir}/fxp_quantize_S.rpt"),
        "quantizer_k": abspath(f"{track_dir}/fxp_quantize_k.rpt"),
        "snr_global": abspath(f"{track_dir}/snr_global.rpt"),
        "fxp_global": abspath(f"{track_dir}/fxp_global.rpt"),
        "fp_global": abspath(f"{track_dir}/fp_global.rpt"),
    },

    "source_vectors": {
        "py_A": abspath(env("PY_A_DAT")),
        "py_b": abspath(env("PY_B_DAT")),
        "py_D": abspath(env("PY_D_DAT")),
        "py_I": abspath(env("PY_I_DAT")),
        "py_L": abspath(env("PY_L_DAT")),
        "py_m_hat": abspath(env("PY_M_HAT_DAT")),
        "py_x": abspath(env("PY_X_DAT")),
        "py_z": abspath(env("PY_Z_DAT")),
        "py_S": abspath(env("PY_S_DAT")),
        "py_y": abspath(env("PY_Y_DAT")),
    },

    "track_vectors": {
        "py_A": dst_file(track_vectors_py, env("PY_A_DAT")),
        "py_b": dst_file(track_vectors_py, env("PY_B_DAT")),
        "py_D": dst_file(track_vectors_py, env("PY_D_DAT")),
        "py_I": dst_file(track_vectors_py, env("PY_I_DAT")),
        "py_L": dst_file(track_vectors_py, env("PY_L_DAT")),
        "py_m_hat": dst_file(track_vectors_py, env("PY_M_HAT_DAT")),
        "py_x": dst_file(track_vectors_py, env("PY_X_DAT")),
        "py_z": dst_file(track_vectors_py, env("PY_Z_DAT")),
        "py_S": dst_file(track_stimuli, env("PY_S_DAT")),
        "py_y": dst_file(track_stimuli, env("PY_Y_DAT")),
    },

    "generated_rtl_files": {
        "track_params_pkg": abspath(env("SV_PKG_FILE")),
        "tb_compute_Aij_flist": abspath(f"{env('FLIST_DIR')}/tb_compute_Aij.flist"),
        "tb_compute_bi_flist": abspath(f"{env('FLIST_DIR')}/tb_compute_bi.flist"),
        "synth_compute_Aij_flist": abspath(f"{env('FLIST_DIR')}/synth_compute_Aij.flist"),
        "synth_compute_bi_flist": abspath(f"{env('FLIST_DIR')}/synth_compute_bi.flist"),
        "clock_Aij_xdc": abspath(env("Aij_XDC")),
        "clock_bi_xdc": abspath(env("BI_XDC")),
    },

    "track_directories": {
        "vectors_py": abspath(env("TRACK_VECTORS_PY")),
        "vectors_rtl": abspath(env("TRACK_VECTORS_RTL")),
        "stimuli": abspath(env("TRACK_STIMULI")),
        "package": abspath(str(Path(env("SV_PKG_FILE")).parent)),
        "flist": abspath(env("FLIST_DIR")),
        "simulation": abspath(env("SIMULATION_DIR")),
        "synthesis": abspath(env("SYNTHESIS_DIR")),
        "constraints": abspath(env("CONSTRAINTS_DIR")),
    },

    "rtl_root": abspath(env("RTL_ROOT")),
}

out_path = Path(env("TRACK_MANIFEST_JSON"))
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=False)
    f.write("\n")

print(f"[track_manifest_helper.sh] Generated manifest: {out_path}")
PY
}
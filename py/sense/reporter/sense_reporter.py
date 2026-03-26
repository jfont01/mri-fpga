
#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess


# ------------------------- ENVIRONMENT SET -------------------------
SENSE_FP_DIR = os.environ.get("SENSE_FP_DIR")
if SENSE_FP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FP_DIR not defined")

sys.path.insert(0, SENSE_FP_DIR)

from fp_compute_A       import fp_compute_A
from fp_compute_b       import fp_compute_b

FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    )

    parser.add_argument(
        "--smaps-npy-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--aliased-coils-npy-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--smaps-npz-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--aliased-coils-npz-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        required=True
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        required=True
    )

    parser.add_argument(
        "--save-images",
        type=str,
        required=True
    )

    return parser.parse_args()

def source_sh(path: str) -> None:
    cmd = f"source {path} >/dev/null 2>&1 && env"
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value


def main() -> None:
    args = parse_args()

    smaps_path_npy       = args.smaps_npy_path
    coils_alias_path_npy = args.aliased_coils_npy_path
    smaps_path_npz       = args.smaps_npz_path
    coils_alias_path_npz = args.aliased_coils_npz_path
    fxp_dir              = args.fxp_dir
    fp_dir               = args.fp_dir
    out_dir              = args.output_dir
    max_workers          = args.max_workers
    chunksize            = args.chunksize
    save_images          = True if (args.save_images == "True") else False

    print("[sense_runner.py]   Running sense runner with smaps:", smaps_path_npz)
    print("[sense_runner.py]   Running sense runner with coils:", coils_alias_path_npz)

    os.makedirs(out_dir, exist_ok=True)

    S_fp = np.load(smaps_path_npy).astype(np.complex128)
    y_fp = np.load(coils_alias_path_npy).astype(np.complex128)
    # reportar errores de cuantización
    S_fxp = CFxpTensor.from_npz(smaps_path_npz)
    y_fxp = CFxpTensor.from_npz(coils_alias_path_npz)

    A_fp        =  np.load(os.path.join(fp_dir, "A", "A.npy")).astype(np.complex128)
    b_fp        =  np.load(os.path.join(fp_dir, "b", "b.npy")).astype(np.complex128)
    L_fp        =  np.load(os.path.join(fp_dir, "L", "L.npy")).astype(np.complex128)
    D_fp        =  np.load(os.path.join(fp_dir, "D", "D.npy")).astype(np.complex128)
    m_hat_fp    =  np.load(os.path.join(fp_dir, "m_hat", "m_hat.npy")).astype(np.complex128)
    I_fp        =  np.load(os.path.join(fp_dir, "I", "I.npy")).astype(np.complex128)

    A_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "A", "A.npz"))
    b_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "b", "b.npz"))
    L_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "L", "L.npz"))
    D_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "D", "D.npz"))
    m_hat_fxp   = CFxpTensor.from_npz(os.path.join(fxp_dir, "m_hat", "m_hat.npz"))
    I_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "I", "I.npz"))

    # ---------------------------------------------------------
    # fp
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # fxp
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # reporte
    # ---------------------------------------------------------
    # print("[sense_runner.py]   Running write_compare_report ...")
    # out_rpt_path = os.path.join(out_dir, "report.rpt")
    # write_compare_report(
    #     out_rpt_path=out_rpt_path,
    #     S_f_input_path=smaps_path_npy,
    #     S_q_input_path=smaps_path_npz,
    #     y_f_input_path=coils_alias_path_npy,
    #     y_q_input_path=coils_alias_path_npz,
    #     A_data=A_data,
    #     b_data=b_data
    # )

    # if save_images:
    #     print("[sense_runner.py]   Saving A comparison figures ...")
    #     save_A_compare_figures(A_fp, A_fxp, out_dir, prefix="A")

    #     print("[sense_runner.py]   Saving b comparison figures ...")
    #     save_b_compare_figures(b_fp, b_fxp, out_dir, prefix="b")
    

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt

from fxp_compute_A              import fxp_compute_A   # A: (2, 2, Nx, offset)
from comparision                import compare_A
from rpt_writer                 import write_compare_A_report

# ------------------------- ENVIRONMENT SET -------------------------
SENSE_FP_DIR = os.environ.get("SENSE_FP_DIR")
if SENSE_FP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FP_DIR not defined")

sys.path.insert(0, SENSE_FP_DIR)

from fp_compute_A import fp_compute_A
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrucción SENSE Af=2 en punto flotante (núcleo np.linalg.solve)."
    )

    parser.add_argument(
        "--smaps-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy de mapas de sensibilidad S (L, Nx, Ny).",
    )

    parser.add_argument(
        "--aliased-coils-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy con imágenes de bobina aliasadas y (L, Nx, Ny_full o Ny_alias).",
    )

    parser.add_argument(
        "--smaps-npz-path",
        type=str,
        required=True,
        help="Ruta al .npz de mapas de sensibilidad cuantizados.",
    )

    parser.add_argument(
        "--aliased-coils-npz-path",
        type=str,
        required=True,
        help="Ruta al .npz con imágenes de bobina aliasadas cuantizadas.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )

    parser.add_argument(
        "--NB",
        type=int,
        required=True,
        help="Number of total bits",
    )

    parser.add_argument(
        "--NBF",
        type=int,
        required=True,
        help="Number of fractional bits.",
    )

    parser.add_argument(
        "--signed",
        type=str,
        required=True,
        help="signed/unsigned",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="round/trunc",
    )




    return parser.parse_args()

from pathlib import Path
def require_file(path_str: str, name: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] {name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"[ERROR] {name} is not a file: {path}")
    return path


def require_dir(path_str: str, name: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] {name} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"[ERROR] {name} is not a directory: {path}")
    return path

def main() -> None:
    args = parse_args()

    smaps_path_npy       = require_file(args.smaps_npy_path, "smaps_npy_path")
    coils_alias_path_npy = require_file(args.aliased_coils_npy_path, "aliased_coils_npy_path")
    smaps_path_npz       = require_file(args.smaps_npz_path, "smaps_npz_path")
    coils_alias_path_npz = require_file(args.aliased_coils_npz_path, "aliased_coils_npz_path")
    out_dir              = require_dir(args.output_dir, "output_dir")

    print("[fxp_sense.py]   Running fxp sense with smaps:", smaps_path_npz)
    print("[fxp_sense.py]   Running fxp sense with coils:", coils_alias_path_npz)

    os.makedirs(out_dir, exist_ok=True)

    S_fp = np.load(smaps_path_npy).astype(np.complex128)
    y_fp = np.load(coils_alias_path_npy).astype(np.complex128)

    S_fxp = np.load(smaps_path_npz)
    y_fxp = np.load(coils_alias_path_npz)


    print("fp_compute_A")
    A_fp    = fp_compute_A(S_fp)
    print("fxp_compute_A")
    A_fxp = fxp_compute_A(S_fxp, max_workers=8, chunksize=8)
    print("compare_A")
    rpt_A = compare_A(A_fp, A_fxp)
    print("write_compare_A_report")
    out_rpt_path = os.path.join(out_dir,"report.rpt" )
    write_compare_A_report(out_rpt_path, smaps_path_npy, smaps_path_npz, rpt_A)




    # b = fp_compute_b(S, y)

    # print("A shape:", A.shape)
    # print("b shape:", b.shape)

    # # referencia
    # m_hat_solve = fp_compute_m_hat(A, b, compute_type="numpy-linalg-solve", cholesky_type=None)
    # img_solve = fp_img_recon(m_hat_solve)

    # # L via numpy
    # m_hat_np_l = fp_compute_m_hat(A, b, compute_type="numpy-linalg-cholesky", cholesky_type=None)
    # img_np_l = fp_img_recon(m_hat_np_l)


    # # LLH
    # m_hat_llh = fp_compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LLH")
    # img_llh = fp_img_recon(m_hat_llh)

    # # LDLH
    # m_hat_ldlh = fp_compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LDLH")
    # img_ldlh = fp_img_recon(m_hat_ldlh)


    # if save_all:
    #     np.save(os.path.join(out_dir, "sense_rec_solve.npy"), img_solve)
    #     np.save(os.path.join(out_dir, "sense_rec_llh.npy"), img_llh)
    #     np.save(os.path.join(out_dir, "sense_rec_np_l.npy"), img_np_l)

    #     plt.imsave(os.path.join(out_dir, "sense_rec_solve_mag.png"), img_solve, cmap="gray")
    #     plt.imsave(os.path.join(out_dir, "sense_rec_np_l_mag.png"), img_np_l, cmap="gray")
    #     plt.imsave(os.path.join(out_dir, "sense_rec_llh_mag.png"), img_llh, cmap="gray")


    # np.save(os.path.join(out_dir, "sense_rec_ldlh.npy"), img_ldlh)
    # plt.imsave(os.path.join(out_dir, "sense_rec_ldlh.png"), img_ldlh, cmap="gray")



    print(f"Reporte guardado en {os.path.join(out_dir, 'sense_compare_report.rpt')}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.fxp_multiprocessing_compute_A              import fxp_compute_A   # A: (2, 2, Nx, offset)
from multiprocess.fxp_multiprocessing_compute_b              import fxp_compute_b   # A: (2, 2, Nx, offset)
from helpers.comparision                                     import compare_A, compare_b
from helpers.rpt_writer                                      import write_compare_report

# ------------------------- ENVIRONMENT SET -------------------------
SENSE_FP_DIR = os.environ.get("SENSE_FP_DIR")
if SENSE_FP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FP_DIR not defined")

sys.path.insert(0, SENSE_FP_DIR)

from fp_compute_A import fp_compute_A
from fp_compute_b import fp_compute_b
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
        "--max-workers",
        type=int,
        required=True,
        help="round/trunc",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        required=True,
        help="round/trunc",
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    smaps_path_npy       = args.smaps_npy_path
    coils_alias_path_npy = args.aliased_coils_npy_path
    smaps_path_npz       = args.smaps_npz_path
    coils_alias_path_npz = args.aliased_coils_npz_path
    out_dir              = args.output_dir
    max_workers          = args.max_workers
    chunksize            = args.chunksize

    print("[fxp_sense.py]   Running fxp sense with smaps:", smaps_path_npz)
    print("[fxp_sense.py]   Running fxp sense with coils:", coils_alias_path_npz)

    os.makedirs(out_dir, exist_ok=True)

    S_fp = np.load(smaps_path_npy).astype(np.complex128)
    y_fp = np.load(coils_alias_path_npy).astype(np.complex128)

    S_fxp = np.load(smaps_path_npz)
    y_fxp = np.load(coils_alias_path_npz)


    print("[fxp_sense.py]   Running fp_compute_A ...")
    A_fp    = fp_compute_A(S_fp)

    print(f"[fxp_sense.py]   Running fxp_compute_A ...")
    A_fxp = fxp_compute_A(S_fxp, max_workers, chunksize)

    print("[fxp_sense.py]   Running compare_A ...")
    A_data = compare_A(A_fp, A_fxp)



    print("[fxp_sense.py]   Running fp_compute_b ...")
    b_fp    = fp_compute_b(S_fp, y_fp)

    print(f"[fxp_sense.py]   Running fxp_compute_b ...")
    b_fxp = fxp_compute_b(S_fxp, y_fxp, max_workers, chunksize)

    print("[fxp_sense.py]   Running compare_b ...")
    b_data = compare_b(b_fp, b_fxp)



    print("[fxp_sense.py]   Running write_compare_report ...")
    out_rpt_path = os.path.join(out_dir,"report.rpt" )
    write_compare_report(
        out_rpt_path=out_rpt_path               ,
        S_f_input_path=smaps_path_npy           ,
        S_q_input_path=smaps_path_npz           ,
        y_f_input_path=coils_alias_path_npy     ,
        y_q_input_path=coils_alias_path_npz     ,
        A_data=A_data                           ,  
        b_data=b_data
    )
    


    print(f"[fxp_sense.py]   Report saved in : {out_rpt_path}")



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



    

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.fxp_multiprocessing_compute_A              import fxp_compute_A   # A: (2, 2, Nx, offset)
from multiprocess.fxp_multiprocessing_compute_b                             import fxp_compute_b   # A: (2, 2, Nx, offset)
from helpers.comparision                                     import compare_fxp_vs_fp
from helpers.rpt_writer                                      import write_compare_report
from helpers.img_savers                                      import *
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

    parser.add_argument(
        "--save-images",
        type=str,
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
    save_images        = True if (args.save_images == "True") else False

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

    A_result = fxp_compute_A(S_fxp, max_workers, chunksize)

    A_fxp = A_result["A"]
    stats_A = A_result["stats"]

    print("[fxp_sense.py]   Running compare_fxp_vs_fp for A ...")
    A_data = compare_fxp_vs_fp(A_fp, A_fxp, stats_A)

    print("[fxp_sense.py]   Running fp_compute_b ...")
    b_fp    = fp_compute_b(S_fp, y_fp)

    print(f"[fxp_sense.py]   Running fxp_compute_b ...")
    b_result = fxp_compute_b(S_fxp, y_fxp, max_workers, chunksize)
    b_fxp = b_result["b"]
    stats_b = b_result["stats"]
    print("[fxp_sense.py]   Running compare_fxp_vs_fp for b ...")
    b_data = compare_fxp_vs_fp(b_fp, b_fxp, stats_b)



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

    if save_images:
        print("[fxp_sense.py]   Saving A comparison figures ...")
        save_A_compare_figures(A_fp, A_fxp, out_dir, prefix="A")

        print("[fxp_sense.py]   Saving b comparison figures ...")
        save_b_compare_figures(b_fp, b_fxp, out_dir, prefix="b")
            

    

if __name__ == "__main__":
    main()
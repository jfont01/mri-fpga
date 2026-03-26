
#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.fxp_multiprocessing_compute_A              import fxp_multiprocessing_compute_A
from multiprocess.fxp_multiprocessing_compute_b              import fxp_multiprocessing_compute_b
from helpers.fxp_rpt_writer                                  import fxp_rpt_writer
# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")


sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import *
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
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



def main() -> None:
    args = parse_args()

    smaps_path_npz       = args.smaps_npz_path
    coils_alias_path_npz = args.aliased_coils_npz_path
    out_dir              = args.output_dir
    max_workers          = args.max_workers
    chunksize            = args.chunksize
    save_images          = True if (args.save_images == "True") else False

    print("[fxp_sense.py]   Running fxp sense with smaps:", smaps_path_npz)
    print("[fxp_sense.py]   Running fxp sense with coils:", coils_alias_path_npz)

    os.makedirs(out_dir, exist_ok=True)

    S_fxp = CFxpTensor.from_npz(smaps_path_npz)
    y_fxp = CFxpTensor.from_npz(coils_alias_path_npz)

    # ---------------------------------------------------------
    # A
    # ---------------------------------------------------------
    print("[fxp_sense.py]   Running fxp_multiprocessing_compute_A ...")
    A_fxp, stats_A = fxp_multiprocessing_compute_A(S_fxp, max_workers, chunksize)

    A_hermitian_stats = hermitian_error_metrics_A(A_fxp)
    stats_A["hermitian_checks"] = A_hermitian_stats
    stats_A["structure_checks"] = A_structure_metrics(A_fxp, eps=1e-12)
    
    print("[fxp_sense.py]   Saving A_fxp ...")
    A_dir = os.path.join(out_dir, "A")
    os.makedirs(A_dir, exist_ok=True)
    A_fxp.save_as_npz(os.path.join(A_dir, "A.npz"))
    fxp_rpt_writer(os.path.join(A_dir, "A.rpt"), stats_A, os.path.join(A_dir, "A.npz"))

    # ---------------------------------------------------------
    # b
    # ---------------------------------------------------------
    print("[fxp_sense.py]   Running fxp_multiprocessing_compute_b ...")
    b_fxp, stats_b = fxp_multiprocessing_compute_b(S_fxp, y_fxp, max_workers, chunksize)

    print("[fxp_sense.py]   Saving b_fxp ...")
    b_dir = os.path.join(out_dir, "b")
    os.makedirs(b_dir, exist_ok=True)
    b_fxp.save_as_npz(os.path.join(b_dir, "b.npz"))
    fxp_rpt_writer(os.path.join(b_dir, "b.rpt"), stats_b, os.path.join(b_dir, "b.npz"))

    # ---------------------------------------------------------
    # LD
    # ---------------------------------------------------------
    """   
    print("[fxp_sense.py]   Running fxp_compute_LD ...")
    L_fxp, D_fxp, stats_L, stats_D = fxp_multiprocessing_compute_LD(A_fxp, max_workers, chunksize)


    print("[fxp_sense.py]   Saving L_fxp ...")
    L_dir = os.path.join(out_dir, "L")
    os.makedirs(L_dir, exist_ok=True)
    L_fxp.save_as_npz(os.path.join(L_dir, "L.npz"))
    fxp_rpt_writer(os.path.join(L_dir, "L.rpt"), stats_L, os.path.join(L_dir, "L.npz"))

    print("[fxp_sense.py]   Saving D_fxp ...")
    D_dir = os.path.join(out_dir, "D")
    os.makedirs(D_dir, exist_ok=True)
    D_fxp.save_as_npz(os.path.join(D_dir, "D.npz"))
    fxp_rpt_writer(os.path.join(D_dir, "D.rpt"), stats_D, os.path.join(D_dir, "D.npz"))
    """
    # ---------------------------------------------------------
    # m_hat
    # ---------------------------------------------------------
    """    
    print("[fxp_sense.py]   Running fxp_compute_m_hat ...")
    m_hat_fxp, stats_m_hat = fxp_multiprocessing_compute_m_hat(A_fxp, b_fxp, max_workers, chunksize)

    print("[fxp_sense.py]   Saving m_hat_fxp ...")
    m_hat_dir = os.path.join(out_dir, "m_hat")
    os.makedirs(m_hat_dir, exist_ok=True)
    m_hat_fxp.save_as_npz(os.path.join(m_hat_dir, "m_hat.npz"))
    fxp_rpt_writer(os.path.join(m_hat_dir, "m_hat.rpt"), stats_m_hat, os.path.join(m_hat_dir, "m_hat.npz"))
    """
    # ---------------------------------------------------------
    # reporte
    # ---------------------------------------------------------
    stats_list = [
        stats_A,
        stats_b
        #stats_L,
        #stats_D,
        #stats_m_hat,
    ]

    paths_list = [
        os.path.join(A_dir, "A.npz"),
        os.path.join(b_dir, "b.npz")
        #os.path.join(L_dir, "L.npz"),
        #os.path.join(D_dir, "D.npz"),
       # os.path.join(m_hat_dir, "m_hat.npz"),
    ]

    global_fxp_rpt_path = os.path.join(out_dir, "global_report.rpt")

    fxp_rpt_writer(global_fxp_rpt_path, stats_list, paths_list)
    fxp_rpt_writer(global_fxp_rpt_path, stats_list, paths_list)




if __name__ == "__main__":
    main()
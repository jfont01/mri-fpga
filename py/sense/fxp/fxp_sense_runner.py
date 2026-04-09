
#!/usr/bin/env python3
import argparse, sys
import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.fxp_multiprocessing_compute_A              import fxp_multiprocessing_compute_A
from multiprocess.fxp_multiprocessing_compute_b              import fxp_multiprocessing_compute_b
from multiprocess.fxp_multiprocessing_compute_D              import fxp_multiprocessing_compute_D
from multiprocess.fxp_multiprocessing_compute_L              import fxp_multiprocessing_compute_L
from multiprocess.fxp_multiprocessing_compute_z              import fxp_multiprocessing_compute_z
from multiprocess.fxp_multiprocessing_compute_x              import fxp_multiprocessing_compute_x
from multiprocess.fxp_multiprocessing_compute_m_hat          import fxp_multiprocessing_compute_m_hat
from singleprocess.fxp_compute_I                             import fxp_compute_I
from helpers.fxp_rpt_writer                                  import fxp_rpt_writer
from helpers.fxp_save_tensor_png                             import fxp_save_tensor_png
from helpers.fxp_dat_saver                                   import save_full_tensor_dat
# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")


sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))
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
        "--NB-A",
        type=int,
        required=True
    )

    parser.add_argument(
        "--NBF-A",
        type=int,
        required=True
    )

    parser.add_argument(
        "--NB-B",
        type=int,
        required=True
    )

    parser.add_argument(
        "--NBF-B",
        type=int,
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
    NB_A                 = args.NB_A
    NBF_A                = args.NBF_A
    NB_B                 = args.NB_B
    NBF_B                = args.NBF_B
    max_workers          = args.max_workers
    chunksize            = args.chunksize
    save_images          = True if (args.save_images == "True") else False

    print("[fxp_sense_runner.py]   Loading smaps:", smaps_path_npz)
    print("[fxp_sense_runner.py]   Loading aliased coils:", coils_alias_path_npz)

    os.makedirs(out_dir, exist_ok=True)

    S_fxp = CFxpTensor.from_npz(smaps_path_npz)
    y_fxp = CFxpTensor.from_npz(coils_alias_path_npz)

    input_stimuli = {
        "S": {
            "path": smaps_path_npz,
            "shape": S_fxp.shape,
            "NB": S_fxp.NB,
            "NBF": S_fxp.NBF,
            "signed": S_fxp.signed,
        },
        "y": {
            "path": coils_alias_path_npz,
            "shape": y_fxp.shape,
            "NB": y_fxp.NB,
            "NBF": y_fxp.NBF,
            "signed": y_fxp.signed,
        },
    }
    
    # ---------------------------------------------------------
    # A
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_A ...")

    A_fxp, stats_A = fxp_multiprocessing_compute_A(S_fxp, NB_A, NBF_A, max_workers, chunksize)

    print("[fxp_sense_runner.py]   A shape:", A_fxp.shape)
    print("[fxp_sense_runner.py]   Saving A npz file ...")
    A_dir = os.path.join(out_dir, "A")
    os.makedirs(A_dir, exist_ok=True)
    A_fxp.save_as_npz(os.path.join(A_dir, "A.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving A png's ...")
        fxp_save_tensor_png(
            A_fxp,
            A_dir,
            base_names=["A00", "A01_mag", "A10_mag", "A11"],
            mode_per_channel=["real", "abs", "abs", "real"],
        )
        

    # ---------------------------------------------------------
    # b
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_b ...")

    b_fxp, stats_b = fxp_multiprocessing_compute_b(S_fxp, y_fxp, NB_B, NBF_B, max_workers, chunksize)

    print("[fxp_sense_runner.py]   b shape:", b_fxp.shape)
    print("[fxp_sense_runner.py]   Saving b npz file ...")
    b_dir = os.path.join(out_dir, "b")
    os.makedirs(b_dir, exist_ok=True)
    b_fxp.save_as_npz(os.path.join(b_dir, "b.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving b png's ...")
        fxp_save_tensor_png(
            b_fxp,
            b_dir,
            base_names=["b0_mag", "b1_mag"],
            mode_per_channel=["abs", "abs"],
        )


    # ---------------------------------------------------------
    # D
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_D ...")

    D_fxp, stats_D = fxp_multiprocessing_compute_D(A_fxp, max_workers, chunksize)

    print("[fxp_sense_runner.py]   D shape:", D_fxp.shape)
    print("[fxp_sense_runner.py]   Saving D npz file ...")
    D_dir = os.path.join(out_dir, "D")
    os.makedirs(D_dir, exist_ok=True)
    D_fxp.save_as_npz(os.path.join(D_dir, "D.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving D png's ...")
        fxp_save_tensor_png(
            D_fxp,
            D_dir,
            base_names=["D00", "D01", "D10", "D11"],
            mode_per_channel=["real", "real", "real", "real"],
        )
    # ---------------------------------------------------------
    # L
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_L ...")

    L_fxp, stats_L = fxp_multiprocessing_compute_L(A_fxp, D_fxp, max_workers, chunksize)

    print("[fxp_sense_runner.py]   L shape:", L_fxp.shape)
    print("[fxp_sense_runner.py]   Saving L npz file ...")
    L_dir = os.path.join(out_dir, "L")
    os.makedirs(L_dir, exist_ok=True)
    L_fxp.save_as_npz(os.path.join(L_dir, "L.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving L png's ...")
        fxp_save_tensor_png(
            L_fxp,
            L_dir,
            base_names=["L00", "L01", "L10_mag", "L11"],
            mode_per_channel=["real", "real", "abs", "real"],
        )


    # ---------------------------------------------------------
    # x
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_x ...")

    x_fxp, stats_x = fxp_multiprocessing_compute_x(L_fxp, b_fxp, max_workers, chunksize)

    print("[fxp_sense_runner.py]   x shape:", x_fxp.shape)
    print("[fxp_sense_runner.py]   Saving x npz file ...")
    x_dir = os.path.join(out_dir, "x")
    os.makedirs(x_dir, exist_ok=True)
    x_fxp.save_as_npz(os.path.join(x_dir, "x.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving x png's ...")
        fxp_save_tensor_png(
            x_fxp,
            x_dir,
            base_names=["x0_mag", "x1_mag"],
            mode_per_channel=["abs", "abs"],
        )


    # ---------------------------------------------------------
    # z
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_z ...")

    z_fxp, stats_z = fxp_multiprocessing_compute_z(D_fxp, x_fxp, max_workers, chunksize)

    print("[fxp_sense_runner.py]   z shape:", z_fxp.shape)
    print("[fxp_sense_runner.py]   Saving z npz file ...")
    z_dir = os.path.join(out_dir, "z")
    os.makedirs(z_dir, exist_ok=True)
    z_fxp.save_as_npz(os.path.join(z_dir, "z.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving z png's ...")
        fxp_save_tensor_png(
            z_fxp,
            z_dir,
            base_names=["z0_mag", "z1_mag"],
            mode_per_channel=["abs", "abs"],
        )

    # ---------------------------------------------------------
    # m_hat
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_multiprocessing_compute_m_hat ...")

    m_hat_fxp, stats_m_hat = fxp_multiprocessing_compute_m_hat(L_fxp, z_fxp, max_workers, chunksize)

    print("[fxp_sense_runner.py]   m_hat shape:", m_hat_fxp.shape)
    print("[fxp_sense_runner.py]   Saving m_hat npz file ...")
    m_hat_dir = os.path.join(out_dir, "m_hat")
    os.makedirs(m_hat_dir, exist_ok=True)
    m_hat_fxp.save_as_npz(os.path.join(m_hat_dir, "m_hat.npz"))
    if save_images:
        print("[fxp_sense_runner.py]   Saving m_hat png's ...")
        fxp_save_tensor_png(
            m_hat_fxp,
            m_hat_dir,
            base_names=["m_hat0_mag", "m_hat1_mag"],
            mode_per_channel=["abs", "abs"],
        )

    # ---------------------------------------------------------
    # I
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Running fxp_compute_I ...")
    stats_I = {}

    I_fxp = fxp_compute_I(m_hat_fxp, stats_I=stats_I)
    
    print("[fxp_sense_runner.py]   I shape:", I_fxp.shape)

    print("[fxp_sense_runner.py]   Saving I npz file ...")
    I_dir = os.path.join(out_dir, "I")
    os.makedirs(I_dir, exist_ok=True)
    I_fxp.save_as_npz(os.path.join(I_dir, "I.npz"))

    print("[fxp_sense_runner.py]   Saving I png's ...")
    fxp_save_tensor_png(
        I_fxp,
        I_dir,
        base_names=["I"],
        mode_per_channel=["abs"],
    )


    
    # -------------------------------------------------------------------
    # Individual report writer
    # -------------------------------------------------------------------
    stats_A["hermitian_checks"] = hermitian_error_metrics_A(A_fxp)
    stats_A["structure_checks"] = A_structure_metrics(A_fxp, eps=1e-12)
    print("[fxp_sense_runner.py]   Writing individual reports ...")
    fxp_rpt_writer(os.path.join(A_dir, "A.rpt"), stats_A, os.path.join(A_dir, "A.npz"))
    fxp_rpt_writer(os.path.join(b_dir, "b.rpt"), stats_b, os.path.join(b_dir, "b.npz"))
    fxp_rpt_writer(os.path.join(L_dir, "L.rpt"), stats_L, os.path.join(L_dir, "L.npz"))
    fxp_rpt_writer(os.path.join(D_dir, "D.rpt"), stats_D, os.path.join(D_dir, "D.npz"))
    fxp_rpt_writer(os.path.join(z_dir, "z.rpt"), stats_z, os.path.join(z_dir, "z.npz"))
    fxp_rpt_writer(os.path.join(x_dir, "x.rpt"), stats_x, os.path.join(x_dir, "x.npz"))
    fxp_rpt_writer(os.path.join(m_hat_dir, "m_hat.rpt"), stats_m_hat, os.path.join(m_hat_dir, "m_hat.npz"))
    fxp_rpt_writer(os.path.join(I_dir, "I.rpt"), stats_I, os.path.join(I_dir, "I.npz"))


    # -------------------------------------------------------------------
    # Vector Matching .dat file saver
    # -------------------------------------------------------------------
    save_full_tensor_dat(os.path.join(A_dir, "A.npz"), os.path.join(A_dir, "A.dat"))
    save_full_tensor_dat(os.path.join(b_dir, "b.npz"), os.path.join(b_dir, "b.dat"))
    save_full_tensor_dat(os.path.join(L_dir, "L.npz"), os.path.join(L_dir, "L.dat"))
    save_full_tensor_dat(os.path.join(D_dir, "D.npz"), os.path.join(D_dir, "D.dat"))
    save_full_tensor_dat(os.path.join(z_dir, "z.npz"), os.path.join(z_dir, "z.dat"))
    save_full_tensor_dat(os.path.join(x_dir, "x.npz"), os.path.join(x_dir, "x.dat"))
    save_full_tensor_dat(os.path.join(m_hat_dir, "m_hat.npz"), os.path.join(m_hat_dir, "m_hat.dat"))
    save_full_tensor_dat(os.path.join(I_dir, "I.npz"), os.path.join(I_dir, "I.dat"))

    save_full_tensor_dat(os.path.join(A_dir, "A.npz"), os.path.join(os.getenv("VM_ROOT"), "A", "py_A.dat"))
    # ---------------------------------------------------------
    # Gobal report writer
    # ---------------------------------------------------------
    print("[fxp_sense_runner.py]   Writing global report ...")
    stats_list = [
        stats_A,
        stats_b,
        stats_D,
        stats_L,
        stats_x,
        stats_z,
        stats_m_hat,
        stats_I
    ]

    paths_list = [
        os.path.join(A_dir, "A.npz"),
        os.path.join(b_dir, "b.npz"),
        os.path.join(D_dir, "D.npz"),
        os.path.join(L_dir, "L.npz"),
        os.path.join(x_dir, "x.npz"),
        os.path.join(z_dir, "z.npz"),
        os.path.join(m_hat_dir, "m_hat.npz"),
        os.path.join(I_dir, "I.npz")
    ]

    global_fxp_rpt_path = os.path.join(out_dir, "global_fxp_report.rpt")

    fxp_rpt_writer(
        global_fxp_rpt_path,
        stats_list,
        paths_list,
        input_stimuli=input_stimuli,
    )





if __name__ == "__main__":
    main()
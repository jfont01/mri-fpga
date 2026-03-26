
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from helpers.tensor_wrappers                import *
from helpers.fp_rpt_writer                  import fp_stage_stats, fp_rpt_writer
from helpers.rpt_writer_cholesky_methods    import *
from fp_compute_I                           import fp_compute_I

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
        "--output-path",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )



    return parser.parse_args()


def main() -> None:
    args = parse_args()

    smaps_path = args.smaps_npy_path
    coils_alias_path = args.aliased_coils_npy_path
    out_dir = args.output_path

    os.makedirs(out_dir, exist_ok=True)

    S = np.load(smaps_path).astype(np.complex128)
    y = np.load(coils_alias_path).astype(np.complex128)

    print("[fp_sense_runner.py]     S shape:", S.shape)
    print("[fp_sense_runner.py]     y shape:", y.shape)

    # -------------------------------------------------------------------
    # A
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_compute_A ...")
    A = fp_compute_A_tensor(S)
    print("[fp_sense_runner.py]     A shape:", A.shape)
    print("[fp_sense_runner.py]     Saving A npy and png's ...")
    A_dir = os.path.join(out_dir, "A")
    os.makedirs(A_dir, exist_ok=True)
    np.save(os.path.join(A_dir, "A.npy"), A)
    plt.imsave(os.path.join(A_dir, "A00.png"), np.real(A[0, 0]), cmap="gray")
    plt.imsave(os.path.join(A_dir, "A01_mag.png"), np.abs(A[0, 1]), cmap="gray")
    plt.imsave(os.path.join(A_dir, "A10_mag.png"), np.abs(A[1, 0]), cmap="gray")
    plt.imsave(os.path.join(A_dir, "A11.png"), np.real(A[1, 1]), cmap="gray")


    # -------------------------------------------------------------------
    # b
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_compute_b ...")
    b = fp_compute_b_tensor(S, y)
    print("[fp_sense_runner.py]     b shape:", b.shape)
    print("[fp_sense_runner.py]     Saving b npy and png's ...")
    b_dir = os.path.join(out_dir, "b")
    os.makedirs(b_dir, exist_ok=True)
    np.save(os.path.join(b_dir, "b.npy"), b)
    plt.imsave(os.path.join(b_dir, "b0_mag.png"), np.abs(b[0]), cmap="gray")
    plt.imsave(os.path.join(b_dir, "b1_mag.png"), np.abs(b[1]), cmap="gray")

    # -------------------------------------------------------------------
    # LD
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_compute_LD ...")
    L, D = fp_compute_LD_tensor(A)
    print("[fp_sense_runner.py]     L shape:", L.shape)
    print("[fp_sense_runner.py]     Saving L npy and png's ...")
    L_dir = os.path.join(out_dir, "L")
    os.makedirs(L_dir, exist_ok=True)
    np.save(os.path.join(L_dir, "L.npy"), L)
    plt.imsave(os.path.join(L_dir, "l10_mag.png"), np.abs(L[1, 0]), cmap="gray")

    print("[fp_sense_runner.py]     Saving D npy and png's ...")
    print("[fp_sense_runner.py]     D shape:", D.shape)
    D_dir = os.path.join(out_dir, "D")
    os.makedirs(D_dir, exist_ok=True)
    np.save(os.path.join(D_dir, "D.npy"), D)
    plt.imsave(os.path.join(D_dir, "d00.png"), np.real(D[0, 0]), cmap="gray")
    plt.imsave(os.path.join(D_dir, "d11.png"), np.real(D[1, 1]), cmap="gray")


    # -------------------------------------------------------------------
    # x
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_forward_subst_ldlh (computing x) ...")
    x = fp_forward_subst_ldlh_tensor(L, b)
    print("[fp_sense_runner.py]     x shape:", x.shape)
    print("[fp_sense_runner.py]     Saving x npy and png's ...")
    x_dir = os.path.join(out_dir, "x")
    os.makedirs(x_dir, exist_ok=True)
    np.save(os.path.join(x_dir, "x.npy"), x)
    plt.imsave(os.path.join(x_dir, "x0_mag.png"), np.abs(x[0]), cmap="gray")
    plt.imsave(os.path.join(x_dir, "x1_mag.png"), np.abs(x[1]), cmap="gray")


    # -------------------------------------------------------------------
    # z
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_diagonal_subst_ldlh (computing z) ...")
    z = fp_diagonal_subst_ldlh_tensor(D, x)
    print("[fp_sense_runner.py]     z shape:", z.shape)
    print("[fp_sense_runner.py]     Saving z npy and png's ...")
    z_dir = os.path.join(out_dir, "z")
    os.makedirs(z_dir, exist_ok=True)
    np.save(os.path.join(z_dir, "z.npy"), z)
    plt.imsave(os.path.join(z_dir, "z0_mag.png"), np.abs(z[0]), cmap="gray")
    plt.imsave(os.path.join(z_dir, "z1_mag.png"), np.abs(z[1]), cmap="gray")

    # -------------------------------------------------------------------
    # m_hat
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_backward_subst_ldlh (computing m_hat) ...")
    m_hat_ldlh = fp_backward_subst_ldlh_tensor(L, z)
    print("[fp_sense_runner.py]     m_hat shape:", m_hat_ldlh.shape)
    print("[fp_sense_runner.py]     Saving m_hat npy and png's ...")
    m_hat_dir = os.path.join(out_dir, "m_hat")
    os.makedirs(m_hat_dir, exist_ok=True)
    np.save(os.path.join(m_hat_dir, "m_hat.npy"), m_hat_ldlh)
    plt.imsave(os.path.join(m_hat_dir, "m_hat0_mag.png"), np.abs(m_hat_ldlh[0]), cmap="gray")
    plt.imsave(os.path.join(m_hat_dir, "m_hat1_mag.png"), np.abs(m_hat_ldlh[1]), cmap="gray")


    # -------------------------------------------------------------------
    # I
    # -------------------------------------------------------------------
    print("[fp_sense_runner.py]     Running fp_img_recon (computing I) ...")
    I_ldlh = fp_compute_I(m_hat_ldlh)
    print("[fp_sense_runner.py]     I shape:", I_ldlh.shape)
    print("[fp_sense_runner.py]     Saving I npy and png ...")
    I_dir = os.path.join(out_dir, "I")
    os.makedirs(I_dir, exist_ok=True)
    np.save(os.path.join(I_dir, "I.npy"), I_ldlh)
    plt.imsave(os.path.join(I_dir, "I.png"), I_ldlh, cmap="gray")


    # -------------------------------------------------------------------
    # Report writer
    # -------------------------------------------------------------------
    A_stats = fp_stage_stats("A", A)
    b_stats = fp_stage_stats("b", b)
    L_stats = fp_stage_stats("L", L)
    D_stats = fp_stage_stats("D", D)
    x_stats = fp_stage_stats("x", x)
    z_stats = fp_stage_stats("z", z)
    m_hat_stats = fp_stage_stats("m_hat", m_hat_ldlh)
    I_stats = fp_stage_stats("I", I_ldlh)

    print("[fp_sense_runner.py]     Writing individual reports ...")
    fp_rpt_writer(os.path.join(A_dir, "A.rpt"), A_stats, os.path.join(A_dir, "A.npy"))
    fp_rpt_writer(os.path.join(b_dir, "b.rpt"), b_stats, os.path.join(b_dir, "b.npy"))
    fp_rpt_writer(os.path.join(L_dir, "L.rpt"), L_stats, os.path.join(L_dir, "L.npy"))
    fp_rpt_writer(os.path.join(D_dir, "D.rpt"), D_stats, os.path.join(D_dir, "D.npy"))
    fp_rpt_writer(os.path.join(x_dir, "x.rpt"), x_stats, os.path.join(x_dir, "x.npy"))
    fp_rpt_writer(os.path.join(z_dir, "z.rpt"), z_stats, os.path.join(z_dir, "z.npy"))
    fp_rpt_writer(os.path.join(m_hat_dir, "m_hat.rpt"), m_hat_stats, os.path.join(m_hat_dir, "m_hat.npy"))
    fp_rpt_writer(os.path.join(I_dir, "I.rpt"), I_stats, os.path.join(I_dir, "I.npy"))

    print("[fp_sense_runner.py]     Writing global report ...")
    stats_list = [A_stats, b_stats, L_stats, D_stats, x_stats, z_stats, m_hat_stats, I_stats]
    paths_list = [
        os.path.join(A_dir, "A.npy"),
        os.path.join(b_dir, "b.npy"),
        os.path.join(L_dir, "L.npy"),
        os.path.join(D_dir, "D.npy"),
        os.path.join(x_dir, "x.npy"),
        os.path.join(z_dir, "z.npy"),
        os.path.join(m_hat_dir, "m_hat.npy"),
        os.path.join(I_dir, "I.npy"),
    ]
    fp_rpt_writer(os.path.join(out_dir, "global_fp_report.rpt"), stats_list, paths_list)

    # -------------------------------------------------------------------
    # Other methods comparision
    # -------------------------------------------------------------------
    # L via numpy
    m_hat_np_l = fp_compute_m_hat_tensor(A, b, compute_type="numpy-linalg-cholesky", cholesky_type=None)

    # LLH
    m_hat_llh = fp_compute_m_hat_tensor(A, b, compute_type="manual-solve", cholesky_type="LLH")

    # linalg solve
    m_hat_np_solve = fp_compute_m_hat_tensor(A, b, compute_type="numpy-linalg-solve", cholesky_type=None)

    rpt_writer_cholesky_methods(
        os.path.join(out_dir, "cholesky_methods_report.rpt"),
        A, b,
        m_hat_np_solve, m_hat_np_l, m_hat_llh, m_hat_ldlh
    )


if __name__ == "__main__":
    main()
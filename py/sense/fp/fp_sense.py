
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from fp_compute_A          import fp_compute_A   # A: (2, 2, Nx, offset)
from fp_compute_b          import fp_compute_b   # b: (2, Nx, offset)
from fp_compute_m_hat      import fp_compute_m_hat
from fp_img_recon          import fp_img_recon
from helpers               import save_compare_report


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

    parser.add_argument(
        "--save-all",
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
    save_all = True if (args.save_all == "True") else False

    os.makedirs(out_dir, exist_ok=True)

    S = np.load(smaps_path).astype(np.complex128)
    y = np.load(coils_alias_path).astype(np.complex128)

    print("S shape:", S.shape)
    print("y shape:", y.shape)

    A = fp_compute_A(S)
    b = fp_compute_b(S, y)

    print("A shape:", A.shape)
    print("b shape:", b.shape)

    # referencia
    m_hat_solve = fp_compute_m_hat(A, b, compute_type="numpy-linalg-solve", cholesky_type=None)
    img_solve = fp_img_recon(m_hat_solve)

    # L via numpy
    m_hat_np_l = fp_compute_m_hat(A, b, compute_type="numpy-linalg-cholesky", cholesky_type=None)
    img_np_l = fp_img_recon(m_hat_np_l)


    # LLH
    m_hat_llh = fp_compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LLH")
    img_llh = fp_img_recon(m_hat_llh)

    # LDLH
    m_hat_ldlh = fp_compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LDLH")
    img_ldlh = fp_img_recon(m_hat_ldlh)


    if save_all:
        np.save(os.path.join(out_dir, "sense_rec_solve.npy"), img_solve)
        np.save(os.path.join(out_dir, "sense_rec_llh.npy"), img_llh)
        np.save(os.path.join(out_dir, "sense_rec_np_l.npy"), img_np_l)

        plt.imsave(os.path.join(out_dir, "sense_rec_solve_mag.png"), img_solve, cmap="gray")
        plt.imsave(os.path.join(out_dir, "sense_rec_np_l_mag.png"), img_np_l, cmap="gray")
        plt.imsave(os.path.join(out_dir, "sense_rec_llh_mag.png"), img_llh, cmap="gray")


    np.save(os.path.join(out_dir, "sense_rec_ldlh.npy"), img_ldlh)
    plt.imsave(os.path.join(out_dir, "sense_rec_ldlh.png"), img_ldlh, cmap="gray")

    save_compare_report(
        os.path.join(out_dir, "sense_compare_report.rpt"),
        A, b,
        m_hat_solve, m_hat_np_l, m_hat_llh, m_hat_ldlh
    )

    print(f"Reporte guardado en {os.path.join(out_dir, 'sense_compare_report.rpt')}")

if __name__ == "__main__":
    main()
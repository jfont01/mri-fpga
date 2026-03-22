
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from compute_A          import compute_A   # A: (2, 2, Nx, offset)
from compute_b          import compute_b   # b: (2, Nx, offset)
from compute_m_hat      import compute_m_hat
from img_recon          import img_recon


def compute_error_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, test={test.shape}")

    diff = ref - test

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))


    signal_power = float(np.mean(np.abs(ref) ** 2))
    noise_power  = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "snr_db": snr_db,
    }

def save_compare_report(
    out_rpt_path: str,
    A: np.ndarray,
    b: np.ndarray,
    m_hat_solve: np.ndarray,
    m_hat_np_l: np.ndarray,
    m_hat_llh: np.ndarray,
    m_hat_ldlh: np.ndarray,
) -> None:

    m_llh_metrics = compute_error_metrics(m_hat_solve, m_hat_llh)
    m_ldlh_metrics = compute_error_metrics(m_hat_solve, m_hat_ldlh)
    m_np_l_metrics = compute_error_metrics(m_hat_solve, m_hat_np_l)

    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("=========================================================\n")
        f.write("SENSE SOLVER FLOATING POINT COMPARISON REPORT\n")
        f.write("=========================================================\n\n")

        f.write("---------------------------------------------------------\n")
        f.write("INPUT SHAPES AND DTYPES\n")
        f.write(f"A shape      : {A.shape}\n")
        f.write(f"b shape      : {b.shape}\n")
        f.write(f"m_hat shape  : {m_hat_solve.shape}\n")
        f.write(f"A dtype      : {A.dtype}\n")
        f.write(f"b dtype      : {b.dtype}\n")
        f.write(f"m_hat dtype  : {m_hat_solve.dtype}\n\n")



        f.write("---------------------------------------------------------\n")
        f.write("REFERENCE SOLVER\n")
        f.write("Reference    : np.linalg.solve\n\n")

        f.write("---------------------------------------------------------\n")
        f.write("M_HAT ERRORS VS REFERENCE\n")

        f.write("[cholesky_np_l]\n")
        f.write(f"max_abs_err  : {m_np_l_metrics['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err : {m_np_l_metrics['mean_abs_err']:.12e}\n")
        f.write(f"snr_db       : {m_np_l_metrics['snr_db']:.6f}\n\n")

        f.write("[cholesky_llh]\n")
        f.write(f"max_abs_err  : {m_llh_metrics['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err : {m_llh_metrics['mean_abs_err']:.12e}\n")
        f.write(f"snr_db       : {m_llh_metrics['snr_db']:.6f}\n\n")

        f.write("[cholesky_ldlh]\n")
        f.write(f"max_abs_err  : {m_ldlh_metrics['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err : {m_ldlh_metrics['mean_abs_err']:.12e}\n")
        f.write(f"snr_db       : {m_ldlh_metrics['snr_db']:.6f}\n\n")

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

    A = compute_A(S)
    b = compute_b(S, y)

    print("A shape:", A.shape)
    print("b shape:", b.shape)

    # referencia
    m_hat_solve = compute_m_hat(A, b, compute_type="numpy-linalg-solve", cholesky_type=None)
    img_solve = img_recon(m_hat_solve)

    # L via numpy
    m_hat_np_l = compute_m_hat(A, b, compute_type="numpy-linalg-cholesky", cholesky_type=None)
    img_np_l = img_recon(m_hat_np_l)


    # LLH
    m_hat_llh = compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LLH")
    img_llh = img_recon(m_hat_llh)

    # LDLH
    m_hat_ldlh = compute_m_hat(A, b, compute_type="manual-solve", cholesky_type="LDLH")
    img_ldlh = img_recon(m_hat_ldlh)


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
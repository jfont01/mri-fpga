import numpy as np


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
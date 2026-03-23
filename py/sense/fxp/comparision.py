import numpy as np

def compare_A_ij(
    A_ij_fp: np.ndarray,
    A_ij_fxp: np.ndarray
):


    A_ij_fxp_np = np.array([
        [A_ij_fxp[0, 0].to_complex(), A_ij_fxp[0, 1].to_complex()],
        [A_ij_fxp[1, 0].to_complex(), A_ij_fxp[1, 1].to_complex()],
    ], dtype=np.complex128)

    diff = A_ij_fp - A_ij_fxp_np

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))

    signal_power = float(np.mean(np.abs(A_ij_fp) ** 2))
    noise_power = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    return {
        "A_ref": A_ij_fp,
        "A_fix": A_ij_fxp_np,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
    }


def compare_A(
    A_ref: np.ndarray,
    A_fix: np.ndarray,
) -> dict:
    

    if A_ref.shape != A_fix.shape:
        raise ValueError(f"Shape mismatch: A_ref={A_ref.shape}, A_fix={A_fix.shape}")

    diff = A_ref - A_fix

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))

    signal_power = float(np.mean(np.abs(A_ref) ** 2))
    noise_power = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    worst_index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

    return {
        "A_ref": A_ref,
        "A_fix": A_fix,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
        "worst_index": worst_index,
    }

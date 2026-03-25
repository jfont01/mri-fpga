import numpy as np


def compare_fxp_vs_fp(
    ref: np.ndarray,
    fix: np.ndarray,
    stats: dict
) -> dict:
    

    if ref.shape != fix.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, fix={fix.shape}")

    diff = ref - fix

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))

    signal_power = float(np.mean(np.abs(ref) ** 2))
    noise_power = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    worst_index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

    return {
        "ref": ref,
        "fix": fix,
        "fxp_add": stats["fxp_add"],
        "fxp_sub": stats["fxp_sub"],
        "fxp_mul": stats["fxp_mul"],
        "sat": stats["sat"],
        "underflow": stats["underflow"],
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
        "worst_index": worst_index,
    }

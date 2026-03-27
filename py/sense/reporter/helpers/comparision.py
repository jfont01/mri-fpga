import numpy as np
import sys, os
# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor
# ------------------------- ENVIRONMENT SET -------------------------

def compare_fxp_vs_fp(
    ref: np.ndarray,
    fix: CFxpTensor
) -> dict:

    fix_np = fix.to_complex_ndarray().astype(np.complex128)
    ref_np = np.asarray(ref, dtype=np.complex128)

    if ref_np.shape != fix_np.shape:
        raise ValueError(f"Shape mismatch: ref={ref_np.shape}, fix={fix_np.shape}")

    diff = ref_np - fix_np

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))

    signal_power = float(np.mean(np.abs(ref_np) ** 2))
    noise_power = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    worst_index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

    return {
        "ref": ref_np,
        "fix": fix_np,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
        "worst_index": worst_index,
    }


import numpy as np


def compare_fp_vs_fp_arrays(
    ref: np.ndarray,
    test: np.ndarray,
) -> dict:
    
    ref_np = np.asarray(ref)
    test_np = np.asarray(test)

    if ref_np.shape != test_np.shape:
        raise ValueError(f"Shape mismatch: ref={ref_np.shape}, test={test_np.shape}")

    # Promover a complejo si alguno lo es; si no, queda real
    if np.iscomplexobj(ref_np) or np.iscomplexobj(test_np):
        ref_np = ref_np.astype(np.complex128)
        test_np = test_np.astype(np.complex128)
    else:
        ref_np = ref_np.astype(np.float64)
        test_np = test_np.astype(np.float64)

    diff = ref_np - test_np

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))

    signal_power = float(np.mean(np.abs(ref_np) ** 2))
    noise_power = float(np.mean(np.abs(diff) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    worst_index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

    return {
        "ref": ref_np,
        "fix": test_np,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
        "worst_index": worst_index,
    }
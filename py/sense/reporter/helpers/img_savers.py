import os
import numpy as np
import matplotlib.pyplot as plt
from cfxptensor import CFxpTensor

def _as_complex_ndarray(X):
    if isinstance(X, CFxpTensor):
        return X.to_complex_ndarray().astype(np.complex128)
    return np.asarray(X, dtype=np.complex128)

def save_complex_compare_figure(
    X_fp: np.ndarray,
    X_fxp: np.ndarray,
    out_path: str,
    title: str,
    mag_cmap: str = "gray",
    phase_cmap: str = "gray",
) -> None:
    if X_fp.shape != X_fxp.shape:
        raise ValueError(f"Shape mismatch: X_fp={X_fp.shape}, X_fxp={X_fxp.shape}")

    if X_fp.ndim != 2:
        raise ValueError(f"Se esperaba matriz 2D compleja, recibido shape={X_fp.shape}")
    X_fp = _as_complex_ndarray(X_fp)
    X_fxp = _as_complex_ndarray(X_fxp)
    X_diff = X_fp - X_fxp

    mag_fp = np.abs(X_fp)
    mag_fxp = np.abs(X_fxp)
    mag_diff = np.abs(X_diff)

    ph_fp = np.angle(X_fp)
    ph_fxp = np.angle(X_fxp)
    ph_diff = np.angle(X_diff)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title)

    # ---------------- Magnitudes ----------------
    im00 = ax[0, 0].imshow(mag_fp, cmap=mag_cmap)
    ax[0, 0].set_title("fp mag")
    ax[0, 0].axis("off")
    fig.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im01 = ax[0, 1].imshow(mag_fxp, cmap=mag_cmap)
    ax[0, 1].set_title("fxp mag")
    ax[0, 1].axis("off")
    fig.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im02 = ax[0, 2].imshow(mag_diff, cmap=mag_cmap)
    ax[0, 2].set_title("fp - fxp mag")
    ax[0, 2].axis("off")
    fig.colorbar(im02, ax=ax[0, 2], fraction=0.046, pad=0.04)

    # ---------------- Fases ----------------
    im10 = ax[1, 0].imshow(ph_fp, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    ax[1, 0].set_title("fp phase")
    ax[1, 0].axis("off")
    fig.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im11 = ax[1, 1].imshow(ph_fxp, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    ax[1, 1].set_title("fxp phase")
    ax[1, 1].axis("off")
    fig.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im12 = ax[1, 2].imshow(ph_diff, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    ax[1, 2].set_title("fp - fxp phase")
    ax[1, 2].axis("off")
    fig.colorbar(im12, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_tensor_compare_figures(
    X_fp: np.ndarray,
    X_fxp: np.ndarray,
    out_dir: str,
    prefix: str
) -> None:
    if X_fp.shape != X_fxp.shape:
        raise ValueError(f"Shape mismatch: X_fp={X_fp.shape}, X_fxp={X_fxp.shape}")
    X_fp = _as_complex_ndarray(X_fp)
    X_fxp = _as_complex_ndarray(X_fxp)
    os.makedirs(out_dir, exist_ok=True)

    if X_fp.ndim == 2:
        save_complex_compare_figure(
            X_fp, X_fxp,
            os.path.join(out_dir, f"{prefix}_compare.png"),
            f"{prefix} : fp vs fxp vs diff"
        )

    elif X_fp.ndim == 3:
        for i in range(X_fp.shape[0]):
            save_complex_compare_figure(
                X_fp[i], X_fxp[i],
                os.path.join(out_dir, f"{prefix}_{i}_compare.png"),
                f"{prefix}[{i}] : fp vs fxp vs diff"
            )

    elif X_fp.ndim == 4:
        for i in range(X_fp.shape[0]):
            for j in range(X_fp.shape[1]):
                save_complex_compare_figure(
                    X_fp[i, j], X_fxp[i, j],
                    os.path.join(out_dir, f"{prefix}_{i}{j}_compare.png"),
                    f"{prefix}[{i},{j}] : fp vs fxp vs diff"
                )
    else:
        raise ValueError(f"No se soporta guardar figuras para shape={X_fp.shape}")
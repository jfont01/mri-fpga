
from cfxptensor import CFxpTensor
import numpy as np
import matplotlib.pyplot as plt
import os

def fxp_save_tensor_png(
    X_fxp: CFxpTensor,
    out_dir: str,
    base_names: list[str],
    mode_per_channel: list[str],
) -> None:
    X = X_fxp.to_complex_ndarray()

    imgs = []

    if X.ndim == 2:
        # (H,W)
        imgs.append(X)

    elif X.ndim == 3:
        # (C,H,W)
        for c in range(X.shape[0]):
            imgs.append(X[c])

    elif X.ndim == 4:
        # (C0,C1,H,W)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                imgs.append(X[i, j])

    else:
        raise ValueError(f"No se soporta guardar imágenes para shape={X.shape}")

    if len(imgs) != len(base_names):
        raise ValueError(
            f"Cantidad de imágenes ({len(imgs)}) != cantidad de base_names ({len(base_names)})"
        )

    if len(imgs) != len(mode_per_channel):
        raise ValueError(
            f"Cantidad de imágenes ({len(imgs)}) != cantidad de mode_per_channel ({len(mode_per_channel)})"
        )

    os.makedirs(out_dir, exist_ok=True)

    for img, name, mode in zip(imgs, base_names, mode_per_channel):
        if mode == "real":
            out = np.real(img)
        elif mode == "imag":
            out = np.imag(img)
        elif mode == "abs":
            out = np.abs(img)
        elif mode == "phase":
            out = np.angle(img)
        else:
            raise ValueError(f"Modo no soportado: {mode}")

        plt.imsave(os.path.join(out_dir, f"{name}.png"), out, cmap="gray")
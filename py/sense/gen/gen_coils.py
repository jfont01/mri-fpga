#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def load_phantom(path: str) -> NDArray[np.complex128]:
    """Carga el phantom (anillo) desde .npy y lo convierte a complejo 2D."""
    m = np.load(path)

    if m.ndim != 2:
        raise ValueError(f"Phantom '{path}' debe ser 2D, shape=(N,N), "
                         f"pero tiene shape={m.shape}")

    # Lo tratamos como complejo (imag=0 si era real)
    return m.astype(np.complex128)


def load_sens_maps(path: str) -> NDArray[np.complex128]:
    """Carga los mapas de sensibilidad (L,N,N) desde .npy."""
    S = np.load(path)

    if S.ndim != 3:
        raise ValueError(f"Sens maps '{path}' debe ser 3D, shape=(L,N,N), "
                         f"pero tiene shape={S.shape}")

    return S.astype(np.complex128)


def compute_coil_images(
    m: NDArray[np.complex128],
    S: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Calcula las imágenes de bobina:
        y_l(x,y) = s_l(x,y) * m(x,y)

    m: (N, N)
    S: (L, N, N)
    return: y: (L, N, N)
    """
    L, N1, N2 = S.shape
    Nm1, Nm2 = m.shape

    if (N1 != Nm1) or (N2 != Nm2):
        raise ValueError(
            f"Dimensiones incompatibles: phantom {m.shape} vs sens_maps {S.shape}"
        )

    # Broadcasting: (L,N,N) * (1,N,N) -> (L,N,N)
    y = S * m[None, :, :]
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiplica un phantom (anillo, etc.) por mapas de "
                    "sensibilidad para generar imágenes de bobina."
    )

    p.add_argument(
        "--phantom",
        type=str,
        required=True,
        help="Ruta al .npy del phantom (N x N).",
    )

    p.add_argument(
        "--sens-maps",
        type=str,
        required=True,
        help="Ruta al .npy de los mapas de sensibilidad (L x N x N).",
    )

    p.add_argument(
        "--output-npy",
        type=str,
        default="coil_images.npy",
        help="Nombre del .npy de salida con las imágenes de bobina "
             "(L x N x N). Default: coil_images.npy",
    )

    p.add_argument(
        "--png-prefix",
        type=str,
        default=None,
        help="Prefijo para guardar PNG de cada bobina (magnitud). "
             "Si no se especifica, no se guardan PNG.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Cargando phantom desde: {args.phantom}")
    m = load_phantom(args.phantom)

    print(f"Cargando mapas de sensibilidad desde: {args.sens_maps}")
    S = load_sens_maps(args.sens_maps)

    print("Computando imágenes de bobina (y_l = s_l * m)...")
    y = compute_coil_images(m, S)

    # Guardar .npy
    np.save(args.output_npy, y)
    print(f"Guardado: {args.output_npy}  (shape={y.shape}, dtype={y.dtype})")

    # Opcional: guardar PNG de magnitud para cada bobina
    if args.png_prefix is not None:
        L = y.shape[0]
        for l in range(L):
            mag = np.abs(y[l])
            mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)

            fname = f"{args.png_prefix}_coil{l}.png"
            plt.imsave(fname, mag_norm, cmap="gray")
            print("Guardado PNG:", fname)


if __name__ == "__main__":
    main()


    #python3 gen_coils.py --phantom=concentric_rings_N32.npy --sens-maps=smaps_N32.npy --output-npy=coil_imagesN32.npy --png-prefix=coil_img_N32
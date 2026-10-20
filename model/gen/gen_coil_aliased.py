#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def reconstruct_images_ifft2(
    K: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    
    y = np.fft.ifft2(K, axes=(1, 2))

    return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye imágenes de bobina aplicando IFFT2 "
            "a un k-space (posiblemente aliasado)."
        )
    )

    parser.add_argument(
        "--input-npy",
        type=str,
        required=True,
        help="Archivo .npy con k-space complejo de forma (L, Nx, Ny).",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="coil_imgs_alias",
        help=(
            "Prefijo para los archivos de salida (.npy y .png). "
            "Default: coil_imgs_alias"
        ),
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colormap para las imágenes de magnitud. Default: gray",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_npy = args.input_npy
    out_name = args.output_name
    cmap = args.cmap

    if not os.path.isfile(in_npy):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {in_npy}")

    K = np.load(in_npy)
    K = np.asarray(K, dtype=np.complex128)

    print(f"[gen_coil_aliased.py]        k-space shape: {K.shape}")
    L, Nx, Ny = K.shape

    print("[gen_coil_aliased.py]        Reconstructing coil images with iff2d...")
    y = reconstruct_images_ifft2(K)
    print("[gen_coil_aliased.py]        coil images shape: ", y.shape)

    out_npy = f"{out_name}.npy"
    np.save(out_npy, y)
    print("[gen_coil_aliased.py]        Saved .npy:", out_npy)

    # Guardar magnitud y fase de cada bobina
    eps = 1e-12
    for l in range(L):
        y_l = y[l]

        # Magnitud
        mag = np.abs(y_l)
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + eps)
        fname_mag = f"{out_name}_coil{l}_mag.png"
        plt.imsave(fname_mag, mag_norm, cmap=cmap)
        print("[gen_coil_aliased.py]        Saved .png mag:", fname_mag)



if __name__ == "__main__":
    main()



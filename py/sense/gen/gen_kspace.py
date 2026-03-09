#!/usr/bin/env python
import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def compute_kspace_from_coils(
    coil_imgs: NDArray[np.complex128],
    use_fftshift: bool = False,
    norm: str | None = None,
) -> NDArray[np.complex128]:
    if coil_imgs.ndim != 3:
        raise ValueError(
            f"Se esperaba un array (L, N, N), pero se obtuvo shape={coil_imgs.shape}"
        )

    K = np.fft.fft2(coil_imgs, axes=(-2, -1), norm=norm)

    if use_fftshift:
        K = np.fft.fftshift(K, axes=(-2, -1))

    return K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Computar k-space (FFT2D) a partir de imágenes de bobina guardadas en .npy"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="coil_imagesN32.npy",
        help="Ruta al .npy con las imágenes de bobina (L x N x N). "
             "Default: coil_imagesN32.npy",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="kspace_from_coils.npy",
        help="Ruta de salida para guardar el k-space en .npy. "
             "Default: kspace_from_coils.npy",
    )

    parser.add_argument(
        "--fftshift",
        action="store_true",
        help="Si se pasa esta bandera, se aplica np.fft.fftshift al resultado.",
    )

    parser.add_argument(
        "--norm",
        type=str,
        choices=["backward", "ortho", "forward"],
        default=None,
        help="Parámetro 'norm' de np.fft.fft2 (backward/ortho/forward). "
             "Si no se especifica, se usa el default de NumPy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Cargando imágenes de bobina desde: {args.input}")
    coil_imgs = np.load(args.input)
    print(f"  shape: {coil_imgs.shape}, dtype: {coil_imgs.dtype}")

    print("Computando FFT2D por bobina...")
    K = compute_kspace_from_coils(
        coil_imgs,
        use_fftshift=args.fftshift,
        norm=args.norm,
    )

    # Guardar .npy con todo el k-space
    print(f"Guardando k-space en: {args.output}")
    np.save(args.output, K)

    # Guardar PNG por bobina: magnitud y fase
    L, N1, N2 = K.shape
    print(f"Guardando PNG de magnitud y fase para {L} bobinas...")

    for l in range(L):
        Kl = K[l]

        # Magnitud
        mag = np.abs(Kl)
        fname_mag = f"{args.output}_coil{l}_mag.png"
        plt.imsave(fname_mag, mag, cmap="gray")
        print("  Saved:", fname_mag)

        phase = np.angle(Kl)
        fname_phase = f"{args.output}_coil{l}_phase.png"
        plt.imsave(fname_phase, phase, cmap="gray")
        print("  Saved:", fname_phase)

    print("Listo.")


if __name__ == "__main__":
    main()

    #python3 gen_kspace.py --input=coil_imagesN32.npy --output=kspaceN32.npy --fftshift
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.typing import NDArray


def gen_two_disks_2d(
    N: int,
    r: float | None = None,
    amp1: float = 1.0,
    amp2: float = 1.0,
) -> NDArray[np.float64]:
    """
    Phantom con dos discos que se pliegan uno sobre otro para Af=2 en y.
    - Disco 1 centrado en (cx, N/4)
    - Disco 2 centrado en (cx, 3N/4)
    """
    if r is None:
        r = N / 6.0  # radio razonable

    Y, X = np.mgrid[0:N, 0:N]
    cx = N / 2.0
    cy1 = N / 4.0
    cy2 = 3.0 * N / 4.0

    disk1 = ((X - cx) ** 2 + (Y - cy1) ** 2) <= r**2
    disk2 = ((X - cx) ** 2 + (Y - cy2) ** 2) <= r**2

    img = amp1 * disk1.astype(np.float64) + amp2 * disk2.astype(np.float64)
    return img


def gen_two_gaussian_dots_2d(
    N: int,
    sigma: float | None = None,
    amp1: float = 1.0,
    amp2: float = 0.7,
) -> NDArray[np.float64]:
    """
    Phantom con dos "dots" gaussianos 2D.
    - Gaussiana 1 centrada en (cx, N/4)
    - Gaussiana 2 centrada en (cx, 3N/4)

    sigma controla el ancho; por defecto se toma N/16.
    """
    if sigma is None:
        sigma = N / 16.0

    Y, X = np.mgrid[0:N, 0:N]
    cx = N / 2.0
    cy1 = N / 4.0
    cy2 = 3.0 * N / 4.0

    r2_1 = (X - cx) ** 2 + (Y - cy1) ** 2
    r2_2 = (X - cx) ** 2 + (Y - cy2) ** 2

    g1 = np.exp(-r2_1 / (2.0 * sigma**2))
    g2 = np.exp(-r2_2 / (2.0 * sigma**2))

    img = amp1 * g1.astype(np.float64) + amp2 * g2.astype(np.float64)
    return img


def gen_concentric_rings_2d(
    N: int,
    A: float = 1.0,
    rings_period: float = 16.0,
    phase0: float = 0.0,
) -> NDArray[np.float64]:
    """
    Genera un phantom 2D de anillos concéntricos.

    N            : tamaño de la imagen (N x N)
    A            : amplitud máxima del patrón
    rings_period : período radial de los anillos (en píxeles)
    phase0       : fase inicial (en radianes) del coseno
    """
    c = N / 2.0
    Y, X = np.mgrid[0:N, 0:N]
    r = np.sqrt((X - c) ** 2 + (Y - c) ** 2)

    # Patrón cosenoidal radial, desplazado y escalado a [0, A]
    pattern = 0.5 * A * (1.0 + np.cos(2.0 * np.pi * r / rings_period + phase0))
    img_f = pattern.astype(np.float64)

    return img_f


def gen_shepp_logan_2d(
    N: int,
    A: float = 1.0,
) -> NDArray[np.float64]:
    """
    Phantom Shepp-Logan 2D basado en scikit-image, reescalado a NxN.

    Requiere scikit-image:
        pip install scikit-image
    """
    try:
        from skimage.data import shepp_logan_phantom
        from skimage.transform import resize
    except ImportError as e:
        raise ImportError(
            "Para usar phantom_type='shepp-logan' necesitas instalar scikit-image "
            "(pip install scikit-image)."
        ) from e

    img = shepp_logan_phantom().astype(np.float64)  # ~400x400
    img_resized = resize(
        img,
        (N, N),
        order=1,
        mode="reflect",
        anti_aliasing=True,
    ).astype(np.float64)

    # Escalamos a amplitud máxima A
    img_resized *= A / (img_resized.max() + 1e-12)
    return img_resized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generador de phantoms 2D para MRI (anillos concéntricos, dos discos, "
            "dos gaussianas o Shepp-Logan de scikit-image) y cálculo de su k-space (FFT2D)."
        )
    )

    parser.add_argument(
        "-N",
        "--size",
        type=int,
        default=64,
        help="Tamaño de la imagen (N x N). Default: 64",
    )

    parser.add_argument(
        "-A",
        "--amplitude",
        type=float,
        default=1.0,
        help="Amplitud máxima (para anillos y Shepp-Logan). Default: 1.0",
    )

    parser.add_argument(
        "--rings-period",
        type=float,
        default=16.0,
        help="Periodo de los anillos (en píxeles). Default: 16.0",
    )

    parser.add_argument(
        "--phase0",
        type=float,
        default=0.0,
        help="Fase inicial en radianes (para anillos). Default: 0.0",
    )

    parser.add_argument(
        "--phantom-type",
        type=str,
        choices=["two-disks", "rings", "two-gaussian-dots", "shepp-logan"],
        default="two-disks",
        help=(
            "Tipo de phantom a generar: "
            "'two-disks' (dos discos), "
            "'rings' (anillos concéntricos), "
            "'two-gaussian-dots' (dos manchas gaussianas) o "
            "'shepp-logan' (phantom Shepp-Logan de scikit-image). "
            "Default: two-disks"
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="phantom_concentric_rings",
        help=(
            "Nombre base de salida (.png / .npy / k-space). "
            "Default: phantom_concentric_rings"
        ),
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colour Map: jet, gray, twilight, hsv, ... Default: gray",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    N = args.size
    A = args.amplitude
    rings_period = args.rings_period
    phase0 = args.phase0
    phantom_type = args.phantom_type
    out_name = args.output_name
    cmap = args.cmap

    out_npy = f"{out_name}_N{N}.npy"
    out_png = f"{out_name}_N{N}.png"

    print(
        f"Phantom type={phantom_type}, N={N}, A={A}, "
        f"rings_period={rings_period}, phase0={phase0}"
    )

    # 1) Phantom en espacio imagen
    if phantom_type == "rings":
        img = gen_concentric_rings_2d(
            N=N,
            A=A,
            rings_period=rings_period,
            phase0=phase0,
        )
    elif phantom_type == "two-disks":
        img = gen_two_disks_2d(N=N)
    elif phantom_type == "two-gaussian-dots":
        img = gen_two_gaussian_dots_2d(N=N)
    else:  # "shepp-logan"
        img = gen_shepp_logan_2d(N=N, A=A)

    # Guardar .npy del phantom
    np.save(out_npy, img)
    print(f"Saved phantom to: {out_npy} (shape={img.shape}, dtype={img.dtype})")

    # Guardar PNG del phantom
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    plt.imsave(out_png, img_norm, cmap=cmap)
    print(f"Saved PNG to: {out_png}")

    # 2) FFT2D del phantom (k-space)
    print("Computando FFT2D del phantom (k-space)...")
    K = np.fft.fft2(img.astype(np.complex128))

    out_kspace_npy = f"{out_name}_kspace_N{N}.npy"
    np.save(out_kspace_npy, K)
    print(f"Saved k-space to: {out_kspace_npy} (shape={K.shape}, dtype={K.dtype})")

    # 3) PNGs de magnitud y fase (con fftshift para visualización)
    eps = 1e-12

    K_shift = np.fft.fftshift(K)
    mag = np.abs(K_shift)
    out_kspace_mag_png = f"{out_name}_kspace_N{N}_mag.png"
    plt.imsave(out_kspace_mag_png, mag, cmap=cmap)
    print(f"Saved k-space magnitude PNG to: {out_kspace_mag_png}")

    phase = np.angle(K_shift)
    out_kspace_phase_png = f"{out_name}_kspace_N{N}_phase.png"
    plt.imsave(out_kspace_phase_png, phase, cmap=cmap)
    print(f"Saved k-space phase PNG to: {out_kspace_phase_png}")


if __name__ == "__main__":
    main()
    # Ejemplos:
    # python3 gen_phantom.py -N=32 --phantom-type=two-disks \
    #   --output-name=phantom_two_disks --cmap=gray
    # python3 gen_phantom.py -N=32 --phantom-type=rings \
    #   --output-name=phantom_rings --cmap=gray
    # python3 gen_phantom.py -N=32 --phantom-type=two-gaussian-dots \
    #   --output-name=phantom_two_gaussians --cmap=gray
    # python3 gen_phantom.py -N=256 --phantom-type=shepp-logan \
    #   --output-name=phantom_shepp_logan --cmap=gray
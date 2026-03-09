import math, numpy as np, matplotlib.pyplot as plt, argparse
from numpy.typing import NDArray
from typing import Optional


def gen_sensitivity_maps_2d(
    N: int,
    L: int,
    radius_factor: float = 0.4,
    sigma_factor: float = 0.25,
    phase_scale: float = 1.0,
    normalize: bool = True,
) -> NDArray[np.complex128]:


    cy = N / 2.0
    cx = N / 2.0
    Y, X = np.mgrid[0:N, 0:N]

    R = radius_factor * (N / 2.0)
    sigma = sigma_factor * (N / 2.0)

    sens_maps = np.zeros((L, N, N), dtype=np.complex128)



    for l in range(L):
        theta = 2.0 * math.pi * l / L

        cx_l = cx + R * math.cos(theta)
        cy_l = cy + R * math.sin(theta)

        dx = X - cx_l
        dy = Y - cy_l
        r2 = dx**2 + dy**2

        mag = np.exp(-r2 / (2.0 * sigma**2))

        proj = (dx * math.cos(theta) + dy * math.sin(theta)) / (N / 2.0)
        phase = phase_scale * proj

        total_phase = phase

        sens_maps[l] = mag * np.exp(1j * total_phase)

    if normalize:
        power = np.sum(np.abs(sens_maps)**2, axis=0) 
        power_sqrt = np.sqrt(power)
        mask = power_sqrt > 0
        sens_maps[:, mask] /= power_sqrt[mask]

    return sens_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generador de mapas de sensibilidad sintéticos 2D"
    )

    parser.add_argument(
        "-N",
        "--size",
        type=int,
        default=64,
        help="Tamaño de la imagen (N x N). Default: 64",
    )

    parser.add_argument(
        "-L",
        "--num-coils",
        type=int,
        default=4,
        help="Número de bobinas (L). Default: 4",
    )

    parser.add_argument(
        "--radius-factor",
        type=float,
        default=1.0,
        help="Factor de radio (fracción de N/2) para la posición de las bobinas. "
             "Default: 1.0",
    )

    parser.add_argument(
        "--sigma-factor",
        type=float,
        default=0.8,
        help="Ancho de la gaussiana (fracción de N/2). Default: 0.8",
    )

    parser.add_argument(
        "--phase-scale",
        type=float,
        default=1.0,
        help="Escala de variación de fase espacial. Default: 1.0",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Si se pasa esta bandera, NO se normaliza sum_l |s_l|^2 = 1.",
    )


    parser.add_argument(
        "--output-name",
        type=str,
        default="sens_maps",
        help="Prefijo para los archivos de salida (.npy y .png). Default: sens_maps",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colour Map: jet, gray, twilight, hsv, ..."
             "Default: gray",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    N = args.size
    L = args.num_coils
    radius_factor = args.radius_factor
    sigma_factor = args.sigma_factor
    phase_scale = args.phase_scale
    normalize = not args.no_normalize

    out_name = args.output_name
    out_npy = out_name + f"_N{N}" + ".npy"
    cmap = args.cmap

    print(f"Generating sensitivity maps: N={N}, L={L}")
    print(f" radius_factor={radius_factor}, sigma_factor={sigma_factor}")
    print(f" phase_scale={phase_scale}, normalize={normalize}")

    S = gen_sensitivity_maps_2d(
        N,
        L,
        radius_factor=radius_factor,
        sigma_factor=sigma_factor,
        phase_scale=phase_scale,
        normalize=normalize
    )

    # 1) Guardar TODOS los mapas en un .npy (para SENSE)
    np.save(out_npy, S)
    print("Saved:", out_npy)

    # 2) Guardar imágenes PNG para magnitud y fase
    for l in range(L):
        # Magnitud
        mag = np.abs(S[l])
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
        fname_mag = f"{out_name}_coil{l}_L{L}_N{N}_mag.png"
        plt.imsave(fname_mag, mag_norm, cmap=cmap)
        print("Saved:", fname_mag)

        # Fase en [-pi, pi] → normalizamos a [0,1]
        phase = np.angle(S[l])
        phase_norm = (phase + np.pi) / (2 * np.pi)
        fname_phase = f"{out_name}_coil{l}_L{L}_N{N}_phase.png"
        # twilight es un colormap pensado para fases, pero puedes usar 'hsv' si prefieres
        plt.imsave(fname_phase, phase_norm, cmap=cmap)
        print("Saved:", fname_phase)


if __name__ == "__main__":
    main()

    #python3 gen_smaps.py -N=32 -L=4 --radius-factor=1.0 --sigma-factor=0.8 --phase-scale=1.0 --seed=1 --output-name=smaps --cmap=gray
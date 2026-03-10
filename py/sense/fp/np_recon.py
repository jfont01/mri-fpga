#!/usr/bin/env python3
import argparse
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from compute_A import compute_A   # A: (2, 2, Nx, offset)
from compute_b import compute_b   # b: (2, Nx, offset)
from img_recon import img_recon


def np_linalg_solve(
    A: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    _, _, Nx, offset = A.shape

    m_hat = np.zeros((Nx, offset, 2), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]   # (2, 2)
            bij = b[:, nx, ny_alias]      # (2,)

            # Núcleo de SENSE local (Af=2)
            m_hat[nx, ny_alias, :] = np.linalg.solve(Aij, bij)

    return m_hat




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrucción SENSE Af=2 en punto flotante (núcleo np.linalg.solve)."
    )

    parser.add_argument(
        "--smaps-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy de mapas de sensibilidad S (L, Nx, Ny).",
    )

    parser.add_argument(
        "--aliased-coils-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy con imágenes de bobina aliasadas y (L, Nx, Ny_full o Ny_alias).",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    smaps_path = args.smaps_npy_path
    coils_alias_path = args.aliased_coils_npy_path
    out_dir = args.output_path

    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar S y y
    S = np.load(smaps_path).astype(np.complex128)   # (L, Nx, Ny_full)
    y = np.load(coils_alias_path).astype(np.complex128)

    print("S shape:", S.shape)
    print("y shape:", y.shape)

    # 2) Calcular A y b
    A = compute_A(S)   # (2, 2, Nx, offset)
    b = compute_b(S, y)  # (2, Nx, offset)

    print("A shape:", A.shape)
    print("b shape:", b.shape)

    # 3) Resolver A m_hat = b
    m_hat = np_linalg_solve(A, b)

    # 4) Reconstruir imagen en espacio imagen
    img = img_recon(m_hat)  # magnitud, (Nx, Ny)

    # 5) Guardar resultados
    base = os.path.join(out_dir, "sense_rec")

    np.save(base + ".npy", img)
    print(f"Reconstrucción guardada en {base}.npy")

    plt.imsave(base + "_mag.png", img, cmap="gray")
    print(f"Magnitud guardada en {base}_mag.png")


if __name__ == "__main__":
    main()
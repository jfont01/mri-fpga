#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def undersample_kspace_ny(
    K: NDArray[np.complex128],
    Af: int,
) -> NDArray[np.complex128]:
    """
    Submuestrea el k-space en la dirección ny (último eje) con factor Af.

    K : array complejo de forma (L, Nx, Ny)
    Af : factor de aceleración (entero >= 1)

    Devuelve:
        K_alias : array complejo de forma (L, Nx, Ny_alias),
                  con Ny_alias = Ny / Af
    """
    if Af < 1:
        raise ValueError(f"Af debe ser >= 1, recibido: {Af}")

    if K.ndim != 3:
        raise ValueError(
            f"Se esperaba un k-space 3D (L, Nx, Ny), pero K.ndim={K.ndim}"
        )

    L, Nx, Ny = K.shape

    if Ny % Af != 0:
        raise ValueError(
            f"Ny={Ny} no es múltiplo de Af={Af}. "
            "Necesitamos Ny % Af == 0 para submuestreo uniforme."
        )

    # Índices de líneas de ky que conservamos: 0, Af, 2Af, ...
    ky_idx = np.arange(0, Ny, Af)  # tamaño Ny / Af
    K_alias = K[:, :, ky_idx]      # forma (L, Nx, Ny / Af)

    return K_alias


def build_full_ny_from_alias(
    K_alias: NDArray[np.complex128],
    Ny_full: int,
    Af: int,
) -> NDArray[np.complex128]:
    """
    Construye un k-space de tamaño completo (L, Nx, Ny_full) a partir
    del k-space submuestreado (L, Nx, Ny/Af) rellenando con ceros.

    Las líneas retenidas se colocan en las posiciones 0, Af, 2Af, ...
    y el resto se deja a cero.
    """
    L, Nx, Ny_alias = K_alias.shape

    if Ny_full % Af != 0 or Ny_alias != Ny_full // Af:
        raise ValueError(
            f"Inconsistencia en Ny_full={Ny_full}, Af={Af}, Ny_alias={Ny_alias}"
        )

    K_full = np.zeros((L, Nx, Ny_full), dtype=np.complex128)

    ky_idx = np.arange(0, Ny_full, Af)
    K_full[:, :, ky_idx] = K_alias

    return K_full


def undersample_kspace_nx(
    K: NDArray[np.complex128],
    Af: int,
) -> NDArray[np.complex128]:
    """
    Submuestrea el k-space en la dirección nx (eje 1) con factor Af.

    K : array complejo de forma (L, Nx, Ny)
    Af : factor de aceleración (entero >= 1)

    Devuelve:
        K_alias : array complejo de forma (L, Nx_alias, Ny),
                  con Nx_alias = Nx / Af
    """
    if Af < 1:
        raise ValueError(f"Af debe ser >= 1, recibido: {Af}")

    if K.ndim != 3:
        raise ValueError(
            f"Se esperaba un k-space 3D (L, Nx, Ny), pero K.ndim={K.ndim}"
        )

    L, Nx, Ny = K.shape

    if Nx % Af != 0:
        raise ValueError(
            f"Nx={Nx} no es múltiplo de Af={Af}. "
            "Necesitamos Nx % Af == 0 para submuestreo uniforme."
        )

    # Índices de columnas de kx que conservamos: 0, Af, 2Af, ...
    kx_idx = np.arange(0, Nx, Af)  # tamaño Nx / Af
    K_alias = K[:, kx_idx, :]      # forma (L, Nx / Af, Ny)

    return K_alias


def build_full_nx_from_alias(
    K_alias: NDArray[np.complex128],
    Nx_full: int,
    Af: int,
) -> NDArray[np.complex128]:
    """
    Construye un k-space de tamaño completo (L, Nx_full, Ny) a partir
    del k-space submuestreado (L, Nx/Af, Ny) rellenando con ceros.

    Las columnas retenidas se colocan en las posiciones 0, Af, 2Af, ...
    y el resto se deja a cero.
    """
    L, Nx_alias, Ny = K_alias.shape

    if Nx_full % Af != 0 or Nx_alias != Nx_full // Af:
        raise ValueError(
            f"Inconsistencia en Nx_full={Nx_full}, Af={Af}, Nx_alias={Nx_alias}"
        )

    K_full = np.zeros((L, Nx_full, Ny), dtype=np.complex128)

    kx_idx = np.arange(0, Nx_full, Af)  # mismas posiciones
    K_full[:, kx_idx, :] = K_alias

    return K_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aplica undersampling en el k-space con un factor de aceleración Af, "
            "en la dirección nx o ny."
        )
    )

    parser.add_argument(
        "--input-npy",
        type=str,
        required=True,
        help="Archivo .npy con k-space complejo de forma (L, Nx, Ny).",
    )

    parser.add_argument(
        "--acc-factor",
        "-A",
        type=int,
        required=True,
        help="Factor de aceleración Af (entero >= 1).",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="kspace_undersampled",
        help=(
            "Prefijo para los archivos de salida (.npy y .png). "
            "Default: kspace_undersampled"
        ),
    )

    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y"],
        default="y",
        help=(
            "Eje de undersampling en k-space: 'y' (Ny, ky, último eje) "
            "o 'x' (Nx, kx, eje 1). Default: y"
        ),
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colormap para las imágenes PNG (magnitud y fase). Default: gray",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Además de guardar el k-space reducido, "
            "genera un k-space de tamaño completo en la dirección "
            "submuestreada rellenando con ceros y usa ese para las imágenes PNG."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_npy = args.input_npy
    Af = args.acc_factor
    out_name = args.output_name
    cmap = args.cmap
    use_full = args.full   
    axis = args.axis

    K = np.load(in_npy)
    K = np.asarray(K, dtype=np.complex128)

    print(f"[gen_aliased_kspace.py]      Original k-space shape: {K.shape}")
    L, Nx_full, Ny_full = K.shape

    # Undersampling según eje seleccionado
    if axis == "y":
        print(f"Aplicando undersampling en ny (ky) con Af = {Af} ...")
        K_alias = undersample_kspace_ny(K, Af)
    elif axis == "x":
        print(f"Aplicando undersampling en nx (kx) con Af = {Af} ...")
        K_alias = undersample_kspace_nx(K, Af)
    else:
        raise ValueError(f"Eje de undersampling no soportado: {axis}")

    print(f"[gen_aliased_kspace.py]    New aliased k-space shape: {K_alias.shape}")

    # Guardamos SIEMPRE el k-space reducido
    out_npy_alias = f"{out_name}.npy"
    np.save(out_npy_alias, K_alias)
    print("[gen_aliased_kspace.py]   Saved:", out_npy_alias)

    # Si el usuario pide tamaño completo, construimos K_full rellenando con ceros
    if use_full:
        if axis == "y":
            print(
                "Construyendo k-space de tamaño completo (Ny) "
                "con ceros en líneas faltantes..."
            )
            K_for_imgs = build_full_ny_from_alias(K_alias, Ny_full=Ny_full, Af=Af)
            out_npy_full = f"{out_name}_Af{Af}_fullNy.npy"
        else:  # axis == "x"
            print(
                "Construyendo k-space de tamaño completo (Nx) "
                "con ceros en columnas faltantes..."
            )
            K_for_imgs = build_full_nx_from_alias(K_alias, Nx_full=Nx_full, Af=Af)
            out_npy_full = f"{out_name}_Af{Af}_fullNx.npy"

        np.save(out_npy_full, K_for_imgs)
        print("[gen_aliased_kspace.py]    Saved zpadded:", out_npy_full)
    else:
        # Si no se pide full, usamos el reducido para generar PNGs
        K_for_imgs = K_alias

    # Guardar magnitud y fase de cada bobina para K_for_imgs
    L, Nx_img, Ny_img = K_for_imgs.shape

    for l in range(L):
        K_l = K_for_imgs[l]

        # Magnitud: usamos log para ver mejor el rango dinámico
        mag = np.abs(K_l)

        if use_full:
            suffix = "_mag_fullNy" if axis == "y" else "_mag_fullNx"
        else:
            suffix = "_mag"

        fname_mag = f"{out_name}_coil{l}_Af{Af}{suffix}.png"
        plt.imsave(fname_mag, mag, cmap=cmap)
        print("[gen_aliased_kspace.py]      Saved mag .png:", fname_mag)

        # Fase en [-pi, pi] → [0, 1]
        phase = np.angle(K_l)

        if use_full:
            suffix_p = "_phase_fullNy" if axis == "y" else "_phase_fullNx"
        else:
            suffix_p = "_phase"

        fname_phase = f"{out_name}_coil{l}_Af{Af}{suffix_p}.png"
        plt.imsave(fname_phase, phase, cmap=cmap)
        print("[gen_aliased_kspace.py]      Saved phase .png:", fname_phase)


if __name__ == "__main__":
    main()
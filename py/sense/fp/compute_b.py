#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def compute_A(
        S: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
        A = S^H S
    """

    A = S.conj().T @ S

    return A

def compute_b(
        S: NDArray[np.complex128],
        y: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    
    b = S.conj().T @ y

    return b



def compute_Ab_block(
    S_mat: NDArray[np.complex128],
    y_vec: NDArray[np.complex128],
    reg_eps: float = 0.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Dado un bloque local de SENSE:
        y = S * rho
    construye explícitamente:

        A = S^H S     (R x R)
        b = S^H y     (R,)

    y aplica una regularización Tikhonov suave opcional:

        A_reg = A + lambda * I

    donde:
        lambda = reg_eps * trace(A) / R

    Parámetros
    ----------
    S_mat : (L, R) complejo
        Matriz de sensibilidades local.

    y_vec : (L,) complejo
        Vector de datos aliasados local.

    reg_eps : float
        Factor de regularización. Si es 0, no se regulariza.

    Devuelve
    --------
    A : (R, R) complejo
    b : (R,) complejo
    """
    # S^H S
    A = S_mat.conj().T @ S_mat  # (R, R)

    # S^H y
    b = S_mat.conj().T @ y_vec  # (R,)

    if reg_eps > 0.0:
        trace_val = np.trace(A).real
        R = A.shape[0]
        if trace_val > 0.0:
            lam = reg_eps * trace_val / R
            A = A + lam * np.eye(R, dtype=np.complex128)

    return A, b


def sense_recon_1d(
    coil_aliased: NDArray[np.complex128],
    sens_maps: NDArray[np.complex128],
    acc_factor: int,
    axis: str = "y",
    reg_eps: float = 1e-3,
) -> NDArray[np.complex128]:
    """
    Reconstrucción SENSE 1D en flotante usando ecuaciones normales:

        A rho = b
    con
        A = S^H S
        b = S^H y

    coil_aliased : array complejo (L, Nx, Ny)
        Imágenes de bobina aliasadas (salida de gen_coil_aliased.py).

    sens_maps    : array complejo (L, Nx, Ny)
        Mapas de sensibilidad complejos.

    acc_factor   : int
        Factor de aceleración R (>= 1).

    axis         : str
        'y' -> aliasing en la dimensión Ny (última)
        'x' -> aliasing en la dimensión Nx (intermedia)

    reg_eps      : float
        Parámetro de regularización Tikhonov (default: 1e-3).

    Devuelve
    --------
    img_rec : (Nx, Ny) complejo
        Imagen reconstruida.
    """
    if coil_aliased.shape != sens_maps.shape:
        raise ValueError(
            f"coil_aliased.shape={coil_aliased.shape} y "
            f"sens_maps.shape={sens_maps.shape} no coinciden"
        )

    if coil_aliased.ndim != 3:
        raise ValueError(
            f"Se esperaba (L, Nx, Ny), pero coil_aliased.ndim={coil_aliased.ndim}"
        )

    L, Nx, Ny = coil_aliased.shape
    R = acc_factor

    if axis not in ("x", "y"):
        raise ValueError(f"axis debe ser 'x' o 'y', recibido: {axis}")

    img_rec = np.zeros((Nx, Ny), dtype=np.complex128)

    if axis == "y":
        # Alias a lo largo de Ny
        if Ny % R != 0:
            raise ValueError(f"Ny={Ny} no es múltiplo de R={R}")

        block = Ny // R  # tamaño del bloque sin alias

        for ix in range(Nx):
            for iy_alias in range(block):
                # Vector y (L,) de datos aliasados en este punto
                y_vec = coil_aliased[:, ix, iy_alias]

                # Construimos matriz S (L, R) con las sensibilidades
                S_mat = np.zeros((L, R), dtype=np.complex128)
                valid_mask = np.zeros(R, dtype=bool)

                for r in range(R):
                    iy_real = iy_alias + r * block
                    S_col = sens_maps[:, ix, iy_real]
                    S_mat[:, r] = S_col
                    if np.any(np.abs(S_col) > 1e-8):
                        valid_mask[r] = True

                # Si todas las sensibilidades son ~0, no reconstruimos
                if not np.any(valid_mask):
                    continue

                # Aquí definimos A y b tal como en la teoría:
                #   A = S^H S
                #   b = S^H y
                A, b = compute_Ab_block(S_mat, y_vec, reg_eps=reg_eps)

                # Resolvemos A rho = b
                try:
                    rho = np.linalg.solve(A, b)  # (R,)
                except np.linalg.LinAlgError:
                    # Si A está mal condicionada, saltamos este grupo
                    continue

                # Escribimos los R píxeles reconstruidos en sus posiciones reales
                for r in range(R):
                    iy_real = iy_alias + r * block
                    img_rec[ix, iy_real] = rho[r]

    else:  # axis == "x"
        # Alias a lo largo de Nx
        if Nx % R != 0:
            raise ValueError(f"Nx={Nx} no es múltiplo de R={R}")

        block = Nx // R

        for iy in range(Ny):
            for ix_alias in range(block):
                y_vec = coil_aliased[:, ix_alias, iy]

                S_mat = np.zeros((L, R), dtype=np.complex128)
                valid_mask = np.zeros(R, dtype=bool)

                for r in range(R):
                    ix_real = ix_alias + r * block
                    S_col = sens_maps[:, ix_real, iy]
                    S_mat[:, r] = S_col
                    if np.any(np.abs(S_col) > 1e-8):
                        valid_mask[r] = True

                if not np.any(valid_mask):
                    continue

                # A = S^H S, b = S^H y (con regularización)
                A, b = compute_Ab_block(S_mat, y_vec, reg_eps=reg_eps)

                try:
                    rho = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                for r in range(R):
                    ix_real = ix_alias + r * block
                    img_rec[ix_real, iy] = rho[r]

    return img_rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrucción SENSE 1D (float) a partir de imágenes aliasadas de bobina y mapas de sensibilidad."
    )

    parser.add_argument(
        "--coil-aliased",
        type=str,
        required=True,
        help="Archivo .npy con imágenes de bobina aliasadas (L, Nx, Ny).",
    )

    parser.add_argument(
        "--sens-maps",
        type=str,
        required=True,
        help="Archivo .npy con mapas de sensibilidad (L, Nx, Ny).",
    )

    parser.add_argument(
        "--acc-factor",
        "-A",
        type=int,
        required=True,
        help="Factor de aceleración (R).",
    )

    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y"],
        required=True,
        help="Dirección de aliasing en imagen: 'x' o 'y' (debe coincidir con el undersampling en k-space).",
    )

    parser.add_argument(
        "--reg-eps",
        type=float,
        default=1e-3,
        help="Factor de regularización Tikhonov (default: 1e-3).",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="sense_recon",
        help="Prefijo de salida para .npy y .png (magnitud). Default: sense_recon",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colormap para la imagen reconstruida. Default: gray",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Cargando imágenes aliasadas desde: {args.coil_aliased}")
    coil_aliased = np.load(args.coil_aliased)
    coil_aliased = np.asarray(coil_aliased, dtype=np.complex128)
    print(f"  coil_aliased.shape = {coil_aliased.shape}, dtype = {coil_aliased.dtype}")

    print(f"Cargando mapas de sensibilidad desde: {args.sens_maps}")
    sens_maps = np.load(args.sens_maps)
    sens_maps = np.asarray(sens_maps, dtype=np.complex128)
    print(f"  sens_maps.shape = {sens_maps.shape}, dtype = {sens_maps.dtype}")

    print(
        f"Reconstruyendo SENSE con R={args.acc_factor}, axis={args.axis}, reg_eps={args.reg_eps}..."
    )
    img_rec = sense_recon_1d(
        coil_aliased=coil_aliased,
        sens_maps=sens_maps,
        acc_factor=args.acc_factor,
        axis=args.axis,
        reg_eps=args.reg_eps,
    )

    out_npy = f"{args.output_name}.npy"
    np.save(out_npy, img_rec)
    print(f"Reconstrucción guardada en: {out_npy} (shape={img_rec.shape})")

    # Magnitud para inspección visual
    mag = np.abs(img_rec)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    out_png = f"{args.output_name}_mag.png"
    plt.imsave(out_png, mag_norm, cmap=args.cmap)
    print(f"Magnitud guardada en: {out_png}")


if __name__ == "__main__":
    main()
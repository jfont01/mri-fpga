#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def fp_compute_b_i(
    S: NDArray[np.complex128],
    y: NDArray[np.complex128],
    nx: int,
    ny_alias: int
) -> NDArray[np.complex128]:

    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    ny0 = ny_alias
    ny1 = ny_alias + offset

    b0 = 0.0 + 0.0j
    b1 = 0.0 + 0.0j

    #s0 = S[:, nx, ny0]        
    #s1 = S[:, nx, ny1]        
    #S_block = np.stack([s0, s1], axis=1)
    #A = S_block.conj().T @ S_block

    for l in range(L):
        s0 = S[l, nx, ny0]  # s_l[nx, ny^(0)]
        s1 = S[l, nx, ny1]  # s_l[nx, ny^(1)]
        y0 = y[l, nx, ny_alias]

        b0 += np.conj(s0)*y0
        b1 += np.conj(s1)*y0


    bi = np.array([b0, b1], dtype=np.complex128)

    return bi

def fp_compute_b(
        S: NDArray[np.complex128],
        y: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    
    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    b = np.zeros((2, Nx, offset), dtype=np.complex128)


    for nx in range(Nx):                # recorre nx de [0, Nx - 1]
        for ny_alias in range(offset):  # recorre ny_alias de [0, Ny/2 - 1]
            bi = fp_compute_b_i(S, y, nx, ny_alias)  # (2,1)
            b[:, nx, ny_alias] = bi
    return b

def main() -> None:
    # 1) Cargar mapas de sensibilidad
    S = np.load("smap_N32.npy").astype(np.complex128)
    y = np.load("coil_aliased_Af2_axisy.npy").astype(np.complex128)
    print("S shape: ", S.shape)
    print("y shape: ", y.shape)

    b = fp_compute_b(S, y)

    print("b shape: ", b.shape)

if __name__ == "__main__":
    main()
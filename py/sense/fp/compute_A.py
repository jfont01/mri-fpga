#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray

def compute_A_ij(
    S: NDArray[np.complex128],
    nx: int,
    ny_alias: int
) -> NDArray[np.complex128]:

    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    ny0 = ny_alias
    ny1 = ny_alias + offset

    A00 = 0.0 + 0.0j
    A11 = 0.0 + 0.0j
    A01 = 0.0 + 0.0j

    #s0 = S[:, nx, ny0]        
    #s1 = S[:, nx, ny1]        
    #S_block = np.stack([s0, s1], axis=1)
    #A = S_block.conj().T @ S_block

    for l in range(L):
        s0 = S[l, nx, ny0]  # s_l[nx, ny^(0)]
        s1 = S[l, nx, ny1]  # s_l[nx, ny^(1)]

        A00 += np.abs(s0)**2
        A11 += np.abs(s1)**2
        A01 += np.conj(s0) * s1
    A10 = np.conj(A01)

    A = np.array([[A00, A01],
                  [A10, A11]], dtype=np.complex128)
    return A


def compute_A(
    S: NDArray[np.complex128]
)-> NDArray[np.complex128]:
    
    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    A = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):                # recorre nx de [0, Nx - 1]
        for ny_alias in range(offset):  # recorre ny_alias de [0, Ny/2 - 1]
            Aij = compute_A_ij(S, nx, ny_alias)  # (2,2)
            A[:, :, nx, ny_alias] = Aij
    return A



def main() -> None:
    # 1) Cargar mapas de sensibilidad
    S = np.load("smap_N32.npy").astype(np.complex128)
    print(S.shape)

    A = compute_A(S)

    print(A.shape)

if __name__ == "__main__":
    main()
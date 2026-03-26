#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray

def fp_compute_A_ij(
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


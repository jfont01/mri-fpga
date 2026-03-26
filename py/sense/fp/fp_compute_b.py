#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray

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

    for l in range(L):
        s0 = S[l, nx, ny0]  # s_l[nx, ny^(0)]
        s1 = S[l, nx, ny1]  # s_l[nx, ny^(1)]
        y0 = y[l, nx, ny_alias]

        b0 += np.conj(s0)*y0
        b1 += np.conj(s1)*y0


    bi = np.array([b0, b1], dtype=np.complex128)

    return bi

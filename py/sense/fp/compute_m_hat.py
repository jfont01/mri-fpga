#!/usr/bin/env python3
import argparse
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from compute_A import compute_A   # A: (2, 2, Nx, offset)
from compute_b import compute_b   # b: (2, Nx, offset)
from img_recon import img_recon
from cholesky import cholesky


def np_cholesky(
    Aij: NDArray[np.complex128],
    bi: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    
    # 1) Factorización de Cholesky
    L = np.linalg.cholesky(Aij)   # A = L L^H

    # 2) Forward substitution: L z = b
    z = np.zeros(2, dtype=np.complex128)
    z[0] = bi[0] / L[0, 0]
    z[1] = (bi[1] - L[1, 0] * z[0]) / L[1, 1]

    # 3) Backward substitution: L^H m = z
    m = np.zeros(2, dtype=np.complex128)
    LH = L.conj().T
    m[1] = z[1] / LH[1, 1]
    m[0] = (z[0] - LH[0, 1] * m[1]) / LH[0, 0]

    return m

def compute_m_hat(
    A: NDArray[np.complex128],
    b: NDArray[np.complex128],
    COMPUTE_TYPE: str = "manual-solve"
) -> NDArray[np.complex128]:
    assert COMPUTE_TYPE in ["numpy-linalg-cholesky", "numpy-linalg-solve", "manual-solve"]

    _, _, Nx, offset = A.shape


    m_hat = np.zeros((Nx, offset, 2), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]   # (2, 2)

            A00 = Aij[0, 0].real    #Criterio de Sylvester
            detA = np.linalg.det(Aij).real
            eigvals = np.linalg.eigvalsh(Aij)

            is_hpd = (A00 > 0.0) and (detA > 0.0) and (eigvals[0] > 0.0) and (eigvals[1] > 0.0)

            if not is_hpd:
                print(f"Bloque no HPD en nx={nx}, ny_alias={ny_alias}")

            
            

            bi = b[:, nx, ny_alias]      # (2,)

            match COMPUTE_TYPE:
                case "numpy-linalg-cholesky":
                    m_hat[nx, ny_alias, :] = np_cholesky(Aij, bi)
                case "numpy-linalg-solve":
                    m_hat[nx, ny_alias, :] = np.linalg.solve(Aij, bi)
                case "manual-solve":
                    m_hat[nx, ny_alias, :] = cholesky(Aij, bi)


    return m_hat





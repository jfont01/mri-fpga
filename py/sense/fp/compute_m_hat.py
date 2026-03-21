import numpy as np
from numpy.typing import NDArray

from cholesky_LLH import cholesky_llh
from cholesky_LDLH import cholesky_ldlh


def compute_m_hat(A, b, compute_type, cholesky_type=None):
    if A.shape[0:2] != (2, 2):
        raise ValueError(f"A debe tener shape (2,2,Nx,offset), recibió {A.shape}")
    if b.shape[0] != 2:
        raise ValueError(f"b debe tener shape (2,Nx,offset), recibió {b.shape}")

    _, _, Nx, offset = A.shape

    # NUEVA NOTACIÓN: (Af, Nx, offset)
    m_hat = np.zeros((2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]
            bi  = b[:, nx, ny_alias]

            if compute_type == "numpy-linalg-solve":
                m_hat[:, nx, ny_alias] = np.linalg.solve(Aij, bi)

            elif compute_type == "numpy-linalg-cholesky":
                L = np.linalg.cholesky(Aij)
                z = np.linalg.solve(L, bi)
                m_hat[:, nx, ny_alias] = np.linalg.solve(L.conj().T, z)

            elif compute_type == "manual-solve":
                if cholesky_type == "LLH":
                    m_hat[:, nx, ny_alias] = cholesky_llh(Aij, bi)
                elif cholesky_type == "LDLH":
                    m_hat[:, nx, ny_alias] = cholesky_ldlh(Aij, bi)
                else:
                    raise ValueError(f"cholesky_type inválido: {cholesky_type}")

            else:
                raise ValueError(f"compute_type inválido: {compute_type}")

    return m_hat






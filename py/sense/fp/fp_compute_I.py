import numpy as np
from numpy.typing import NDArray



def fp_compute_I(
    m_hat: NDArray[np.complex128],
) -> NDArray[np.float64]:
    _, Nx, offset = m_hat.shape

    Af = 2
    Ny = offset * Af

    I = np.zeros((Nx, Ny), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            m0 = m_hat[0, nx, ny_alias]  # píxel en ny^(0)
            m1 = m_hat[1, nx, ny_alias]  # píxel en ny^(1)

            ny0 = ny_alias
            ny1 = ny_alias + offset

            I[nx, ny0] = m0
            I[nx, ny1] = m1

    return np.abs(I)

import numpy as np
from numpy.typing import NDArray



def img_recon(
    m_hat: NDArray[np.complex128],
) -> NDArray[np.float64]:
    Nx, offset, comps = m_hat.shape
    assert comps == 2, "Esperamos Af=2 ⇒ vector de longitud 2"

    Af = 2
    Ny = offset * Af

    img_rec = np.zeros((Nx, Ny), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            m0 = m_hat[nx, ny_alias, 0]  # píxel en ny^(0)
            m1 = m_hat[nx, ny_alias, 1]  # píxel en ny^(1)

            ny0 = ny_alias
            ny1 = ny_alias + offset

            img_rec[nx, ny0] = m0
            img_rec[nx, ny1] = m1

    return np.abs(img_rec)

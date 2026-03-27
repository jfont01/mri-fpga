import numpy as np
from numpy.typing import NDArray


def fp_compute_LD_ij(
    A: NDArray[np.complex128],
    eps: float = 1e-12,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    
    a00 = A[0, 0]
    a10 = A[1, 0]
    a11 = A[1, 1]

    a00r = float(np.real(a00))
    a11r = float(np.real(a11))

    if a00r <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD: d0=a00={a00r} no es estrictamente positivo"
        )

    d0 = a00r
    l10 = a10 / d0

    d1 = a11r - (np.abs(a10) ** 2) / d0
    if d1 <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD o está muy mal condicionada: d1={d1}"
        )

    L = np.array(
        [[1.0 + 0.0j, 0.0 + 0.0j],
         [l10,        1.0 + 0.0j]],
        dtype=np.complex128
    )

    D = np.array(
        [[d0 + 0.0j, 0.0 + 0.0j],
         [0.0 + 0.0j, d1 + 0.0j]],
        dtype=np.complex128
    )

    return L, D


def fp_forward_subst_ldlh_i(
    L: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    y = np.zeros(2, dtype=np.complex128)

    y[0] = b[0]
    y[1] = b[1] - L[1, 0] * y[0]

    return y


def fp_diagonal_subst_ldlh_i(
    D: NDArray[np.complex128],
    y: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:

    d0 = float(np.real(D[0, 0]))
    d1 = float(np.real(D[1, 1]))

    if d0 <= eps or d1 <= eps:
        raise np.linalg.LinAlgError(
            f"D no es invertible o no es positiva: d0={d0}, d1={d1}"
        )

    z = np.zeros(2, dtype=np.complex128)
    z[0] = y[0] / D[0, 0]
    z[1] = y[1] / D[1, 1]

    return z


def fp_backward_subst_ldlh_i(
    L: NDArray[np.complex128],
    z: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    m_hat = np.zeros(2, dtype=np.complex128)

    m_hat[1] = z[1]
    m_hat[0] = z[0] - np.conj(L[1, 0]) * m_hat[1]

    return m_hat


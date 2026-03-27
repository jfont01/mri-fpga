import numpy as np
from numpy.typing import NDArray


def fp_compute_L(
    A: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:

    a00 = A[0, 0]
    a10 = A[1, 0]
    a11 = A[1, 1]

    a00r = float(np.real(a00))
    a11r = float(np.real(a11))

    if a00r <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD: a00={a00r} no es estrictamente positivo"
        )

    l00 = np.sqrt(a00r)
    l10 = a10 / l00

    tmp = a11r - np.abs(l10) ** 2
    if tmp <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD o está muy mal condicionada: l11^2={tmp}"
        )

    l11 = np.sqrt(tmp)

    L = np.array(
        [[l00, 0.0 + 0.0j],
         [l10, l11]],
        dtype=np.complex128
    )
    return L

def fp_forward_subst(
    L: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    z = np.zeros(2, dtype=np.complex128)

    z[0] = b[0] / L[0, 0]
    z[1] = (b[1] - L[1, 0] * z[0]) / L[1, 1]

    return z

def fp_backward_subst(
    L: NDArray[np.complex128],
    z: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    m_hat = np.zeros(2, dtype=np.complex128)

    m_hat[1] = z[1] / np.conj(L[1, 1])
    m_hat[0] = (z[0] - np.conj(L[1, 0]) * m_hat[1]) / np.conj(L[0, 0])

    return m_hat



def fp_compute_m_hat_i_llh(
    Aij: NDArray[np.complex128],
    bij: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:

    L = fp_compute_L(Aij, eps=eps)
    z = fp_forward_subst(L, bij)
    m_hat = fp_backward_subst(L, z)
    return m_hat
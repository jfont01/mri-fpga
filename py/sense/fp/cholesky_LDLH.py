import numpy as np
from numpy.typing import NDArray


def compute_LD(
    A: NDArray[np.complex128],
    eps: float = 1e-12,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Factorización LDL^H para A Hermitiana HPD de tamaño 2x2.

    A = L D L^H

    L : (2,2), triangular inferior con diagonal unitaria
    D : (2,2), diagonal real positiva
    """

    if A.shape != (2, 2):
        raise ValueError(f"Se esperaba A con shape (2,2), recibido {A.shape}")

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


def forward_subst_ldlh(
    L: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Resuelve L y = b
    con L triangular inferior y diagonal unitaria.
    """

    if L.shape != (2, 2):
        raise ValueError(f"Se esperaba L con shape (2,2), recibido {L.shape}")
    if b.shape != (2,):
        raise ValueError(f"Se esperaba b con shape (2,), recibido {b.shape}")

    y = np.zeros(2, dtype=np.complex128)

    y[0] = b[0]
    y[1] = b[1] - L[1, 0] * y[0]

    return y


def diagonal_subst(
    D: NDArray[np.complex128],
    y: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """
    Resuelve D z = y
    con D diagonal.
    """

    if D.shape != (2, 2):
        raise ValueError(f"Se esperaba D con shape (2,2), recibido {D.shape}")
    if y.shape != (2,):
        raise ValueError(f"Se esperaba y con shape (2,), recibido {y.shape}")

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


def backward_subst_ldlh(
    L: NDArray[np.complex128],
    z: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Resuelve L^H m = z
    con L triangular inferior y diagonal unitaria.
    """

    if L.shape != (2, 2):
        raise ValueError(f"Se esperaba L con shape (2,2), recibido {L.shape}")
    if z.shape != (2,):
        raise ValueError(f"Se esperaba z con shape (2,), recibido {z.shape}")

    m_hat = np.zeros(2, dtype=np.complex128)

    m_hat[1] = z[1]
    m_hat[0] = z[0] - np.conj(L[1, 0]) * m_hat[1]

    return m_hat


def cholesky_ldlh(
    Aij: NDArray[np.complex128],
    bij: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """
    Resuelve Aij m = bij usando LDL^H manual 2x2.

    Aij : (2,2)
    bij : (2,)
    Devuelve:
        m_hat : (2,)
    """
    L, D = compute_LD(Aij, eps=eps)
    y = forward_subst_ldlh(L, bij)
    z = diagonal_subst(D, y, eps=eps)
    m_hat = backward_subst_ldlh(L, z)
    return m_hat



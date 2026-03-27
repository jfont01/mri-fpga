import numpy as np, os, sys
from numpy.typing import NDArray
# ------------------------- ENVIRONMENT SET -------------------------
SENSE_FP_DIR = os.environ.get("SENSE_FP_DIR")
if SENSE_FP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, SENSE_FP_DIR)

from fp_compute_A           import fp_compute_A_ij
from fp_compute_b           import fp_compute_b_i
from fp_cholesky_LDLH       import fp_backward_subst_ldlh_i, fp_compute_LD_ij, fp_diagonal_subst_ldlh_i, fp_forward_subst_ldlh_i
from fp_compute_m_hat_ldlh  import fp_compute_m_hat_i_ldlh
from fp_cholesky_LLH        import fp_compute_m_hat_i_llh
# ------------------------------------------------------------------



def fp_compute_A_tensor(
    S: NDArray[np.complex128]
)-> NDArray[np.complex128]:
    
    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    A = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):                # recorre nx de [0, Nx - 1]
        for ny_alias in range(offset):  # recorre ny_alias de [0, Ny/2 - 1]
            Aij = fp_compute_A_ij(S, nx, ny_alias)  # (2,2)
            A[:, :, nx, ny_alias] = Aij
    return A

def fp_compute_b_tensor(
        S: NDArray[np.complex128],
        y: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    
    L, Nx, Ny = S.shape
    Af = 2
    offset = Ny // Af
    b = np.zeros((2, Nx, offset), dtype=np.complex128)


    for nx in range(Nx):                # recorre nx de [0, Nx - 1]
        for ny_alias in range(offset):  # recorre ny_alias de [0, Ny/2 - 1]
            bi = fp_compute_b_i(S, y, nx, ny_alias)  # (2,1)
            b[:, nx, ny_alias] = bi
    return b

def fp_compute_LD_tensor(
    A: NDArray[np.complex128],
    eps: float = 1e-12,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    
    _, _, Nx, offset = A.shape

    L = np.zeros((2, 2, Nx, offset), dtype=np.complex128)
    D = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            Lij, Dij = fp_compute_LD_ij(A[:, :, nx, ny_alias], eps=eps)
            L[:, :, nx, ny_alias] = Lij
            D[:, :, nx, ny_alias] = Dij

    return L, D


def fp_forward_subst_ldlh_tensor(
    L: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    _, _, Nx, offset = L.shape
    y = np.zeros((2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            y[:, nx, ny_alias] = fp_forward_subst_ldlh_i(
                L[:, :, nx, ny_alias],
                b[:, nx, ny_alias],
            )

    return y


def fp_diagonal_subst_ldlh_tensor(
    D: NDArray[np.complex128],
    y: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:

    _, _, Nx, offset = D.shape
    z = np.zeros((2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            z[:, nx, ny_alias] = fp_diagonal_subst_ldlh_i(
                D[:, :, nx, ny_alias],
                y[:, nx, ny_alias],
                eps=eps,
            )

    return z


def fp_backward_subst_ldlh_tensor(
    L: NDArray[np.complex128],
    z: NDArray[np.complex128],
) -> NDArray[np.complex128]:

    _, _, Nx, offset = L.shape
    m_hat = np.zeros((2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            m_hat[:, nx, ny_alias] = fp_backward_subst_ldlh_i(
                L[:, :, nx, ny_alias],
                z[:, nx, ny_alias],
            )

    return m_hat



def fp_compute_m_hat_tensor(A, b, compute_type, cholesky_type=None):

    _, _, Nx, offset = A.shape

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
                    m_hat[:, nx, ny_alias] = fp_compute_m_hat_i_llh(Aij, bi)
                elif cholesky_type == "LDLH":
                    m_hat[:, nx, ny_alias] = fp_compute_m_hat_i_ldlh(Aij, bi)
                else:
                    raise ValueError(f"cholesky_type inválido: {cholesky_type}")

            else:
                raise ValueError(f"compute_type inválido: {compute_type}")

    return m_hat

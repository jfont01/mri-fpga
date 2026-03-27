import numpy as np
from numpy.typing import NDArray

from fp_cholesky_LDLH import fp_backward_subst_ldlh_i, fp_compute_LD_ij, fp_diagonal_subst_ldlh_i, fp_forward_subst_ldlh_i


def fp_compute_m_hat_i_ldlh(
    Aij: NDArray[np.complex128],
    bij: NDArray[np.complex128],
    eps: float = 1e-12,
) -> NDArray[np.complex128]:

    Lij, Dij = fp_compute_LD_ij(Aij, eps=eps)
    yi = fp_forward_subst_ldlh_i(Lij, bij)
    zi = fp_diagonal_subst_ldlh_i(Dij, yi, eps=eps)
    m_hati = fp_backward_subst_ldlh_i(Lij, zi)
    return m_hati







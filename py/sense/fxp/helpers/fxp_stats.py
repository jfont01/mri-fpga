from cfxp import CFxp
from fxp import Fxp
from cfxptensor import CFxpTensor
from typing import Dict, Any
import numpy as np

def update_acc_stats(stats: dict, acc_name: str, z: CFxp) -> None:
    if "accumulators" not in stats:
        stats["accumulators"] = {}

    if acc_name not in stats["accumulators"]:
        stats["accumulators"][acc_name] = {
            "NB": z.re.NB,
            "NBF": z.re.NBF,
            "signed": z.re.signed,
            "min_re": float("inf"),
            "max_re": float("-inf"),
            "min_im": float("inf"),
            "max_im": float("-inf")
        }

    d = stats["accumulators"][acc_name]

    zr = float(z.re.get_val())
    zi = float(z.im.get_val())

    d["min_re"] = min(d["min_re"], zr)
    d["max_re"] = max(d["max_re"], zr)
    d["min_im"] = min(d["min_im"], zi)
    d["max_im"] = max(d["max_im"], zi)


def _get_all_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    stats.update(Fxp.get_fxp_stats())
    return stats


def _sum_stats(total: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in part.items():
        if k == "accumulators":
            if "accumulators" not in total:
                total["accumulators"] = {}

            for acc_name, acc_data in v.items():
                if acc_name not in total["accumulators"]:
                    total["accumulators"][acc_name] = dict(acc_data)
                else:
                    tgt = total["accumulators"][acc_name]
                    tgt["min_re"] = min(tgt["min_re"], acc_data["min_re"])
                    tgt["max_re"] = max(tgt["max_re"], acc_data["max_re"])
                    tgt["min_im"] = min(tgt["min_im"], acc_data["min_im"])
                    tgt["max_im"] = max(tgt["max_im"], acc_data["max_im"])
        else:
            total[k] = total.get(k, 0) + int(v)

    return total

def A_structure_metrics(
    A_q: CFxpTensor | np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """
    Métricas estructurales para A con shape (2,2,Nx,offset).

    Reporta:
    - mínimo de diagonal real
    - min/max de det(A)
    - cantidad de bloques con det(A) <= 0
    - pivotes LDL^H:
        d0 = real(A00)
        d1 = real(A11) - |A10|^2 / d0
    """

    if isinstance(A_q, CFxpTensor):
        A = A_q.to_complex_ndarray()
    else:
        A = np.asarray(A_q, dtype=np.complex128)

    if A.ndim != 4 or A.shape[0:2] != (2, 2):
        raise ValueError(f"A debe tener shape (2,2,Nx,offset), recibió {A.shape}")

    _, _, Nx, offset = A.shape
    nblocks = Nx * offset

    min_real_A00 = float("inf")
    min_real_A11 = float("inf")

    min_det_A = float("inf")
    max_det_A = float("-inf")
    max_abs_imag_det_A = 0.0

    min_d0 = float("inf")
    min_d1 = float("inf")

    count_det_le_zero = 0
    count_d0_le_zero = 0
    count_d1_le_zero = 0
    count_d0_le_eps = 0
    count_d1_le_eps = 0

    worst_det_idx = None
    worst_d0_idx = None
    worst_d1_idx = None

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]

            a00 = Aij[0, 0]
            a01 = Aij[0, 1]
            a10 = Aij[1, 0]
            a11 = Aij[1, 1]

            a00r = float(np.real(a00))
            a11r = float(np.real(a11))

            # Para Hermitiana ideal, det(A) es real
            detA = a00 * a11 - a01 * a10
            detA_real = float(np.real(detA))
            detA_imag = float(abs(np.imag(detA)))

            d0 = a00r
            if d0 > 0.0:
                d1 = a11r - (abs(a10) ** 2) / d0
            else:
                d1 = float("-inf")

            # mínimos
            if a00r < min_real_A00:
                min_real_A00 = a00r

            if a11r < min_real_A11:
                min_real_A11 = a11r

            if detA_real < min_det_A:
                min_det_A = detA_real
                worst_det_idx = (nx, ny_alias)

            if detA_real > max_det_A:
                max_det_A = detA_real

            if detA_imag > max_abs_imag_det_A:
                max_abs_imag_det_A = detA_imag

            if d0 < min_d0:
                min_d0 = d0
                worst_d0_idx = (nx, ny_alias)

            if d1 < min_d1:
                min_d1 = d1
                worst_d1_idx = (nx, ny_alias)

            # conteos
            if detA_real <= 0.0:
                count_det_le_zero += 1

            if d0 <= 0.0:
                count_d0_le_zero += 1
            if d1 <= 0.0:
                count_d1_le_zero += 1

            if d0 <= eps:
                count_d0_le_eps += 1
            if d1 <= eps:
                count_d1_le_eps += 1

    return {
        "min_real_A00": min_real_A00,
        "min_real_A11": min_real_A11,
        "min_det_A": min_det_A,
        "max_det_A": max_det_A,
        "count_det_le_zero": count_det_le_zero,
        "count_d0_le_zero": count_d0_le_zero,
        "count_d1_le_zero": count_d1_le_zero,

    }

def hermitian_error_metrics_A(A_q: CFxpTensor) -> dict:
    if A_q.ndim != 4 or A_q.shape[0:2] != (2, 2):
        raise ValueError(f"A debe tener shape (2,2,Nx,offset), recibió {A_q.shape}")

    A = A_q.to_complex_ndarray()
    _, _, Nx, offset = A.shape

    max_abs_imag_A00 = 0.0
    max_abs_imag_A11 = 0.0
    max_abs_offdiag_err = 0.0
    mean_abs_offdiag_err_acc = 0.0
    max_fro_hermitian_err = 0.0
    mean_fro_hermitian_err_acc = 0.0

    worst_offdiag_idx = None
    worst_fro_idx = None

    nblocks = Nx * offset

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]

            imag_a00 = float(abs(np.imag(Aij[0, 0])))
            imag_a11 = float(abs(np.imag(Aij[1, 1])))
            offdiag_err = float(abs(Aij[1, 0] - np.conj(Aij[0, 1])))

            EH = Aij - Aij.conj().T
            fro_err = float(np.linalg.norm(EH))

            if imag_a00 > max_abs_imag_A00:
                max_abs_imag_A00 = imag_a00

            if imag_a11 > max_abs_imag_A11:
                max_abs_imag_A11 = imag_a11

            mean_abs_offdiag_err_acc += offdiag_err
            mean_fro_hermitian_err_acc += fro_err

            if offdiag_err > max_abs_offdiag_err:
                max_abs_offdiag_err = offdiag_err
                worst_offdiag_idx = (nx, ny_alias)

            if fro_err > max_fro_hermitian_err:
                max_fro_hermitian_err = fro_err
                worst_fro_idx = (nx, ny_alias)

    return {
        "max_abs_imag_A00": max_abs_imag_A00,
        "max_abs_imag_A11": max_abs_imag_A11,
        "max_abs_hermitian_offdiag_err": max_abs_offdiag_err
    }
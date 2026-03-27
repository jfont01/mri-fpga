import numpy as np
import os
import sys
from typing import Tuple, Dict

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import update_acc_stats
# ------------------------------------------------------------------


def fxp_compute_D_i(
    Aij_q: CFxpTensor,
    stats_D: Dict | None = None,
    eps: float = 1e-12,
) -> CFxpTensor:

    NB = Aij_q.NB
    NBF = Aij_q.NBF
    signed = Aij_q.signed

    a00 = Aij_q[0, 0]
    a10 = Aij_q[1, 0]
    a11 = Aij_q[1, 1]

    zero = Fxp.quantize(0.0, NB, NBF, signed=signed)

    # d0 = real(a00)
    d0 = a00.re.cast(NB, NBF, mode="round")
    d0_f = float(d0.get_val())
    if d0_f <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD en fixed: d0={d0_f} no es estrictamente positivo"
        )

    # |a10|^2 / d0
    abs_a10_sq = (a10.re * a10.re + a10.im * a10.im).cast(NB, NBF, mode="round")
    quot = Fxp.div(abs_a10_sq, d0, NB_out=NB, NBF_out=NBF, mode="round")

    # d1 = real(a11) - |a10|^2 / d0
    d1 = (a11.re - quot).cast(NB, NBF, mode="round")
    d1_f = float(d1.get_val())
    if d1_f <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD en fixed o está mal condicionada: d1={d1_f}"
        )

    if stats_D is not None:
        update_acc_stats(stats_D, "D00", CFxp(d0, zero))
        update_acc_stats(stats_D, "D11", CFxp(d1, zero))

    Dij_q = CFxpTensor.zeros(
        shape=(2, 2),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    Dij_q[0, 0] = CFxp(d0, zero)
    Dij_q[0, 1] = CFxp(zero, zero)
    Dij_q[1, 0] = CFxp(zero, zero)
    Dij_q[1, 1] = CFxp(d1, zero)

    return Dij_q
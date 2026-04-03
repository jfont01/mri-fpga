import os
import sys
from typing import Dict
import numpy as np

# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))
from fxp_stats import update_acc_stats
# ------------------------------------------------------------------


def fxp_compute_z_i(
    Dij_q: CFxpTensor,
    xi_q: CFxpTensor,
    stats_z: Dict | None = None,
    eps: float = 1e-12,
) -> CFxpTensor:

    NB = xi_q.NB
    NBF = xi_q.NBF
    signed = xi_q.signed

    d0 = Dij_q[0, 0].re.cast(NB, NBF, mode="round")
    d1 = Dij_q[1, 1].re.cast(NB, NBF, mode="round")

    d0_f = float(d0.get_val())
    d1_f = float(d1.get_val())

    if d0_f <= eps or d1_f <= eps:
        raise np.linalg.LinAlgError(
            f"D no es invertible o no es positiva: d0={d0_f}, d1={d1_f}"
        )

    z0 = CFxp.div_by_real(xi_q[0], d0, NB_out=NB, NBF_out=NBF, mode="round")
    z1 = CFxp.div_by_real(xi_q[1], d1, NB_out=NB, NBF_out=NBF, mode="round")

    if stats_z is not None:
        update_acc_stats(stats_z, "Z0", z0)
        update_acc_stats(stats_z, "Z1", z1)

    zi_q = CFxpTensor.zeros(
        shape=(2,),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    zi_q[0] = z0
    zi_q[1] = z1

    return zi_q
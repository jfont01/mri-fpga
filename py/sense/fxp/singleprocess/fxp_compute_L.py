import numpy as np
import os
import sys
from typing import Tuple, Dict

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


def fxp_compute_L_i(
    Aij: CFxpTensor,
    Dij: CFxpTensor,
    stats_L: Dict | None = None,
    eps: float = 1e-12,
) -> CFxpTensor:

    NB = Aij.NB
    NBF = Aij.NBF
    signed = Aij.signed

    a10 = Aij[1, 0]
    d0_c = Dij[0, 0]
    d0 = d0_c.re.cast(NB, NBF, mode="round")

    d0_f = float(d0.get_val())
    if d0_f <= eps:
        raise np.linalg.LinAlgError(
            f"D no es válido para construir L: d0={d0_f} no es estrictamente positivo"
        )

    zero = Fxp.quantize(0.0, NB, NBF, signed=signed)
    one = Fxp.quantize(1.0, NB, NBF, signed=signed)

    l10 = CFxp.div_by_real(a10, d0, NB_out=NB, NBF_out=NBF, mode="round")

    if stats_L is not None:
        update_acc_stats(stats_L, "L10", l10)

    Lij = CFxpTensor.zeros(
        shape=(2, 2),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    Lij[0, 0] = CFxp(one, zero)
    Lij[0, 1] = CFxp(zero, zero)
    Lij[1, 0] = l10.cast(NB, NBF, mode="round")
    Lij[1, 1] = CFxp(one, zero)

    return Lij
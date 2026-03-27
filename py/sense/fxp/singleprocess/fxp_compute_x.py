import os
import sys
from typing import Dict
import numpy as np

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


def fxp_compute_x_i(
    Lij_q: CFxpTensor,
    bi_q: CFxpTensor,
    stats_x: Dict | None = None,
) -> CFxpTensor:

    NB = bi_q.NB
    NBF = bi_q.NBF
    signed = bi_q.signed

    b0 = bi_q[0]
    b1 = bi_q[1]
    l10 = Lij_q[1, 0]

    x0 = b0.cast(NB, NBF, mode="round")
    prod = (l10 * x0).cast(NB, NBF, mode="round")
    x1 = (b1 - prod).cast(NB, NBF, mode="round")

    if stats_x is not None:
        update_acc_stats(stats_x, "X0", x0)
        update_acc_stats(stats_x, "X1", x1)

    xi_q = CFxpTensor.zeros(
        shape=(2,),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    xi_q[0] = x0
    xi_q[1] = x1

    return xi_q
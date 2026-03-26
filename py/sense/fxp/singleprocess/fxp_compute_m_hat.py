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


def fxp_compute_m_hat_i(
    Lij_q: CFxpTensor,
    zi_q: CFxpTensor,
    stats_m_hat: Dict | None = None,
) -> CFxpTensor:
    """
    Resuelve L^H m = z para un bloque local 2x2.

    L = [[1,   0],
         [l10, 1]]

    L^H = [[1, conj(l10)],
           [0, 1]]

    m1 = z1
    m0 = z0 - conj(l10)*m1
    """

    if Lij_q.shape != (2, 2):
        raise ValueError(f"Lij_q debe tener shape (2,2), recibió {Lij_q.shape}")

    if zi_q.shape != (2,):
        raise ValueError(f"zi_q debe tener shape (2,), recibió {zi_q.shape}")

    NB = zi_q.NB
    NBF = zi_q.NBF
    signed = zi_q.signed

    l10h = Lij_q[1, 0].conj().cast(NB, NBF, mode="round")

    m1 = zi_q[1].cast(NB, NBF, mode="round")
    prod = (l10h * m1).cast(NB, NBF, mode="round")
    m0 = (zi_q[0] - prod).cast(NB, NBF, mode="round")

    if stats_m_hat is not None:
        update_acc_stats(stats_m_hat, "M0", m0)
        update_acc_stats(stats_m_hat, "M1", m1)

    mi_q = CFxpTensor.zeros(
        shape=(2,),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    mi_q[0] = m0
    mi_q[1] = m1

    return mi_q
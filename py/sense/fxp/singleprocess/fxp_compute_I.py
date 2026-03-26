import os
import sys
from typing import Dict
import numpy as np

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import update_acc_stats
# ------------------------------------------------------------------


def fxp_compute_I(
    m_hat_q: CFxpTensor,
    stats_I: Dict | None = None,
) -> CFxpTensor:


    _, Nx, offset = m_hat_q.shape


    NB = m_hat_q.NB
    NBF = m_hat_q.NBF
    signed = m_hat_q.signed

    Af = 2
    Ny = offset * Af

    I_q = CFxpTensor.zeros(
        shape=(Nx, Ny),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    for nx in range(Nx):
        for ny_alias in range(offset):
            m0 = m_hat_q[0, nx, ny_alias]
            m1 = m_hat_q[1, nx, ny_alias]

            ny0 = ny_alias
            ny1 = ny_alias + offset

            I_q[nx, ny0] = m0
            I_q[nx, ny1] = m1

            if stats_I is not None:
                update_acc_stats(stats_I, "I", m0)
                update_acc_stats(stats_I, "I", m1)

    return I_q
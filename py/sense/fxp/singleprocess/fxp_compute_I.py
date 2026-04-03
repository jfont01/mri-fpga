import os
import sys
from typing import Dict
import numpy as np

# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))
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
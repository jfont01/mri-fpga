import numpy as np
import os, sys
from numpy.lib.npyio import NpzFile

# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxp import CFxp
from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))

from fxp_stats import update_acc_stats
# ------------------------------------------------------------------



def fxp_compute_b_i(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    NB_B: int,
    NBF_B: int,
    nx: int,
    ny_alias: int,
    stats: dict | None = None
) -> CFxpTensor:

    signed = S_q.signed

    L, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af

    b0 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B)
    b1 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B)

    ny0 = ny_alias
    ny1 = ny_alias + offset

    for l in range(L):
        s0 = S_q[l, nx, ny0]
        s1 = S_q[l, nx, ny1]
        y0 = y_q[l, nx, ny_alias]

        p0 = (s0.conj() * y0).cast(NB_B, NBF_B, mode="round")
        p1 = (s1.conj() * y0).cast(NB_B, NBF_B, mode="round")

        b0 = (b0 + p0).cast(NB_B, NBF_B, mode="round")
        b1 = (b1 + p1).cast(NB_B, NBF_B, mode="round")


        update_acc_stats(stats, "b0", b0)
        update_acc_stats(stats, "b1", b1)

    bi = CFxpTensor.zeros(
        shape=(2,),
        NB=NB_B,
        NBF=NBF_B,
        signed=signed,
    )

    bi[0] = b0
    bi[1] = b1
    
    return bi

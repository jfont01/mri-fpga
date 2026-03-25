import numpy as np
import os, sys
from numpy.lib.npyio import NpzFile

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxp import CFxp
from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))

from fxp_stats import update_acc_stats
# ------------------------------------------------------------------



def fxp_compute_b_i(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    nx: int,
    ny_alias: int,
    stats: dict | None = None
) -> CFxpTensor:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed

    NB_Y = y_q.NB
    NBF_Y = y_q.NBF


    L, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L)))
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y
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


def fxp_compute_b(
        S_q: CFxpTensor,
        y_q: CFxpTensor
    ) -> CFxpTensor:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed

    NB_Y = y_q.NB
    NBF_Y = y_q.NBF


    L, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L)))
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y

    b = CFxpTensor.zeros(
        shape=(2, Nx, offset),
        NB=NB_B,
        NBF=NBF_B,
        signed=signed,
    )


    for nx in range(Nx):
        for ny_alias in range(offset):
            bi_fxp = fxp_compute_b_i(S_q, y_q, nx, ny_alias)

            b[0, nx, ny_alias] = bi_fxp[0]
            b[1, nx, ny_alias] = bi_fxp[1]

    return b
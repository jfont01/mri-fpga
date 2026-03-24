import numpy as np
import os, sys
from numpy.lib.npyio import NpzFile

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
from cfxptensor import CFxpTensor
# ------------------------------------------------------------------



def fxp_compute_b_ij(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    nx: int,
    ny_alias: int
) -> np.ndarray:

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

        b0 += (s0.conj() * y0).cast(NB_B, NBF_B, mode="round")
        b1 += (s1.conj() * y0).cast(NB_B, NBF_B, mode="round")

    bi = np.array([b0, b1], dtype=CFxp)

    return bi


def fxp_compute_b(
        S_q: CFxpTensor,
        y_q: CFxpTensor
    ) -> np.ndarray:

    L, Nx, Ny = S_q.shape
    Af = 2


    offset = Ny // Af

    b = np.zeros((2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            print(f"[{nx},{ny_alias}]")

            bi_fxp = fxp_compute_b_ij(S_q, y_q, nx, ny_alias)
            b[:, nx, ny_alias] = np.array([bi_fxp[0].to_complex(),
                                           bi_fxp[1].to_complex()], dtype=np.complex128)

    return b
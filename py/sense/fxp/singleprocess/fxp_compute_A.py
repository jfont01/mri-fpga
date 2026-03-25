import numpy as np, os, sys
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

def fxp_compute_A_ij(
    S_q: CFxpTensor,
    nx: int,
    ny_alias: int,
    stats: dict | None = None,
) -> CFxpTensor:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed
    L, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L)))
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    
    ny0 = ny_alias  
    ny1 = ny_alias + offset

    A00 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A)
    A11 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A)
    A01 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A)
    zero = Fxp.quantize(0.0, NB_A, NBF_A, signed=signed)
    
    for l in range(L):
        s0 = S_q[l, nx, ny0]
        s1 = S_q[l, nx, ny1]

        p00 = CFxp((s0.re * s0.re + s0.im * s0.im), zero).cast(NB_A, NBF_A, mode="round")
        p11 = CFxp((s1.re * s1.re + s1.im * s1.im), zero).cast(NB_A, NBF_A, mode="round")
        p01 = (s0.conj() * s1).cast(NB_A, NBF_A, mode="round")

        A00 = (A00 + p00).cast(NB_A, NBF_A, mode="round")
        A11 = (A11 + p11).cast(NB_A, NBF_A, mode="round")
        A01 = (A01 + p01).cast(NB_A, NBF_A, mode="round")

        update_acc_stats(stats, "A00", A00)
        update_acc_stats(stats, "A11", A11)
        update_acc_stats(stats, "A01", A01)

    A10 = (A01.conj()).cast(NB_A, NBF_A, mode="round")


    Aij = CFxpTensor.zeros(
        shape=(2, 2),
        NB=NB_A,
        NBF=NBF_A,
        signed=signed,
    )

    Aij[0, 0] = A00
    Aij[0, 1] = A01
    Aij[1, 0] = A10
    Aij[1, 1] = A11




    return Aij



def fxp_compute_A(
    S_q: CFxpTensor
) -> CFxpTensor:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed
    L, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L)))
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    A = CFxpTensor.zeros(
        shape=(2, 2, Nx, offset),
        NB=NB_A,
        NBF=NBF_A,
        signed=signed,
    )

    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = fxp_compute_A_ij(S_q, nx, ny_alias)

            A[0, 0, nx, ny_alias] = Aij[0, 0]
            A[0, 1, nx, ny_alias] = Aij[0, 1]
            A[1, 0, nx, ny_alias] = Aij[1, 0]
            A[1, 1, nx, ny_alias] = Aij[1, 1]

    return A






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

SENSE_FP_DIR = os.environ.get("SENSE_FP_DIR")
if SENSE_FP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FP_DIR not defined")

sys.path.insert(0, SENSE_FP_DIR)

from compute_A import compute_A_ij
# ------------------------------------------------------------------



def compute_A_ij_fixed(
    S_q: NpzFile,
    nx: int,
    ny_alias: int
) -> np.ndarray:

    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]
    NB_S = int(S_q["NB"])
    NBF_S = int(S_q["NBF"])
    signed = True if (S_q["signed"]==1) else False


    print(re_raw.shape)

    L, Nx, Ny = re_raw.shape
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
    A10 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A)
    zero = Fxp.quantize(0.0, NB_A, NBF_A)

    for l in range(L):
        s0 = CFxp.from_uint_pair(
            re_raw[l, nx, ny0],
            im_raw[l, nx, ny0],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )

        s1 = CFxp.from_uint_pair(
            re_raw[l, nx, ny1],
            im_raw[l, nx, ny1],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )


        A00 += CFxp((s0.re*s0.re + s0.im*s0.im), zero)
        A11 += CFxp((s1.re*s1.re + s1.im*s1.im), zero)
        A01 += s0.conj() * s1
    A10 = A01.conj()


    A = np.array([
        [A00, A01],
        [A10, A11],
    ], dtype=CFxp)

    return A


def compare_A_ij(
    S_f: np.ndarray,
    S_q: NpzFile,
    nx: int,
    ny_alias: int
):
    A_ref = compute_A_ij(S_f, nx, ny_alias)
    A_fix = compute_A_ij_fixed(S_q, nx, ny_alias)
    A_fix_np = np.array([
        [A_fix[0,0].to_complex(), A_fix[0,1].to_complex()],
        [A_fix[1,0].to_complex(), A_fix[1,1].to_complex()],
    ], dtype=np.complex128)


    diff = A_ref - A_fix_np

    return {
        "A_ref": A_ref,
        "A_fix": A_fix_np,
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_abs_err": float(np.mean(np.abs(diff)))
    }


if __name__ == "__main__":

    S_f = np.load("S.npy")

    S_q = np.load("S.npz")

    NB = int(S_q["NB"])
    NBF = int(S_q["NBF"])
    signed = bool(int(S_q["signed"]))

    nx = 0
    ny_alias = 0

    data = compare_A_ij(S_f, S_q, nx, ny_alias)

    print("A_ref =\n", data["A_ref"])
    print("A_fix =\n", data["A_fix"])
    print("max_abs_err =", data["max_abs_err"])
    print("mean_abs_err =", data["mean_abs_err"])

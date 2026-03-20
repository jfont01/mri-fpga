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
    ny_alias: int,
    NB: int,
    NBF: int,
    signed: bool = True,
    quant_mode: str = "round",
    cast_mode: str = "trunc",
) -> np.ndarray:

    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]

    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    if re_raw.ndim != 3:
        raise ValueError(f"re_raw debe ser 3D, recibió shape={re_raw.shape}")

    L, Nx, Ny = re_raw.shape

    Af = 2
    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af

    if not (0 <= nx < Nx):
        raise IndexError(f"nx fuera de rango: nx={nx}, Nx={Nx}")

    if not (0 <= ny_alias < offset):
        raise IndexError(f"ny_alias fuera de rango: ny_alias={ny_alias}, offset={offset}")

    ny0 = ny_alias
    ny1 = ny_alias + offset

    prod_NB = 2 * NB
    prod_NBF = 2 * NBF
    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    acc_NB = prod_NB + grow_bits
    acc_NBF = prod_NBF

    A00_acc = CFxp.from_complex(0.0 + 0.0j, NB=acc_NB, NBF=acc_NBF,
                                mode=quant_mode, signed=signed)
    A11_acc = CFxp.from_complex(0.0 + 0.0j, NB=acc_NB, NBF=acc_NBF,
                                mode=quant_mode, signed=signed)
    A01_acc = CFxp.from_complex(0.0 + 0.0j, NB=acc_NB, NBF=acc_NBF,
                                mode=quant_mode, signed=signed)

    for l in range(L):
        s0 = CFxp.from_uint_pair(
            re_raw[l, nx, ny0],
            im_raw[l, nx, ny0],
            NB=NB,
            NBF=NBF,
            signed=signed
        )

        s1 = CFxp.from_uint_pair(
            re_raw[l, nx, ny1],
            im_raw[l, nx, ny1],
            NB=NB,
            NBF=NBF,
            signed=signed
        )

        p00 = s0.conj() * s0
        p11 = s1.conj() * s1
        p01 = s0.conj() * s1

        p00_acc = p00.cast(acc_NB, acc_NBF, mode=cast_mode)
        p11_acc = p11.cast(acc_NB, acc_NBF, mode=cast_mode)
        p01_acc = p01.cast(acc_NB, acc_NBF, mode=cast_mode)

        A00_acc = A00_acc + p00_acc
        A11_acc = A11_acc + p11_acc
        A01_acc = A01_acc + p01_acc

    A10_acc = A01_acc.conj()

    A = np.array([
        [A00_acc.to_complex(), A01_acc.to_complex()],
        [A10_acc.to_complex(), A11_acc.to_complex()],
    ], dtype=np.complex128)

    return A


def compare_A_ij(
    S_f: np.ndarray,
    S_q: NpzFile,
    nx: int,
    ny_alias: int,
    NB: int,
    NBF: int,
    signed: bool = True,
):
    A_ref = compute_A_ij(S_f, nx, ny_alias)
    A_fix = compute_A_ij_fixed(S_q, nx, ny_alias, NB, NBF, signed=signed)

    diff = A_ref - A_fix

    return {
        "A_ref": A_ref,
        "A_fix": A_fix,
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_abs_err": float(np.mean(np.abs(diff))),
        "fro_err": float(np.linalg.norm(diff)),
    }


if __name__ == "__main__":

    S_f = np.load("S.npy")

    S_q = np.load("S.npz")

    NB = int(S_q["NB"])
    NBF = int(S_q["NBF"])
    signed = bool(int(S_q["signed"]))

    nx = 0
    ny_alias = 0

    data = compare_A_ij(S_f, S_q, nx, ny_alias, NB, NBF, signed)

    print("A_ref =\n", data["A_ref"])
    print("A_fix =\n", data["A_fix"])
    print("max_abs_err =", data["max_abs_err"])
    print("mean_abs_err =", data["mean_abs_err"])
    print("fro_err =", data["fro_err"])
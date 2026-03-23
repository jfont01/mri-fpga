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
# ------------------------------------------------------------------



def fxp_compute_A_ij(
    S_q: NpzFile,
    nx: int,
    ny_alias: int
) -> np.ndarray:

    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]
    NB_S = int(S_q["NB"])
    NBF_S = int(S_q["NBF"])
    signed = True if (S_q["signed"]==1) else False


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



def fxp_compute_A(S_q: NpzFile) -> np.ndarray:
    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]

    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    if re_raw.ndim != 3:
        raise ValueError(f"re_raw debe ser 3D, recibió shape={re_raw.shape}")

    _, Nx, Ny = re_raw.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af

    A = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    for nx in range(Nx):
        for ny_alias in range(offset):
            if nx % 32 == 0 and ny_alias == 0:
                print(f"Procesando nx={nx}/{Nx-1}")
            Aij_fxp = fxp_compute_A_ij(S_q, nx, ny_alias)
            A[:, :, nx, ny_alias] = np.array([
                    [Aij_fxp[0, 0].to_complex(), Aij_fxp[0, 1].to_complex()],
                    [Aij_fxp[1, 0].to_complex(), Aij_fxp[1, 1].to_complex()],
                ], dtype=np.complex128)

    return A







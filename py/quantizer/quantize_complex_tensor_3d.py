import numpy as np
import os, sys
# ------------------------- ENVIROMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxp import CFxp
# -------------------------------------------------------------------


import numpy as np

def quantize_complex_tensor_3d(
    S: np.ndarray,
    NB: int,
    NBF: int,
    mode: str = "round",
    signed: bool = True,
):
    if S.ndim != 3:
        raise ValueError(f"S debe ser 3D, recibió shape={S.shape}")

    L, Nx, Ny = S.shape

    if NB <= 8:
        raw_dtype = np.uint8
    elif NB <= 16:
        raw_dtype = np.uint16
    elif NB <= 32:
        raw_dtype = np.uint32
    elif NB <= 64:
        raw_dtype = np.uint64
    else:
        raise ValueError(f"NB={NB} no soportado para almacenamiento raw en NumPy")

    re = np.zeros((L, Nx, Ny), dtype=raw_dtype)
    im = np.zeros((L, Nx, Ny), dtype=raw_dtype)

    for l in range(L):
        for nx in range(Nx):
            for ny in range(Ny):
                z_fx = CFxp.from_complex(
                    complex(S[l, nx, ny]),
                    NB=NB,
                    NBF=NBF,
                    mode=mode,
                    signed=signed
                )

                re_u, im_u = z_fx.to_uint()
                re[l, nx, ny] = re_u
                im[l, nx, ny] = im_u

    return re, im




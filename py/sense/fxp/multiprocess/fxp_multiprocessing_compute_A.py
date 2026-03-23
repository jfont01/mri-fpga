import os
import sys
import numpy as np
from numpy.lib.npyio import NpzFile
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variables globales en workers
# ------------------------------------------------------------------
_RE_RAW = None
_IM_RAW = None
_NB_S = None
_NBF_S = None
_SIGNED = None


def _init_worker_A(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    signed: bool,
) -> None:

    global _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED
    _RE_RAW = re_raw
    _IM_RAW = im_raw
    _NB_S = NB_S
    _NBF_S = NBF_S
    _SIGNED = signed


def _fxp_compute_A_ij(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    signed: bool,
    nx: int,
    ny_alias: int,
) -> np.ndarray:

    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    L, Nx, Ny = re_raw.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af
    ny0 = ny_alias
    ny1 = ny_alias + offset

    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    zero = Fxp.quantize(0.0, NB_A, NBF_A, mode="round", signed=signed)

    A00 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A11 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A01 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)

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

    Aij_np = np.array([
        [A00.to_complex(), A01.to_complex()],
        [A10.to_complex(), A11.to_complex()],
    ], dtype=np.complex128)

    return Aij_np


def _worker_compute_A_nx(nx: int) -> Tuple[int, np.ndarray]:

    global _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED

    L, Nx, Ny = _RE_RAW.shape
    Af = 2
    offset = Ny // Af

    A_nx = np.zeros((2, 2, offset), dtype=np.complex128)

    for ny_alias in range(offset):
        A_nx[:, :, ny_alias] = _fxp_compute_A_ij(
            _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED, nx, ny_alias
        )

    return nx, A_nx


def fxp_compute_A(
    S_q: NpzFile,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> np.ndarray:

    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]
    NB_S = int(S_q["NB"])
    NBF_S = int(S_q["NBF"])
    signed = bool(int(S_q["signed"]))

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

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_A,
        initargs=(re_raw, im_raw, NB_S, NBF_S, signed),
    ) as executor:

        for nx, A_nx in executor.map(_worker_compute_A_nx, range(Nx), chunksize=chunksize):
            A[:, :, nx, :] = A_nx

    return A
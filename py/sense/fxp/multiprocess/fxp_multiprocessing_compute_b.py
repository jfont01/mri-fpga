import os
import sys
import numpy as np
from numpy.lib.npyio import NpzFile
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any

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
_S_RE_RAW = None
_S_IM_RAW = None
_Y_RE_RAW = None
_Y_IM_RAW = None

_NB_S = None
_NBF_S = None
_NB_Y = None
_NBF_Y = None
_SIGNED = None


def _init_worker_b(
    S_re_raw: np.ndarray,
    S_im_raw: np.ndarray,
    y_re_raw: np.ndarray,
    y_im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    NB_Y: int,
    NBF_Y: int,
    signed: bool,
) -> None:
    global _S_RE_RAW, _S_IM_RAW, _Y_RE_RAW, _Y_IM_RAW
    global _NB_S, _NBF_S, _NB_Y, _NBF_Y, _SIGNED

    _S_RE_RAW = S_re_raw
    _S_IM_RAW = S_im_raw
    _Y_RE_RAW = y_re_raw
    _Y_IM_RAW = y_im_raw

    _NB_S = NB_S
    _NBF_S = NBF_S
    _NB_Y = NB_Y
    _NBF_Y = NBF_Y
    _SIGNED = signed



def _get_all_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}

    stats.update(Fxp.get_fxp_stats())

    return stats


def _sum_stats(total: Dict[str, int], part: Dict[str, int]) -> Dict[str, int]:
    for k, v in part.items():
        total[k] = total.get(k, 0) + int(v)
    return total


def _fxp_compute_b_ij(
    S_re_raw: np.ndarray,
    S_im_raw: np.ndarray,
    y_re_raw: np.ndarray,
    y_im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    NB_Y: int,
    NBF_Y: int,
    signed: bool,
    nx: int,
    ny_alias: int,
) -> np.ndarray:


    Ls, NxS, Ny = S_re_raw.shape
    Ly, NxY, offset = y_re_raw.shape


    Af = 2

    grow_bits = int(np.ceil(np.log2(Ls))) if Ls > 1 else 0
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y

    b0 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B, mode="round", signed=signed)
    b1 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B, mode="round", signed=signed)

    ny0 = ny_alias
    ny1 = ny_alias + offset

    for l in range(Ls):
        s0 = CFxp.from_uint_pair(
            S_re_raw[l, nx, ny0],
            S_im_raw[l, nx, ny0],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )

        s1 = CFxp.from_uint_pair(
            S_re_raw[l, nx, ny1],
            S_im_raw[l, nx, ny1],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )

        y0 = CFxp.from_uint_pair(
            y_re_raw[l, nx, ny_alias],
            y_im_raw[l, nx, ny_alias],
            NB=NB_Y,
            NBF=NBF_Y,
            signed=signed
        )

        p0 = (s0.conj() * y0).cast(NB_B, NBF_B, mode="round")
        p1 = (s1.conj() * y0).cast(NB_B, NBF_B, mode="round")

        b0 = b0 + p0
        b1 = b1 + p1

    bi_np = np.array(
        [b0.to_complex(), b1.to_complex()],
        dtype=np.complex128
    )

    return bi_np


def _worker_compute_b_nx(nx: int) -> Tuple[int, np.ndarray, Dict[str, int]]:

    global _S_RE_RAW, _S_IM_RAW, _Y_RE_RAW, _Y_IM_RAW
    global _NB_S, _NBF_S, _NB_Y, _NBF_Y, _SIGNED

    Fxp.reset_fxp_stats()

    _, _, Ny = _S_RE_RAW.shape
    Af = 2
    offset = Ny // Af

    b_nx = np.zeros((2, offset), dtype=np.complex128)

    for ny_alias in range(offset):
        b_nx[:, ny_alias] = _fxp_compute_b_ij(
            _S_RE_RAW, _S_IM_RAW,
            _Y_RE_RAW, _Y_IM_RAW,
            _NB_S, _NBF_S,
            _NB_Y, _NBF_Y,
            _SIGNED,
            nx, ny_alias
        )

    stats_nx = _get_all_stats()

    return nx, b_nx, stats_nx


def fxp_compute_b(
    S_q: NpzFile,
    y_q: NpzFile,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Dict[str, Any]:

    S_re_raw = S_q["re_raw"]
    S_im_raw = S_q["im_raw"]
    y_re_raw = y_q["re_raw"]
    y_im_raw = y_q["im_raw"]

    NB_S = int(S_q["NB"])
    NBF_S = int(S_q["NBF"])
    NB_Y = int(y_q["NB"])
    NBF_Y = int(y_q["NBF"])

    signed_S = bool(int(S_q["signed"]))
    signed_Y = bool(int(y_q["signed"]))

    signed = signed_S

    _, Nx, Ny = S_re_raw.shape
    Ly, NxY, offset = y_re_raw.shape

    Af = 2

    b = np.zeros((2, Nx, offset), dtype=np.complex128)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_total: Dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_b,
        initargs=(
            S_re_raw, S_im_raw,
            y_re_raw, y_im_raw,
            NB_S, NBF_S,
            NB_Y, NBF_Y,
            signed
        ),
    ) as executor:

        for nx, b_nx, stats_nx in executor.map(
            _worker_compute_b_nx,
            range(Nx),
            chunksize=chunksize
        ):
            b[:, nx, :] = b_nx
            _sum_stats(stats_total, stats_nx)

    return {
        "b": b,
        "stats": stats_total,
    }
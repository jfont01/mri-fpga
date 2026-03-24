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
from cfxptensor import  CFxpTensor
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variables globales en workers
# ------------------------------------------------------------------
_S_Q = None
_Y_Q = None


def _init_worker_b(
    S_q: CFxpTensor,
    y_q: CFxpTensor
) -> None:
    global _S_Q, _Y_Q
    _S_Q = S_q
    _Y_Q = y_q


def _get_all_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    stats.update(Fxp.get_fxp_stats())

    return stats


def _sum_stats(total: Dict[str, int], part: Dict[str, int]) -> Dict[str, int]:
    for k, v in part.items():
        total[k] = total.get(k, 0) + int(v)
    return total


def _fxp_compute_b_ij(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    nx: int,
    ny_alias: int
) -> np.ndarray:


    Ls, NxS, Ny = S_q.shape
    Ly, NxY, offset = y_q.shape

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed

    NB_Y = y_q.NB
    NBF_Y = y_q.NBF

    Af = 2

    grow_bits = int(np.ceil(np.log2(Ls))) if Ls > 1 else 0
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y

    b0 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B, mode="round", signed=signed)
    b1 = CFxp.from_complex(0.0 + 0.0j, NB_B, NBF_B, mode="round", signed=signed)

    ny0 = ny_alias
    ny1 = ny_alias + offset

    for l in range(Ls):
        s0 = S_q[l, nx, ny0]
        s1 = S_q[l, nx, ny1]

        y0 = y_q[l, nx, ny_alias]

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

    global _S_Q, _Y_Q

    Fxp.reset_fxp_stats()

    _, _, Ny = _S_Q.shape

    Af = 2
    offset = Ny // Af

    b_nx = np.zeros((2, offset), dtype=np.complex128)

    for ny_alias in range(offset):
        b_nx[:, ny_alias] = _fxp_compute_b_ij(_S_Q, _Y_Q, nx, ny_alias)

    stats_nx = _get_all_stats()

    return nx, b_nx, stats_nx


def fxp_compute_b(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Dict[str, Any]:

    Ls, Nx, Ny = S_q.shape
    Ly, NxY, offset = y_q.shape

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed

    NB_Y = y_q.NB
    NBF_Y = y_q.NBF

    Af = 2

    b = np.zeros((2, Nx, offset), dtype=np.complex128)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_total: Dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_b,
        initargs=(S_q, y_q),
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
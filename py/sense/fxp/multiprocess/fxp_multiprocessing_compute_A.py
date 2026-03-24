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
from cfxptensor import CFxpTensor
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variables globales en workers
# ------------------------------------------------------------------
_S_Q = None


def _init_worker_A(
    S_q: CFxpTensor,
) -> None:
    global _S_Q
    _S_Q = S_q

def _get_all_stats() -> Dict[str, int]:
    """
    Lee stats disponibles en Fxp y, si existe, también en CFxp.
    Devuelve un único dict plano.
    """
    stats: Dict[str, int] = {}

    if hasattr(Fxp, "get_fxp_stats"):
        stats.update(Fxp.get_fxp_stats())

    return stats


def _sum_stats(total: Dict[str, int], part: Dict[str, int]) -> Dict[str, int]:
    """
    Acumula stats parciales dentro del proceso padre.
    """
    for k, v in part.items():
        total[k] = total.get(k, 0) + int(v)
    return total


def _fxp_compute_A_ij(
    S_q: CFxpTensor,
    nx: int,
    ny_alias: int,
) -> np.ndarray:

    L, Nx, Ny = S_q.shape
    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed


    Af = 2

    offset = Ny // Af
    ny0 = ny_alias
    ny1 = ny_alias + offset

    grow_bits = int(np.ceil(np.log2(L)))
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    zero = Fxp.quantize(0.0, NB_A, NBF_A, mode="round", signed=signed)

    A00 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A11 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A01 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)

    for l in range(L):
        s0 = S_q[l, nx, ny0]
        s1 = S_q[l, nx, ny1]

        A00 += CFxp((s0.re * s0.re + s0.im * s0.im), zero)
        A11 += CFxp((s1.re * s1.re + s1.im * s1.im), zero)
        A01 += s0.conj() * s1

    A10 = A01.conj()

    Aij_np = np.array([
        [A00.to_complex(), A01.to_complex()],
        [A10.to_complex(), A11.to_complex()],
    ], dtype=np.complex128)

    return Aij_np


def _worker_compute_A_nx(nx: int) -> Tuple[int, np.ndarray, Dict[str, int]]:
    global _S_Q

    Fxp.reset_fxp_stats()

    _, _, Ny = _S_Q.shape
    Af = 2
    offset = Ny // Af

    A_nx = np.zeros((2, 2, offset), dtype=np.complex128)

    for ny_alias in range(offset):
        A_nx[:, :, ny_alias] = _fxp_compute_A_ij(
            _S_Q, nx, ny_alias
        )

    stats_nx = _get_all_stats()

    return nx, A_nx, stats_nx


def fxp_compute_A(
    S_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Dict[str, Any]:
    
    _, Nx, Ny = S_q.shape
    Af = 2

    offset = Ny // Af
    A = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_total: Dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_A,
        initargs=(S_q,),
    ) as executor:

        for nx, A_nx, stats_nx in executor.map(
            _worker_compute_A_nx,
            range(Nx),
            chunksize=chunksize
        ):
            A[:, :, nx, :] = A_nx
            _sum_stats(stats_total, stats_nx)

    return {
        "A": A,
        "stats": stats_total,
    }
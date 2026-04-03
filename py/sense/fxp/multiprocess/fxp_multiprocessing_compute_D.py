import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any

# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from fxp import Fxp
from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "singleprocess"))
from fxp_compute_D import fxp_compute_D_i

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variable global en workers
# ------------------------------------------------------------------
_A_Q = None


def _init_worker_D(A_q: CFxpTensor) -> None:
    global _A_Q
    _A_Q = A_q


def _worker_compute_D_nx(
    nx: int,
    eps: float,
) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _A_Q

    Fxp.reset_fxp_stats()

    NB = _A_Q.NB
    NBF = _A_Q.NBF
    signed = _A_Q.signed

    _, _, _, offset = _A_Q.shape

    D_nx = CFxpTensor.zeros(
        shape=(2, 2, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    stats_D_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Aij = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Aij[0, 0] = _A_Q[0, 0, nx, ny_alias]
        Aij[0, 1] = _A_Q[0, 1, nx, ny_alias]
        Aij[1, 0] = _A_Q[1, 0, nx, ny_alias]
        Aij[1, 1] = _A_Q[1, 1, nx, ny_alias]

        Dij = fxp_compute_D_i(
            Aij,
            stats_D=stats_D_nx,
            eps=eps,
        )

        D_nx[0, 0, ny_alias] = Dij[0, 0]
        D_nx[0, 1, ny_alias] = Dij[0, 1]
        D_nx[1, 0, ny_alias] = Dij[1, 0]
        D_nx[1, 1, ny_alias] = Dij[1, 1]

    low_level_stats = _get_all_stats()
    _sum_stats(stats_D_nx, low_level_stats)

    return nx, D_nx, stats_D_nx


def fxp_multiprocessing_compute_D(
    A_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
    eps: float = 1e-12,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    NB = A_q.NB
    NBF = A_q.NBF
    signed = A_q.signed

    _, _, Nx, offset = A_q.shape

    D_q = CFxpTensor.zeros(
        shape=(2, 2, Nx, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_D_total: Dict[str, Any] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_D,
        initargs=(A_q,),
    ) as executor:

        for nx, D_nx, stats_D_nx in executor.map(
            _worker_compute_D_nx,
            range(Nx),
            [eps] * Nx,
            chunksize=chunksize,
        ):
            for ny_alias in range(offset):
                D_q[0, 0, nx, ny_alias] = D_nx[0, 0, ny_alias]
                D_q[0, 1, nx, ny_alias] = D_nx[0, 1, ny_alias]
                D_q[1, 0, nx, ny_alias] = D_nx[1, 0, ny_alias]
                D_q[1, 1, nx, ny_alias] = D_nx[1, 1, ny_alias]

            _sum_stats(stats_D_total, stats_D_nx)

    return D_q, stats_D_total
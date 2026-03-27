import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "singleprocess"))
from fxp_compute_L import fxp_compute_L_i

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variables globales en workers
# ------------------------------------------------------------------
_A_Q = None
_D_Q = None


def _init_worker_L(
    A_q: CFxpTensor,
    D_q: CFxpTensor,
) -> None:
    global _A_Q, _D_Q
    _A_Q = A_q
    _D_Q = D_q


def _worker_compute_L_nx(
    nx: int,
    eps: float,
) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _A_Q, _D_Q

    Fxp.reset_fxp_stats()

    NB = _A_Q.NB
    NBF = _A_Q.NBF
    signed = _A_Q.signed

    _, _, _, offset = _A_Q.shape

    L_nx = CFxpTensor.zeros(
        shape=(2, 2, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    stats_L_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Aij = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Dij = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Aij[0, 0] = _A_Q[0, 0, nx, ny_alias]
        Aij[0, 1] = _A_Q[0, 1, nx, ny_alias]
        Aij[1, 0] = _A_Q[1, 0, nx, ny_alias]
        Aij[1, 1] = _A_Q[1, 1, nx, ny_alias]

        Dij[0, 0] = _D_Q[0, 0, nx, ny_alias]
        Dij[0, 1] = _D_Q[0, 1, nx, ny_alias]
        Dij[1, 0] = _D_Q[1, 0, nx, ny_alias]
        Dij[1, 1] = _D_Q[1, 1, nx, ny_alias]

        Lij = fxp_compute_L_i(
            Aij,
            Dij,
            stats_L=stats_L_nx,
            eps=eps,
        )

        L_nx[0, 0, ny_alias] = Lij[0, 0]
        L_nx[0, 1, ny_alias] = Lij[0, 1]
        L_nx[1, 0, ny_alias] = Lij[1, 0]
        L_nx[1, 1, ny_alias] = Lij[1, 1]

    low_level_stats = _get_all_stats()
    _sum_stats(stats_L_nx, low_level_stats)

    return nx, L_nx, stats_L_nx


def fxp_multiprocessing_compute_L(
    A_q: CFxpTensor,
    D_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
    eps: float = 1e-12,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    NB = A_q.NB
    NBF = A_q.NBF
    signed = A_q.signed

    _, _, Nx, offset = A_q.shape

    L_q = CFxpTensor.zeros(
        shape=(2, 2, Nx, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_L_total: Dict[str, Any] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_L,
        initargs=(A_q, D_q),
    ) as executor:

        for nx, L_nx, stats_L_nx in executor.map(
            _worker_compute_L_nx,
            range(Nx),
            [eps] * Nx,
            chunksize=chunksize,
        ):
            for ny_alias in range(offset):
                L_q[0, 0, nx, ny_alias] = L_nx[0, 0, ny_alias]
                L_q[0, 1, nx, ny_alias] = L_nx[0, 1, ny_alias]
                L_q[1, 0, nx, ny_alias] = L_nx[1, 0, ny_alias]
                L_q[1, 1, nx, ny_alias] = L_nx[1, 1, ny_alias]

            _sum_stats(stats_L_total, stats_L_nx)

    return L_q, stats_L_total
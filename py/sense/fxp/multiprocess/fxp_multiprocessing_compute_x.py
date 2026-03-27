import os
import sys
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
from fxp_compute_x import fxp_compute_x_i

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


_L_Q = None
_B_Q = None


def _init_worker_x(L_q: CFxpTensor, b_q: CFxpTensor) -> None:
    global _L_Q, _B_Q
    _L_Q = L_q
    _B_Q = b_q


def _worker_compute_x_nx(nx: int) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _L_Q, _B_Q

    Fxp.reset_fxp_stats()

    NB = _B_Q.NB
    NBF = _B_Q.NBF
    signed = _B_Q.signed

    _, _, _, offset = _L_Q.shape

    x_nx = CFxpTensor.zeros(
        shape=(2, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    stats_x_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Lij_q = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )
        bi_q = CFxpTensor.zeros(
            shape=(2,),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Lij_q[0, 0] = _L_Q[0, 0, nx, ny_alias]
        Lij_q[0, 1] = _L_Q[0, 1, nx, ny_alias]
        Lij_q[1, 0] = _L_Q[1, 0, nx, ny_alias]
        Lij_q[1, 1] = _L_Q[1, 1, nx, ny_alias]

        bi_q[0] = _B_Q[0, nx, ny_alias]
        bi_q[1] = _B_Q[1, nx, ny_alias]

        xi_q = fxp_compute_x_i(Lij_q, bi_q, stats_x=stats_x_nx)

        x_nx[0, ny_alias] = xi_q[0]
        x_nx[1, ny_alias] = xi_q[1]

    low_level_stats = _get_all_stats()
    _sum_stats(stats_x_nx, low_level_stats)

    return nx, x_nx, stats_x_nx


def fxp_multiprocessing_compute_x(
    L_q: CFxpTensor,
    b_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    NB = b_q.NB
    NBF = b_q.NBF
    signed = b_q.signed

    _, _, Nx, offset = L_q.shape
    
    x_q = CFxpTensor.zeros(
        shape=(2, Nx, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_x_total: Dict[str, Any] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_x,
        initargs=(L_q, b_q),
    ) as executor:

        for nx, x_nx, stats_x_nx in executor.map(
            _worker_compute_x_nx,
            range(Nx),
            chunksize=chunksize,
        ):
            for ny_alias in range(offset):
                x_q[0, nx, ny_alias] = x_nx[0, ny_alias]
                x_q[1, nx, ny_alias] = x_nx[1, ny_alias]

            _sum_stats(stats_x_total, stats_x_nx)

    return x_q, stats_x_total
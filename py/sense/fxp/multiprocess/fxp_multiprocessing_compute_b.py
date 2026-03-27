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

from fxp_compute_b import fxp_compute_b_i

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
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



def _worker_compute_b_nx(nx: int) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _S_Q, _Y_Q

    Fxp.reset_fxp_stats()

    NB_S = _S_Q.NB
    NBF_S = _S_Q.NBF
    NB_Y = _Y_Q.NB
    NBF_Y = _Y_Q.NBF
    signed = _S_Q.signed

    L, _, Ny = _S_Q.shape
    Ly, _, offset_y = _Y_Q.shape
    Af = 2

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y

    b_nx = CFxpTensor.zeros(
        shape=(2, offset),
        NB=NB_B,
        NBF=NBF_B,
        signed=signed,
    )

    # stats locales del worker
    stats_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        bi = fxp_compute_b_i(_S_Q, _Y_Q, nx, ny_alias, stats_nx)   # shape (2,)

        b_nx[0, ny_alias] = bi[0]
        b_nx[1, ny_alias] = bi[1]

    # fusionar contadores de bajo nivel de Fxp
    low_level_stats = _get_all_stats()
    _sum_stats(stats_nx, low_level_stats)

    return nx, b_nx, stats_nx


def fxp_multiprocessing_compute_b(
    S_q: CFxpTensor,
    y_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    NB_Y = y_q.NB
    NBF_Y = y_q.NBF
    signed = S_q.signed

    Ls, Nx, Ny = S_q.shape
    Ly, NxY, offset = y_q.shape
    Af = 2

    grow_bits = int(np.ceil(np.log2(Ls))) if Ls > 1 else 0
    NB_B = NB_S + NB_Y + grow_bits
    NBF_B = NBF_S + NBF_Y

    b = CFxpTensor.zeros(
        shape=(2, Nx, offset),
        NB=NB_B,
        NBF=NBF_B,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_total: Dict[str, Any] = {}

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
            for ny_alias in range(offset):
                b[0, nx, ny_alias] = b_nx[0, ny_alias]
                b[1, nx, ny_alias] = b_nx[1, ny_alias]

            _sum_stats(stats_total, stats_nx)

    return b, stats_total
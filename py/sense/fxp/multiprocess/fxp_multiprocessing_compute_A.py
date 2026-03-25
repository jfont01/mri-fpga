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
from fxp_compute_A import fxp_compute_A_ij



sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Variable global en workers
# ------------------------------------------------------------------
_S_Q = None


def _init_worker_A(S_q: CFxpTensor) -> None:
    global _S_Q
    _S_Q = S_q


def _worker_compute_A_nx(nx: int) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _S_Q

    Fxp.reset_fxp_stats()

    NB_S = _S_Q.NB
    NBF_S = _S_Q.NBF
    signed = _S_Q.signed
    L, _, Ny = _S_Q.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    A_nx = CFxpTensor.zeros(
        shape=(2, 2, offset),
        NB=NB_A,
        NBF=NBF_A,
        signed=signed,
    )

    # stats locales del worker
    stats_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Aij = fxp_compute_A_ij(_S_Q, nx, ny_alias, stats_nx)

        A_nx[0, 0, ny_alias] = Aij[0, 0]
        A_nx[0, 1, ny_alias] = Aij[0, 1]
        A_nx[1, 0, ny_alias] = Aij[1, 0]
        A_nx[1, 1, ny_alias] = Aij[1, 1]

    # fusionar contadores globales de Fxp
    low_level_stats = _get_all_stats()
    _sum_stats(stats_nx, low_level_stats)

    return nx, A_nx, stats_nx


def fxp_multiprocessing_compute_A(
    S_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    NB_S = S_q.NB
    NBF_S = S_q.NBF
    signed = S_q.signed
    L, Nx, Ny = S_q.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af

    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    A = CFxpTensor.zeros(
        shape=(2, 2, Nx, offset),
        NB=NB_A,
        NBF=NBF_A,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_total: Dict[str, Any] = {}

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
            for ny_alias in range(offset):
                A[0, 0, nx, ny_alias] = A_nx[0, 0, ny_alias]
                A[0, 1, nx, ny_alias] = A_nx[0, 1, ny_alias]
                A[1, 0, nx, ny_alias] = A_nx[1, 0, ny_alias]
                A[1, 1, nx, ny_alias] = A_nx[1, 1, ny_alias]

            _sum_stats(stats_total, stats_nx)

    return A, stats_total
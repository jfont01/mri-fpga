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
from fxp_compute_z import fxp_compute_z_i

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


_D_Q = None
_X_Q = None


def _init_worker_z(D_q: CFxpTensor, x_q: CFxpTensor) -> None:
    global _D_Q, _X_Q
    _D_Q = D_q
    _X_Q = x_q


def _worker_compute_z_nx(
    nx: int,
    eps: float,
) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _D_Q, _X_Q

    Fxp.reset_fxp_stats()

    NB = _X_Q.NB
    NBF = _X_Q.NBF
    signed = _X_Q.signed

    _, _, _, offset = _D_Q.shape

    z_nx = CFxpTensor.zeros(
        shape=(2, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    stats_z_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Dij_q = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )
        xi_q = CFxpTensor.zeros(
            shape=(2,),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Dij_q[0, 0] = _D_Q[0, 0, nx, ny_alias]
        Dij_q[0, 1] = _D_Q[0, 1, nx, ny_alias]
        Dij_q[1, 0] = _D_Q[1, 0, nx, ny_alias]
        Dij_q[1, 1] = _D_Q[1, 1, nx, ny_alias]

        xi_q[0] = _X_Q[0, nx, ny_alias]
        xi_q[1] = _X_Q[1, nx, ny_alias]

        zi_q = fxp_compute_z_i(Dij_q, xi_q, stats_z=stats_z_nx, eps=eps)

        z_nx[0, ny_alias] = zi_q[0]
        z_nx[1, ny_alias] = zi_q[1]

    low_level_stats = _get_all_stats()
    _sum_stats(stats_z_nx, low_level_stats)

    return nx, z_nx, stats_z_nx


def fxp_multiprocessing_compute_z(
    D_q: CFxpTensor,
    x_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
    eps: float = 1e-12,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    if D_q.ndim != 4 or D_q.shape[0:2] != (2, 2):
        raise ValueError(f"D_q debe tener shape (2,2,Nx,offset), recibió {D_q.shape}")

    if x_q.ndim != 3 or x_q.shape[0] != 2:
        raise ValueError(f"x_q debe tener shape (2,Nx,offset), recibió {x_q.shape}")

    NB = x_q.NB
    NBF = x_q.NBF
    signed = x_q.signed

    _, _, Nx, offset = D_q.shape
    if x_q.shape[1:] != (Nx, offset):
        raise ValueError("D_q y x_q deben tener la misma grilla (Nx,offset)")

    z_q = CFxpTensor.zeros(
        shape=(2, Nx, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_z_total: Dict[str, Any] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_z,
        initargs=(D_q, x_q),
    ) as executor:

        for nx, z_nx, stats_z_nx in executor.map(
            _worker_compute_z_nx,
            range(Nx),
            [eps] * Nx,
            chunksize=chunksize,
        ):
            for ny_alias in range(offset):
                z_q[0, nx, ny_alias] = z_nx[0, ny_alias]
                z_q[1, nx, ny_alias] = z_nx[1, ny_alias]

            _sum_stats(stats_z_total, stats_z_nx)

    return z_q, stats_z_total
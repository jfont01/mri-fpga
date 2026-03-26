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
from fxp_compute_m_hat import fxp_compute_m_hat_i

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import _get_all_stats, _sum_stats
# ------------------------------------------------------------------


_L_Q = None
_Z_Q = None


def _init_worker_m_hat(L_q: CFxpTensor, z_q: CFxpTensor) -> None:
    global _L_Q, _Z_Q
    _L_Q = L_q
    _Z_Q = z_q


def _worker_compute_m_hat_nx(nx: int) -> Tuple[int, CFxpTensor, Dict[str, Any]]:
    global _L_Q, _Z_Q

    Fxp.reset_fxp_stats()

    NB = _Z_Q.NB
    NBF = _Z_Q.NBF
    signed = _Z_Q.signed

    _, _, _, offset = _L_Q.shape

    m_hat_nx = CFxpTensor.zeros(
        shape=(2, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    stats_m_hat_nx: Dict[str, Any] = {}

    for ny_alias in range(offset):
        Lij_q = CFxpTensor.zeros(
            shape=(2, 2),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )
        zi_q = CFxpTensor.zeros(
            shape=(2,),
            NB=NB,
            NBF=NBF,
            signed=signed,
        )

        Lij_q[0, 0] = _L_Q[0, 0, nx, ny_alias]
        Lij_q[0, 1] = _L_Q[0, 1, nx, ny_alias]
        Lij_q[1, 0] = _L_Q[1, 0, nx, ny_alias]
        Lij_q[1, 1] = _L_Q[1, 1, nx, ny_alias]

        zi_q[0] = _Z_Q[0, nx, ny_alias]
        zi_q[1] = _Z_Q[1, nx, ny_alias]

        mi_q = fxp_compute_m_hat_i(Lij_q, zi_q, stats_m_hat=stats_m_hat_nx)

        m_hat_nx[0, ny_alias] = mi_q[0]
        m_hat_nx[1, ny_alias] = mi_q[1]

    low_level_stats = _get_all_stats()
    _sum_stats(stats_m_hat_nx, low_level_stats)

    return nx, m_hat_nx, stats_m_hat_nx


def fxp_multiprocessing_compute_m_hat(
    L_q: CFxpTensor,
    z_q: CFxpTensor,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> Tuple[CFxpTensor, Dict[str, Any]]:

    if L_q.ndim != 4 or L_q.shape[0:2] != (2, 2):
        raise ValueError(f"L_q debe tener shape (2,2,Nx,offset), recibió {L_q.shape}")

    if z_q.ndim != 3 or z_q.shape[0] != 2:
        raise ValueError(f"z_q debe tener shape (2,Nx,offset), recibió {z_q.shape}")

    NB = z_q.NB
    NBF = z_q.NBF
    signed = z_q.signed

    _, _, Nx, offset = L_q.shape
    if z_q.shape[1:] != (Nx, offset):
        raise ValueError("L_q y z_q deben tener la misma grilla (Nx,offset)")

    m_hat_q = CFxpTensor.zeros(
        shape=(2, Nx, offset),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats_m_hat_total: Dict[str, Any] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_m_hat,
        initargs=(L_q, z_q),
    ) as executor:

        for nx, m_hat_nx, stats_m_hat_nx in executor.map(
            _worker_compute_m_hat_nx,
            range(Nx),
            chunksize=chunksize,
        ):
            for ny_alias in range(offset):
                m_hat_q[0, nx, ny_alias] = m_hat_nx[0, ny_alias]
                m_hat_q[1, nx, ny_alias] = m_hat_nx[1, ny_alias]

            _sum_stats(stats_m_hat_total, stats_m_hat_nx)

    return m_hat_q, stats_m_hat_total
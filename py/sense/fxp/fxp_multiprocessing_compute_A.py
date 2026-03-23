import os
import sys
import numpy as np
from numpy.lib.npyio import NpzFile
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

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
_RE_RAW = None
_IM_RAW = None
_NB_S = None
_NBF_S = None
_SIGNED = None


def _init_worker_A(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    signed: bool,
) -> None:
    """
    Inicializa variables globales dentro de cada proceso worker.
    """
    global _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED
    _RE_RAW = re_raw
    _IM_RAW = im_raw
    _NB_S = NB_S
    _NBF_S = NBF_S
    _SIGNED = signed


def _fxp_compute_A_ij_from_raw(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB_S: int,
    NBF_S: int,
    signed: bool,
    nx: int,
    ny_alias: int,
) -> np.ndarray:
    """
    Calcula un bloque A_ij en fixed-point a partir de tensores raw.

    Input
    -----
    re_raw, im_raw : np.ndarray, shape (L, Nx, Ny)
    NB_S, NBF_S    : formato de entrada de S
    signed         : signed / unsigned
    nx, ny_alias   : índices locales

    Output
    ------
    Aij_np : np.ndarray, shape (2, 2), dtype=np.complex128
    """
    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    L, Nx, Ny = re_raw.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af
    ny0 = ny_alias
    ny1 = ny_alias + offset

    # crecimiento conservador:
    # producto: ~ 2*NB_S, 2*NBF_S
    # acumulación: sumar ceil(log2(L)) bits enteros
    grow_bits = int(np.ceil(np.log2(L))) if L > 1 else 0
    NB_A = 2 * NB_S + grow_bits
    NBF_A = 2 * NBF_S

    zero_re = Fxp.quantize(0.0, NB_A, NBF_A, mode="round", signed=signed)

    A00 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A11 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)
    A01 = CFxp.from_complex(0.0 + 0.0j, NB_A, NBF_A, mode="round", signed=signed)

    for l in range(L):
        s0 = CFxp.from_uint_pair(
            re_raw[l, nx, ny0],
            im_raw[l, nx, ny0],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )

        s1 = CFxp.from_uint_pair(
            re_raw[l, nx, ny1],
            im_raw[l, nx, ny1],
            NB=NB_S,
            NBF=NBF_S,
            signed=signed
        )

        # |s0|^2 = sr^2 + si^2   (real)
        p00_re = (s0.re * s0.re + s0.im * s0.im).cast(NB_A, NBF_A, mode="trunc")
        p11_re = (s1.re * s1.re + s1.im * s1.im).cast(NB_A, NBF_A, mode="trunc")

        p00 = CFxp(p00_re, zero_re)
        p11 = CFxp(p11_re, zero_re)

        # conj(s0) * s1   (complejo)
        p01 = (s0.conj() * s1).cast(NB_A, NBF_A, mode="trunc")

        A00 = A00 + p00
        A11 = A11 + p11
        A01 = A01 + p01

    A10 = A01.conj()

    Aij_np = np.array([
        [A00.to_complex(), A01.to_complex()],
        [A10.to_complex(), A11.to_complex()],
    ], dtype=np.complex128)

    return Aij_np


def _worker_compute_A_nx(nx: int) -> Tuple[int, np.ndarray]:
    """
    Worker: calcula toda la columna de bloques para un nx fijo.

    Output
    ------
    nx   : int
    A_nx : np.ndarray, shape (2, 2, offset), dtype=np.complex128
    """
    global _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED

    L, Nx, Ny = _RE_RAW.shape
    Af = 2
    offset = Ny // Af

    A_nx = np.zeros((2, 2, offset), dtype=np.complex128)

    for ny_alias in range(offset):
        A_nx[:, :, ny_alias] = _fxp_compute_A_ij_from_raw(
            _RE_RAW, _IM_RAW, _NB_S, _NBF_S, _SIGNED, nx, ny_alias
        )

    return nx, A_nx


def fxp_compute_A(
    S_q: NpzFile,
    max_workers: int | None = None,
    chunksize: int = 4,
) -> np.ndarray:
    """
    Calcula A completo en fixed-point usando multiprocessing por nx.

    Input
    -----
    S_q : NpzFile
        Debe contener:
            - re_raw
            - im_raw
            - NB
            - NBF
            - signed

    max_workers : int | None
        Cantidad de procesos. Si es None, usa os.cpu_count().

    chunksize : int
        Tamaño de chunk para executor.map.

    Output
    ------
    A : np.ndarray, shape (2, 2, Nx, offset), dtype=np.complex128
    """
    re_raw = S_q["re_raw"]
    im_raw = S_q["im_raw"]
    NB_S = int(S_q["NB"])
    NBF_S = int(S_q["NBF"])
    signed = bool(int(S_q["signed"]))

    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    if re_raw.ndim != 3:
        raise ValueError(f"re_raw debe ser 3D, recibió shape={re_raw.shape}")

    _, Nx, Ny = re_raw.shape
    Af = 2

    if Ny % Af != 0:
        raise ValueError("Ny debe ser par para Af = 2")

    offset = Ny // Af
    A = np.zeros((2, 2, Nx, offset), dtype=np.complex128)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_A,
        initargs=(re_raw, im_raw, NB_S, NBF_S, signed),
    ) as executor:

        for nx, A_nx in executor.map(_worker_compute_A_nx, range(Nx), chunksize=chunksize):
            A[:, :, nx, :] = A_nx

    return A
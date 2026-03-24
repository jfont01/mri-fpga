from cfxp import CFxp
from old.cfxp2d import CFxp2D
from fft1d import fft, build_twiddles
from typing import Optional, Dict, Tuple, List
from helpers import _accum_ops
import numpy as np, math

def fft2d(
    img: CFxp2D                 ,
    Wx: Optional[List[CFxp]]    ,
    Wy: Optional[List[CFxp]]    ,
    cast: bool                  ,
    NB_round: Optional[int]     ,
    NBF_round: Optional[int]    ,
    debug: bool                 ,
    shift_right_stage: bool     ,
) -> Tuple[CFxp2D, Dict[str, int]]:

    Ny, Nx = img.shape
    total_ops = {"bfly": 0, "cmul": 0, "caddsub": 0, "mul": 0, "addsub": 0}


    for y in range(Ny):
        row = img.data[y]
        row_fft, ops_row = fft(row, Wx, cast, NB_round, NBF_round, debug, shift_right_stage)
        img.data[y] = row_fft

        _accum_ops(total_ops, ops_row)

    for x in range(Nx):
        col = [img.data[y][x] for y in range(Ny)] 
        col_fft, ops_col = fft(col, Wy, cast, NB_round, NBF_round, debug, shift_right_stage)

        for y in range(Ny):
            img.data[y][x] = col_fft[y]

        _accum_ops(total_ops, ops_col)

    return img, total_ops




def fft2d_norm(
    img_q: CFxp2D,
    N: int,
    NB: int,
    NBF: int,
    cast: bool,
    mode: str,
    NB_round: int,
    NBF_round: int,
    debug: bool,
):
    """
    Ejecuta la FFT2D en punto fijo, calcula magnitud (log), fase,
    y estima cuántos bits enteros se necesitan.
    Devuelve: (mag_fft_log, phase_fft, total_ops, int_bits_needed, max_complex)
    """
    # Twiddles 1D (mismos en x e y)
    W = build_twiddles(N, NB, NBF, mode)

    # FFT2D
    img_fft, total_ops = fft2d(
        img=img_q,
        Wx=W,
        Wy=W,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
    )

    # Pasar a complejo 2D en float
    Ny, Nx = img_fft.shape
    arr_fft = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            arr_fft[y, x] = img_fft.data[y][x].to_complex()

    # Magnitud y fase
    mag_fft   = np.abs(arr_fft)
    phase_fft = np.angle(arr_fft)

    # Centrar DC
    mag_fft   = np.fft.fftshift(mag_fft)
    phase_fft = np.fft.fftshift(phase_fft)

    # Escalado log
    mag_fft_log = np.log1p(mag_fft)

    # Métricas de rango dinámico
    max_val     = img_fft.max_abs_value()
    max_complex = img_fft.max_abs_components()

    if max_val <= 0.0 or max_val <= 1.0:
        int_bits_needed = 0
    else:
        int_bits_needed = math.ceil(math.log2(max_val))

    return mag_fft_log, phase_fft, total_ops, int_bits_needed, max_complex




# =========================================================
#   IFFT2D
# =========================================================

def ifft2d(
    X: CFxp2D                 ,
    Wx: Optional[List[CFxp]]    ,
    Wy: Optional[List[CFxp]]    ,
    cast: bool                  ,
    NB_round: Optional[int]     ,
    NBF_round: Optional[int]    ,
    debug: bool                 ,
    shift_right_stage: bool     ,
    normalize: bool = False     ,
) -> Tuple[CFxp2D, Dict[str, int]]:
    """
    IFFT2D en punto fijo, basada en:
        ifft2(X) = conj( fft2( conj(X) ) ) / (N^2 si normalize=True)

    - img: CFxp2D con el espectro X[kx,ky]
    - Wx, Wy: twiddles de FFT
    - cast, NB_round, NBF_round, debug, shift_right_stage: igual que en fft2d
    - normalize: aplica un factor 1/N^2

    Devuelve:
        (img_spatial, ops_totales)
    """
    Ny, Nx = X.shape
    if Ny != Nx:
        raise ValueError(f"IFFT2D asumida sólo para matrices cuadradas, got {Ny}x{Nx}")
    N = Ny

    total_ops = {"bfly": 0, "cmul": 0, "caddsub": 0, "mul": 0, "addsub": 0}

    # 1) Conjugar entrada
    X.conj()

    # 2) Aplicar FFT2D "normal"
    img_fft, ops_fft = fft2d(
        img=X,
        Wx=Wx,
        Wy=Wy,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        shift_right_stage=shift_right_stage,
    )

    # 3) Volver a conjugar => ifft2(X) * (N^2 factor, si no normalizás)
    img_fft.conj()

    total_ops = ops_fft  # conj no lo contamos, pero podrías sumar si quisieras

    # 4) Normalización opcional por N^2
    if normalize:
        scale = 1.0 / (N * N)
        scale_c = CFxp.from_float(scale, NB_round if NB_round else img_fft.NB,
                                  img_fft.NBF)  # ajusta según tu CFxp

        for y in range(Ny):
            for x in range(Nx):
                img_fft.data[y][x] = img_fft.data[y][x] * scale_c
        total_ops["mul"] += Ny * Nx

    return img_fft, total_ops



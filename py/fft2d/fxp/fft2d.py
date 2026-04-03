import math, numpy as np, os, sys
from typing import Optional, Dict, Tuple, List, Any
from fft1d import fft, build_twiddles
# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxp           import CFxp
from cfxptensor import CFxpTensor
# ------------------------------------------------------------------
def _merge_stats(total: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in part.items():
        if k == "accumulators":
            if "accumulators" not in total:
                total["accumulators"] = {}

            for acc_name, acc_data in v.items():
                if acc_name not in total["accumulators"]:
                    total["accumulators"][acc_name] = dict(acc_data)
                else:
                    tgt = total["accumulators"][acc_name]
                    if "min_re" in acc_data:
                        tgt["min_re"] = min(tgt["min_re"], acc_data["min_re"])
                        tgt["max_re"] = max(tgt["max_re"], acc_data["max_re"])
                        tgt["min_im"] = min(tgt["min_im"], acc_data["min_im"])
                        tgt["max_im"] = max(tgt["max_im"], acc_data["max_im"])
                        if "min_abs" in acc_data:
                            tgt["min_abs"] = min(tgt["min_abs"], acc_data["min_abs"])
                            tgt["max_abs"] = max(tgt["max_abs"], acc_data["max_abs"])
        else:
            total[k] = total.get(k, 0) + int(v)

    return total

def _tensor_row_to_list(img: CFxpTensor, l: int, y: int) -> List[CFxp]:
    _, _, Nx = img.shape
    return [img[l, y, x] for x in range(Nx)]

def _tensor_col_to_list(img: CFxpTensor, l: int, x: int) -> List[CFxp]:
    _, Ny, _ = img.shape
    return [img[l, y, x] for y in range(Ny)]

def _write_row_from_list(img: CFxpTensor, l: int, y: int, row: List[CFxp]) -> None:
    for x, val in enumerate(row):
        img[l, y, x] = val

def _write_col_from_list(img: CFxpTensor, l: int, x: int, col: List[CFxp]) -> None:
    for y, val in enumerate(col):
        img[l, y, x] = val

def fxp_fft2d(
    img: CFxpTensor,
    Wx: Optional[List[CFxp]],
    Wy: Optional[List[CFxp]],
    cast: bool,
    NB_round: Optional[int],
    NBF_round: Optional[int],
    debug: bool,
    shift_right_stage: bool,
) -> Tuple[CFxpTensor, Dict[str, int]]:

    
    L, Ny, Nx = img.shape

    if Wx is None:
        Wx = build_twiddles(Nx, img.NB, img.NBF, mode="round")
    if Wy is None:
        Wy = build_twiddles(Ny, img.NB, img.NBF, mode="round")

    total_ops: Dict[str, int] = {}


    for l in range(L):
        # FFT por filas
        for y in range(Ny):
            row = _tensor_row_to_list(img, l, y)
            row_fft, ops_row = fft(
                x=row,
                W=Wx,
                cast=cast,
                NB_round=NB_round,
                NBF_round=NBF_round,
                debug=debug,
                shift_right_stage=shift_right_stage,
            )
            _write_row_from_list(img, l, y, row_fft)
            _merge_stats(total_ops, ops_row)

        # FFT por columnas
        for x in range(Nx):
            col = _tensor_col_to_list(img, l, x)
            col_fft, ops_col = fft(
                x=col,
                W=Wy,
                cast=cast,
                NB_round=NB_round,
                NBF_round=NBF_round,
                debug=debug,
                shift_right_stage=shift_right_stage,
            )
            _write_col_from_list(img, l, x, col_fft)
            _merge_stats(total_ops, ops_col)


    return img, total_ops


def fft2d_norm(
    img_q: CFxpTensor,
    NB: int,
    NBF: int,
    cast: bool,
    mode: str,
    NB_round: int,
    NBF_round: int,
    debug: bool,
    shift_right_stage: bool,
):
    """
    Ejecuta la FFT2D en punto fijo y devuelve:
    - magnitud logarítmica centrada
    - fase centrada
    - stats totales
    - bits enteros necesarios estimados
    - máximo valor absoluto complejo
    """

    if img_q.ndim != 3:
        raise ValueError(f"fft2d_norm espera un tensor 3D, recibió shape={img_q.shape}")

    L, Ny, Nx = img_q.shape

    Wx = build_twiddles(Nx, NB, NBF, mode)
    Wy = build_twiddles(Ny, NB, NBF, mode)

    img_fft, total_ops = fxp_fft2d(
        img=img_q,
        Wx=Wx,
        Wy=Wy,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        shift_right_stage=shift_right_stage,
    )

    arr_fft = img_fft.to_complex_ndarray().astype(np.complex128)

    mag_fft = np.abs(arr_fft)
    phase_fft = np.angle(arr_fft)

    mag_fft = np.fft.fftshift(mag_fft, axes=(-2, -1))
    phase_fft = np.fft.fftshift(phase_fft, axes=(-2, -1))

    mag_fft_log = np.log1p(mag_fft)

    max_val = float(np.max(np.abs(arr_fft)))

    if max_val <= 1.0:
        int_bits_needed = 0
    else:
        int_bits_needed = int(math.ceil(math.log2(max_val)))

    return mag_fft_log, phase_fft, total_ops, int_bits_needed, max_val


def fxp_ifft2d(
    K,
    Wx,
    Wy,
    cast,
    NB_round,
    NBF_round,
    debug,
    shift_right_stage,
):
    L, Ny, Nx = K.shape

    if Wx is None:
        Wx = build_twiddles(Nx, K.NB, K.NBF, mode="round")
    if Wy is None:
        Wy = build_twiddles(Ny, K.NB, K.NBF, mode="round")

    X_work = CFxpTensor.zeros(
        shape=(L, Ny, Nx),
        NB=K.NB,
        NBF=K.NBF,
        signed=K.signed,
    )

    for l in range(L):
        for y in range(Ny):
            for x in range(Nx):
                X_work[l, y, x] = K[l, y, x].conj()

    img_fft, ops_fft = fxp_fft2d(
        img=X_work,
        Wx=Wx,
        Wy=Wy,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        shift_right_stage=shift_right_stage,
    )


    for l in range(L):
        for y in range(Ny):
            for x in range(Nx):
                img_fft[l, y, x] = img_fft[l, y, x].conj().cast(
                    NB_round, NBF_round, mode="round", overflow="saturate"
                )

    total_ops = dict(ops_fft)
    return img_fft, total_ops
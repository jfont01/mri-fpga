
import math, numpy as np, os, sys
from typing import Tuple, Dict, List
# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxp           import CFxp
from fxp            import Fxp
# ------------------------------------------------------------------


def bfly(
    u: CFxp, 
    v: CFxp, 
    w: CFxp,
    cast: bool, 
    NB_round: int,
    NBF_round: int,
    shift_right_stage: bool,
    mode: str = "round",
    overflow: str = "saturate",
) -> Tuple[CFxp, CFxp]:

    t = w * v

    y0 = u + t
    y1 = u - t

    if cast:
        y0 = y0.cast(NB_round, NBF_round, mode=mode, overflow=overflow)
        y1 = y1.cast(NB_round, NBF_round, mode=mode, overflow=overflow)

    if shift_right_stage:           #Dividir por 2 por cada stage
        y0 = y0.shift_right(1)
        y1 = y1.shift_right(1)

    return y0, y1

def stage(
    x: list[CFxp],
    N: int,
    m: int,
    W: list[CFxp],
    cast: bool,
    NB_round: int,
    NBF_round: int,
    shift_right_stage: bool
) -> None:
    "Stages con Decimation In Time. Luego reacomodamos la salida X[k]"
    half = m // 2
    step = N // m

    for k in range(0, N, m):
        for j in range(half):
            w = W[j * step]
            u = x[k + j]
            v = x[k + j + half]
            x[k + j], x[k + j + half] = bfly(
                u, v, w,
                cast, NB_round, NBF_round, shift_right_stage
            )

def build_twiddles(N: int, NB: int = 16, NBF: int = 14, mode: str = "round") -> List[CFxp]:
    assert (N & (N - 1)) == 0, f"N debe ser potencia de 2"

    W = []
    for k in range(N):
        theta = -2.0 * math.pi * k / N
        Wk = CFxp.quantize(
            re_f=math.cos(theta),
            im_f=math.sin(theta),
            NB=NB,
            NBF=NBF,
            mode=mode,
            signed=True
        )
        W.append(Wk)
    return W

def bit_reverse(i: int, nbits: int) -> int:
    r = 0
    for _ in range(nbits):
        r = (r << 1) | (i & 1) #Corre r a la izquierda y agrega el último bit de i
        i >>= 1
    return r

def bit_reverse_reorder_out(x: list, nbits: int) -> List:
    N = len(x)
    return [x[bit_reverse(k, nbits)] for k in range(N)]

def fft(
    x: list[CFxp],
    W: list[CFxp],
    cast: bool,
    NB_round: int,
    NBF_round: int,
    debug: bool,
    shift_right_stage: bool
) -> Tuple[List[CFxp], Dict[str, int]]:
    x = x[:]
    N = len(x)
    stages = int(math.log2(N))

    if debug:
        print(f"[FFT] N={N}, stages={stages}")

    if hasattr(Fxp, "reset_fxp_stats"):
        Fxp.reset_fxp_stats()
    if hasattr(CFxp, "reset_cfxp_stats"):
        CFxp.reset_cfxp_stats()

    tmp = [None] * N
    for i in range(N):
        tmp[bit_reverse(i, stages)] = x[i]
    x[:] = tmp

    for s in range(1, stages + 1):
        m = 2**s

        stage(
            x, N, m, W,
            cast=cast,
            NB_round=NB_round,
            NBF_round=NBF_round,
            shift_right_stage=shift_right_stage
        )


    total_ops: Dict[str, int] = {}
    if hasattr(Fxp, "get_fxp_stats"):
        total_ops.update(Fxp.get_fxp_stats())
    if hasattr(CFxp, "get_cfxp_stats"):
        total_ops.update(CFxp.get_cfxp_stats())

    return x, total_ops

def ifft(
    X: list[CFxp],
    W: list[CFxp],
    cast: bool = False,
    NB_round: int = None,
    NBF_round: int = None,
    debug: bool = False,
    shift_right_stage: bool = False,
):
    N = len(X)

    # 1) Conjugar la entrada
    X_conj = [z.conj() for z in X]

    # 2) FFT de X_conj
    Y_q, ops_fft = fft(
        x=X_conj,
        W=W,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        shift_right_stage=shift_right_stage,
    )

    # 3) Conjugar salida y escalar por 1/N
    invN = 1.0 / N
    x = []
    for z in Y_q:
        zc = z.conj().to_complex()
        x.append(
            CFxp.quantize(
                zc.real * invN,
                zc.imag * invN,
                NB_round,
                NBF_round,
            )
        )

    return x, ops_fft

def fft_norm(
    N: int,
    Fs: float,
    x_q: list[CFxp],
    time: list,
    W: list[CFxp],
    NB: int,
    NBF: int,
    debug: bool = False,
    cast: bool = True,
    shift_right_stage: bool = False,
):
    coeffs, ops = fft(
        x_q[:],
        W,
        cast,
        NB,
        NBF,
        debug=debug,
        shift_right_stage=shift_right_stage
    )
    coeffs_complex = [z.to_complex() for z in coeffs]

    hz = np.linspace(0, Fs/2, int(math.floor(N/2.0)+1))

    amps = 2.0 * np.abs(coeffs_complex) / N
    amps = amps[:len(hz)]

    phases = np.angle(coeffs_complex)
    phases = phases[:len(hz)]

    x_q_float = [z.to_complex().real for z in x_q]

    return time, x_q_float, hz, amps, phases, ops


if __name__ == "__main__":
    pass

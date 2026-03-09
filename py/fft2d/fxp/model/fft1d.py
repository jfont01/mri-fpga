from cfxp import CFxp
import math, numpy as np
from helpers import _debug_stage_bits, _accum_ops, _calculate_stages
from typing import Tuple, Dict, List

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
) -> Dict[str, int]:
    "Stages con Decimation In Time. Luego reacomodamos la salida X[k]"
    half = m // 2
    step = N // m

    ops_fft = {"bfly": 0, "cmul": 0, "caddsub": 0}  # caddsub cuenta + y - (2 por butterfly)

    for k in range(0, N, m):
        for j in range(half):
            w = W[j * step]
            u = x[k + j]
            v = x[k + j + half]
            x[k + j], x[k + j + half] = bfly(u, v, w, cast, NB_round, NBF_round, shift_right_stage)

            ops_fft["bfly"] += 1
            ops_fft["cmul"] += 1         # t = w*v (1 mul compleja)
            ops_fft["caddsub"] += 2      # y0=u+t y y1=u-t (2 addsub complejas)

    return ops_fft

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
    """
    Si x está en orden bit-reversed, devuelve una lista en orden natural:
    out[k] = x[bit_reverse(k)]
    """
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
    stages = _calculate_stages(N)

    if debug:
        print(f"[FFT] N={N}, stages={stages}")

    # bit-reversal in-place
    tmp = [None] * N
    for i in range(N):
        tmp[bit_reverse(i, stages)] = x[i]
    x[:] = tmp

    # acumulador de operaciones
    total_ops = {"bfly": 0, "cmul": 0, "caddsub": 0, "mul": 0, "addsub": 0}

    for s in range(1, stages + 1):
        m = 2**s

        ops_stage = stage(
            x, N, m, W,
            cast=cast,
            NB_round=NB_round,
            NBF_round=NBF_round,
            shift_right_stage=shift_right_stage
        )
        _accum_ops(total_ops, ops_stage)

        if debug:
            _debug_stage_bits(x, s, stages)

    # calcular operaciones reales equivalentes
    total_ops["mul"]    = 4 * total_ops["cmul"]
    total_ops["addsub"] = 2 * total_ops["cmul"] + 2 * total_ops["caddsub"]

    return x, total_ops

def ifft(X: list[CFxp], W: list[CFxp], cast: bool = False, NB_round: int = None, NBF_round: int = None, debug: bool = False):

    N = len(X)

    # 1) Conjugar la entrada
    X_conj = [z.conj() for z in X]

    # 2) FFT de X_conj (mismos twiddles W)
    Y_q, ops_fft = fft(
        X=X_conj,
        W=W,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
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

    ops_ifft = ops_fft.copy()

    return x, ops_ifft

def fft_norm(N: int, Fs: float, x_q: list[CFxp], time: list, W: list[CFxp], NB: int, NBF: int, debug: bool = False, cast: bool = True):

    coeffs, ops = fft(x_q[:], W, cast, NB, NBF, debug=debug)
    coeffs_complex = [z.to_complex() for z in coeffs]

    hz = np.linspace(0, Fs/2, int(math.floor(N/2.0)+1))

    amps = 2.0 * np.abs(coeffs_complex) / N
    amps = amps[:len(hz)]

    phases = np.angle(coeffs_complex)
    phases = phases[:len(hz)]

    x_q_float = [z.to_complex().real for z in x_q]

    return time, x_q_float, hz, amps, phases


if __name__ == "__main__":
    pass

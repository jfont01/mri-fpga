import numpy as np
import os, sys

# ------------------------- ENVIROMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
# -------------------------------------------------------------------


def quantize_S(
    S: np.ndarray,
    NB: int,
    NBF: int,
    mode: str = "round",
    signed: bool = True,
):
    if S.ndim != 3:
        raise ValueError(f"S debe ser 3D, recibió shape={S.shape}")

    L, Nx, Ny = S.shape

    if NB <= 8:
        raw_dtype = np.uint8
    elif NB <= 16:
        raw_dtype = np.uint16
    elif NB <= 32:
        raw_dtype = np.uint32
    elif NB <= 64:
        raw_dtype = np.uint64
    else:
        raise ValueError(f"NB={NB} no soportado para almacenamiento raw en NumPy")

    re = np.zeros((L, Nx, Ny), dtype=raw_dtype)
    im = np.zeros((L, Nx, Ny), dtype=raw_dtype)

    for l in range(L):
        for nx in range(Nx):
            for ny in range(Ny):
                z_fx = CFxp.from_complex(
                    complex(S[l, nx, ny]),
                    NB=NB,
                    NBF=NBF,
                    mode=mode,
                    signed=signed
                )
                re[l, nx, ny] = z_fx.re.to_uint_raw()
                im[l, nx, ny] = z_fx.im.to_uint_raw()

    return re, im


def save_quantized_tensor_npz(
    out_path: str,
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB: int,
    NBF: int,
    mode: str = "round",
    signed: bool = True,
):
    np.savez(
        out_path,
        re_raw=re_raw,
        im_raw=im_raw,
        NB=np.int32(NB),
        NBF=np.int32(NBF),
        signed=np.int32(1 if signed else 0),
        mode=np.array(mode),
        layout=np.array("split_raw_words"),
        shape=np.array(re_raw.shape, dtype=np.int32),
        storage_dtype=np.array(str(re_raw.dtype)),
    )


def dequantize_raw_complex(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    NB: int,
    NBF: int,
    signed: bool = True,
) -> np.ndarray:
    if re_raw.shape != im_raw.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    L, Nx, Ny = re_raw.shape
    S_q = np.zeros((L, Nx, Ny), dtype=np.complex128)

    for l in range(L):
        for nx in range(Nx):
            for ny in range(Ny):
                z_fxp = CFxp.from_uint_pair(
                    re_raw[l, nx, ny],
                    im_raw[l, nx, ny],
                    NB=NB,
                    NBF=NBF,
                    signed=signed
                )
                S_q[l, nx, ny] = z_fxp.to_complex()

    return S_q

def write_quant_report(
    out_rpt_path: str,
    S_ref: np.ndarray,
    S_q: np.ndarray,
    NB: int,
    NBF: int,
    mode: str = "round",
    signed: bool = True,
):
    err = S_ref - S_q

    err_re = np.abs(S_ref.real - S_q.real)
    err_im = np.abs(S_ref.imag - S_q.imag)
    err_c = np.abs(err)

    max_abs_err_re = float(np.max(err_re))
    max_abs_err_im = float(np.max(err_im))
    max_abs_err_c  = float(np.max(err_c))

    mean_abs_err_re = float(np.mean(err_re))
    mean_abs_err_im = float(np.mean(err_im))
    mean_abs_err_c  = float(np.mean(err_c))

    rmse_c = float(np.sqrt(np.mean(np.abs(err) ** 2)))

    ref_norm = np.linalg.norm(S_ref.ravel())
    err_norm = np.linalg.norm(err.ravel())
    rel_l2_c = float(err_norm / ref_norm) if ref_norm > 0 else float(err_norm)

    scale = float(1 << NBF)
    qmin = -(1 << (NB - 1)) / scale
    qmax = ((1 << (NB - 1)) - 1) / scale

    sat_re = int(np.sum((S_ref.real < qmin) | (S_ref.real > qmax)))
    sat_im = int(np.sum((S_ref.imag < qmin) | (S_ref.imag > qmax)))

    idx_max = np.unravel_index(np.argmax(err_c), err_c.shape)
    l_max, nx_max, ny_max = idx_max

    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("QUANTIZATION REPORT\n")
        f.write("=========================================================\n\n")

        f.write(f"shape           : {S_ref.shape}\n")
        f.write(f"NB              : {NB}\n")
        f.write(f"NBF             : {NBF}\n")
        f.write(f"signed          : {signed}\n")
        f.write(f"mode            : {mode}\n")
        f.write(f"range_realizable: [{qmin}, {qmax}]\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err_re  : {max_abs_err_re:.12e}\n")
        f.write(f"max_abs_err_im  : {max_abs_err_im:.12e}\n")
        f.write(f"max_abs_err_c   : {max_abs_err_c:.12e}\n")
        f.write(f"mean_abs_err_re : {mean_abs_err_re:.12e}\n")
        f.write(f"mean_abs_err_im : {mean_abs_err_im:.12e}\n")
        f.write(f"mean_abs_err_c  : {mean_abs_err_c:.12e}\n")
        f.write(f"rmse_c          : {rmse_c:.12e}\n")
        f.write(f"rel_l2_c        : {rel_l2_c:.12e}\n\n")

        f.write("SATURATION COUNTS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"sat_re          : {sat_re}\n")
        f.write(f"sat_im          : {sat_im}\n\n")

        f.write("WORST COMPLEX SAMPLE\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"index           : (l={l_max}, nx={nx_max}, ny={ny_max})\n")
        f.write(f"S_ref           : {S_ref[idx_max]}\n")
        f.write(f"S_q             : {S_q[idx_max]}\n")
        f.write(f"abs_err_complex : {err_c[idx_max]:.12e}\n")
        f.write(f"abs_err_re      : {err_re[idx_max]:.12e}\n")
        f.write(f"abs_err_im      : {err_im[idx_max]:.12e}\n")


S = np.load("S.npy")

NB = 8
NBF = 7
MODE = "round"
SIGNED = True

re, im = quantize_S(S, NB, NBF, MODE, SIGNED)
save_quantized_tensor_npz("S.npz", re, im, NB, NBF, MODE, SIGNED)

S_q = dequantize_raw_complex(re, im, NB, NBF, SIGNED)
write_quant_report("S_quantization_report.rpt", S, S_q, NB, NBF, MODE, SIGNED)
import numpy as np, os, sys
# ------------------------- ENVIROMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from cfxp import CFxp
# -------------------------------------------------------------------

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
        shape=np.array(re_raw.shape, dtype=np.int32),
        storage_dtype=np.array(str(re_raw.dtype)),
    )

    


def cast_q_to_f_complex(
    re_u: np.ndarray,
    im_u: np.ndarray,
    NB: int,
    NBF: int,
    signed: bool = True,
) -> np.ndarray:
    if re_u.shape != im_u.shape:
        raise ValueError("re_raw e im_raw deben tener el mismo shape")

    L, Nx, Ny = re_u.shape
    S_q_f = np.zeros((L, Nx, Ny), dtype=np.complex128)

    for l in range(L):
        for nx in range(Nx):
            for ny in range(Ny):
                z_fxp = CFxp.from_uint_pair(
                    re_u[l, nx, ny],
                    im_u[l, nx, ny],
                    NB=NB,
                    NBF=NBF,
                    signed=signed
                )
                S_q_f[l, nx, ny] = z_fxp.to_complex()

    return S_q_f

def write_quant_report(
    out_rpt_path: str,
    S_ref: np.ndarray,
    S_q: np.ndarray,
    NB: int,
    NBF: int,
    ref_input_path: str,
    quant_input_path: str,
    mode: str = "round",
    signed: bool = True,
):
    if S_ref.shape != S_q.shape:
        raise ValueError(f"Shape mismatch: ref={S_ref.shape}, q={S_q.shape}")

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

    signal_power = float(np.mean(np.abs(S_ref) ** 2))
    noise_power  = float(np.mean(np.abs(err) ** 2))

    if noise_power > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

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

        f.write("INPUT FILES\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"ref_input_path        : {ref_input_path}\n")
        f.write(f"quant_input_path      : {quant_input_path}\n\n")

        f.write("FORMAT\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"shape                 : {S_ref.shape}\n")
        f.write(f"dtype_ref             : {S_ref.dtype}\n")
        f.write(f"dtype_q               : {S_q.dtype}\n")
        f.write(f"NB                    : {NB}\n")
        f.write(f"NBF                   : {NBF}\n")
        f.write(f"signed                : {signed}\n")
        f.write(f"mode                  : {mode}\n")
        f.write(f"range_realizable      : [{qmin}, {qmax}]\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err_re        : {max_abs_err_re:.12e}\n")
        f.write(f"max_abs_err_im        : {max_abs_err_im:.12e}\n")
        f.write(f"max_abs_err_complex   : {max_abs_err_c:.12e}\n")
        f.write(f"mean_abs_err_re       : {mean_abs_err_re:.12e}\n")
        f.write(f"mean_abs_err_im       : {mean_abs_err_im:.12e}\n")
        f.write(f"mean_abs_err_complex  : {mean_abs_err_c:.12e}\n")
        f.write(f"rmse_complex          : {rmse_c:.12e}\n")

        f.write("POWER AND SNR\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"signal_power          : {signal_power:.12e}\n")
        f.write(f"noise_power           : {noise_power:.12e}\n")
        f.write(f"snr_db                : {snr_db:.6f}\n\n")

        f.write("SATURATION COUNTS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"sat_re                : {sat_re}\n")
        f.write(f"sat_im                : {sat_im}\n\n")

        f.write("WORST COMPLEX SAMPLE\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"index                 : (l={l_max}, nx={nx_max}, ny={ny_max})\n")
        f.write(f"S_ref                 : {S_ref[idx_max]}\n")
        f.write(f"S_q                   : {S_q[idx_max]}\n")
        f.write(f"abs_err_complex       : {err_c[idx_max]:.12e}\n")
        f.write(f"abs_err_re            : {err_re[idx_max]:.12e}\n")
        f.write(f"abs_err_im            : {err_im[idx_max]:.12e}\n")

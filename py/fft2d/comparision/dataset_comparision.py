
import numpy as np, matplotlib.pyplot as plt, sys, argparse
import math, os, json, time
from typing import List, Tuple

# ---------------------------------------------------------
#   FLIST
# ---------------------------------------------------------
PY_FLIST_FXP_MODEL = os.environ.get("PY_FLIST_FXP_MODEL")
PY_FLIST_FFT2D     = os.environ.get("PY_FLIST_FFT2D")
PY_NPY_DATA     = os.environ.get("PY_NPY_DATA")

if PY_NPY_DATA is None:
    raise RuntimeError("[ERROR] PY_NPY_DATA no definido") 

if PY_FLIST_FXP_MODEL is None:
    raise RuntimeError("[ERROR] PY_FLIST_FXP_MODEL no definido") 

if PY_FLIST_FFT2D is None:
    raise RuntimeError("[ERROR] PY_FLIST_FFT2D no definido") 

for flist in (PY_FLIST_FXP_MODEL, PY_FLIST_FFT2D):

    with open(flist) as f:
        paths = [line.strip() for line in f if line.strip()]

    path_py = [r for r in paths if r.endswith(".py")]

    for pt in path_py:
        folder = os.path.dirname(
            os.path.abspath(os.path.join(os.path.dirname(flist), pt))
        )
        if folder not in sys.path:
            sys.path.append(folder)

from fft1d import build_twiddles
from fft2d import fft2d, ifft2d
from cfxp2d import CFxp2D


# ---------------------------------------------------------
#   COMPARISION FUNCTIONS
# ---------------------------------------------------------
def run_ifft2d_comparision(
    N: int,
    NB: int,
    NBF: int,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    npy_input: str,
    outdir: str,
    debug: bool,
    coil: int,   # <--- NUEVO
):
    """
    Testea la IFFT2D fxp sobre un k-space complejo DE UNA SOLA COIL:
      - npy_input: MTR_030_2d_kspace.npy con shape (Nc, N, N) o (N, N)
      - Se selecciona la coil mediante 'coil' (índice 0..Nc-1)
      - Compara: IFFT2D fxp vs numpy.fft.ifft2
      - Calcula métricas complejas y SNR/PSNR en dB
      - Guarda un PNG con |ifft numpy|, |ifft fxp| y |error|

    Devuelve:
        metrics: Dict[str,float]
        ops_ifft: Dict[str,int]
        elapsed: float
    """
    os.makedirs(outdir, exist_ok=True)

    t0 = time.perf_counter()

    # 1) Cargar k-space
    kspace = np.load(npy_input)
    if kspace.ndim == 2:
        # Caso un solo coil: (Ny, Nx) -> (1, Ny, Nx)
        kspace = kspace[np.newaxis, ...]
    elif kspace.ndim != 3:
        raise ValueError(f"Se esperaba shape (Nc,N,N) o (N,N), got {kspace.shape}")

    kspace = kspace.astype(np.complex128)
    Nc, Ny, Nx = kspace.shape

    if Ny != N or Nx != N:
        raise ValueError(f"k-space shape={kspace.shape}, N={N} no coincide con Ny=Nx")

    if not (0 <= coil < Nc):
        raise ValueError(f"Índice de coil inválido: coil={coil}, Nc={Nc}")

    # 2) Normalización global opcional (para no reventar el fixed point)
    max_abs = np.max(np.abs(kspace))
    if max_abs > 0:
        kspace_norm = kspace / max_abs
    else:
        kspace_norm = kspace

    # 3) Twiddles
    W = build_twiddles(N, NB, NBF, mode)

    # 4) Seleccionar la coil deseada
    Xc = kspace_norm[coil]  # (N, N), complejo

    # 4a) Referencia numpy
    img_np = np.fft.ifft2(Xc)       # complejo
    img_np_mag = np.abs(img_np)     # magnitud para visualizar

    # 4b) k-space fijo
    Xc_fx = CFxp2D.from_complex(Xc, NB, NBF)

    # 4c) IFFT2D fxp
    img_fx_2d, ops_ifft = ifft2d(
        X=Xc_fx,
        Wx=W,
        Wy=W,
        cast=cast,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        shift_right_stage=shift_right_stage,
        normalize=False,   # compensamos fuera
    )

    # 4d) Pasar resultado fxp a array complejo
    Ny2, Nx2 = img_fx_2d.shape
    img_fx = np.zeros((Ny2, Nx2), dtype=np.complex128)
    for y in range(Ny2):
        for x in range(Nx2):
            img_fx[y, x] = img_fx_2d.data[y][x].to_complex()

    # Compensar 1/N^2 si estás usando shift_right_stage en ifft2d
    #if shift_right_stage:
       # img_fx *= (N * N)

    img_fx_mag = np.abs(img_fx)

    # 4e) Métricas complejas
    diff    = img_fx - img_np
    err_mag = np.abs(diff)

    mse  = np.mean(err_mag**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(err_mag)
    max_abs_err = np.max(err_mag)

    signal_power = np.mean(np.abs(img_np)**2)
    if mse > 0:
        snr_db = 10.0 * np.log10(signal_power / mse)
    else:
        snr_db = np.inf

    peak = np.max(np.abs(img_np))
    if mse > 0:
        psnr_db = 20.0 * np.log10(peak / np.sqrt(mse))
    else:
        psnr_db = np.inf

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAX_ABS_ERR": max_abs_err,
        "SNR_dB": snr_db,
        "PSNR_dB": psnr_db,
    }

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # 5) Figura resumen: 3 filas x 1 columna
    fig, axes = plt.subplots(3, 1, figsize=(4, 10))

    # Fila 0: |IFFT numpy|
    ax = axes[0]
    im0 = ax.imshow(img_np_mag, cmap="gray")
    ax.set_title(f"coil {coil} |IFFT numpy|")
    ax.axis("off")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # Fila 1: |IFFT fxp|
    ax = axes[1]
    im1 = ax.imshow(img_fx_mag, cmap="gray")
    ax.set_title(f"coil {coil} |IFFT fxp|")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2: |error| + métricas en el título
    ax = axes[2]
    im2 = ax.imshow(err_mag, cmap="gray")
    ax.set_title(
        f"coil {coil} |err| SNR={metrics['SNR_dB']:.1f} dB\n"
        f"RMSE={metrics['RMSE']:.2e}"
    )
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    base = os.path.splitext(os.path.basename(npy_input))[0]
    png_path = os.path.join(
        outdir,
        f"ifft2d_kspace_coil{coil}_NB{NB}_NBF{NBF}_{base}.png"
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    if debug:
        print(f"[DEBUG] Figura IFFT2D k-space (coil={coil}) guardada en: {png_path}")

    return metrics, ops_ifft, elapsed

def run_fft2d_comparision(
    N: int,
    NB: int,
    NBF: int,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    npy_input: str,
    outdir: str,
    debug: bool
):
    
    # asegurar outdir
    os.makedirs(outdir, exist_ok=True)

    # ----------------- medir tiempo total FFT fxp -----------------
    t0 = time.perf_counter()

    # 1) Cargar la imagen desde npy (espacial, real o float)
    img_f = np.load(npy_input).astype(np.float64)
    max_abs = np.max(np.abs(img_f))
    if max_abs > 0:
        img_f = img_f / max_abs

    # 2) Imagen en fijo-punto
    img_fx = CFxp2D.from_float(img_f, NB, NBF)

    # 3) Twiddles S(NB, NBF)
    W = build_twiddles(N, NB, NBF, mode)

    # 4) FFT2D fxp
    img_fft_fx, ops_fft = fft2d(
        img=img_fx,
        Wx=W,
        Wy=W,
        cast=cast,
        shift_right_stage=shift_right_stage,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
    )

    # 5) Referencia en float (numpy)
    if shift_right_stage:
        X_ref = np.fft.fft2(img_f.astype(np.complex128)) / (N**2)
    else:
        X_ref = np.fft.fft2(img_f.astype(np.complex128))

    # 6) Métricas de error complejas (CFxp2D vs referencia compleja)
    metrics = img_fft_fx.quant_metrics_complex(X_ref)

    # 7) Construir array complejo 2D de la FFT fxp
    Ny, Nx = img_fft_fx.shape
    X_fxp = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            X_fxp[y, x] = img_fft_fx.data[y][x].to_complex()

    # 8) Magnitud y fase (con fftshift)
    mag_fxp   = np.fft.fftshift(np.abs(X_fxp))
    mag_ref   = np.fft.fftshift(np.abs(X_ref))
    phase_fxp = np.fft.fftshift(np.angle(X_fxp))
    phase_ref = np.fft.fftshift(np.angle(X_ref))

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # ----------------- Figura resumen -----------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Fila 1, col 1: imagen de entrada (espacial)
    ax = axes[0, 0]
    im0 = ax.imshow(img_f, cmap="gray")
    ax.set_title("Input Real)")
    ax.axis("off")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # Fila 1, col 2: |FFT2D fxp|
    ax = axes[0, 1]
    im1 = ax.imshow(mag_fxp, cmap="gray")
    ax.set_title("FFT2D fxp mag")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Fila 1, col 3: |FFT2D numpy|
    ax = axes[0, 2]
    im2 = ax.imshow(mag_ref, cmap="gray")
    ax.set_title("FFT2D numpy mag")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 1: fase FFT2D fxp
    ax = axes[1, 0]
    im3 = ax.imshow(phase_fxp, cmap="gray")
    ax.set_title("FFT2D fxp phase")
    ax.axis("off")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 2: fase FFT2D numpy
    ax = axes[1, 1]
    im4 = ax.imshow(phase_ref, cmap="gray")
    ax.set_title("FFT2D numpy phase")
    ax.axis("off")
    fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 3: panel de texto
    ax_text = axes[1, 2]
    ax_text.axis("off")

    # Construir texto con parámetros, métricas, ops y tiempo
    lines = []
    lines.append(f"N = {N}")
    lines.append(f"S({NB},{NBF}), cast={cast}, shift={shift_right_stage}")
    lines.append(f"mode = {mode}")
    if NB_round is not None and NBF_round is not None and cast:
        lines.append(f"round to S({NB_round},{NBF_round}) in each stage")
    lines.append("")
    lines.append("Metrics:")
    for k, v in metrics.items():
        lines.append(f"  {k}: {v:.4}")
    lines.append("")
    lines.append("Ops FFT2D:")
    for k, v in ops_fft.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Elapsed: {elapsed:.3f} s")

    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

    # Nombre de archivo según el .npy de entrada
    base = os.path.splitext(os.path.basename(npy_input))[0]
    png_path = os.path.join(outdir, f"fft2d_summary_{base}.png")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    if debug:
        print(f"[DEBUG] Figura resumen guardada en: {png_path}")

    return ops_fft, metrics, elapsed


# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
    description="Corre FFT2D fxp vs np.fft.fft2 sobre una imagen .npy"
    )

    parser.add_argument(
        "--coil",
        type=int,
        default=0,
        help="Índice de coil a procesar (0..Nc-1) para la opción --op ifft",
    )

    parser.add_argument(
        "--op",
        choices=["fft", "ifft"],
        default="fft",
        help="Operación a testear: 'fft' (default) o 'ifft' (pipeline fft+ifft)",
    )
    # Parámetros numéricos del formato
    parser.add_argument("--N", type=int, required=True,
                        help="Tamaño de la FFT2D (N x N)")
    parser.add_argument("--NB", type=int, required=True,
                        help="Número de bits totales S(NB,NBF)")
    parser.add_argument("--NBF", type=int, required=True,
                        help="Número de bits fraccionales S(NB,NBF)")


    # Flags booleanos
    parser.add_argument(
        "--cast",
        action="store_true",
        help="Activar cast por etapa (usa NB_round/NBF_round)"
    )

    parser.add_argument(
        "--shift-right-stage",
        action="store_true",
        help="Divide por 2 en cada etapa (equivale a /N^2 al final)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Imprimir info de depuración"
    )

    # Modo de twiddles
    parser.add_argument("--mode", type=str, default="natural",
                        help="Modo de twiddles (ej: 'natural', 'bitrev', etc.)")


    # Formato de redondeo opcional
    parser.add_argument("--NB-round", type=int, default=None,
                        help="NB de salida tras cast; si omitido, se usa el NB interno")
    parser.add_argument("--NBF-round", type=int, default=None,
                        help="NBF de salida tras cast; si omitido, se usa el NBF interno")


    # Entradas y salidas
    parser.add_argument("--npy-input", type=str, required=True,
                        help="Ruta al .npy con la imagen espacial (float, normalmente normalizada)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directorio de salida para futuros resultados/figuras")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"========= Running {args.op} comparision from {args.npy_input} =========")

    if args.op == "fft":
        ops_fft, metrics, elapsed = run_fft2d_comparision(
            N=args.N,
            NB=args.NB,
            NBF=args.NBF,
            cast=args.cast,
            shift_right_stage=args.shift_right_stage,
            mode=args.mode,
            NB_round=args.NB_round,
            NBF_round=args.NBF_round,
            npy_input=args.npy_input,
            outdir=args.outdir,
            debug=args.debug,
        )

        print("========= Métricas FFT2D vs np.fft.fft2 =========")
        for k, v in metrics.items():
            print(f"{k:>10}: {v}")
        print("Ops FFT2D:", ops_fft)
        print(f"Elapsed time: ({elapsed:.2f} s)")

    else:  # args.op == "ifft"
        metrics_ifft, ops_ifft, elapsed = run_ifft2d_comparision(
            N=args.N,
            NB=args.NB,
            NBF=args.NBF,
            cast=args.cast,
            shift_right_stage=args.shift_right_stage,
            mode=args.mode,
            NB_round=args.NB_round,
            NBF_round=args.NBF_round,
            npy_input=args.npy_input,
            outdir=args.outdir,
            debug=args.debug,
            coil=args.coil,
        )

        print("========= Métricas IFFT2D fxp vs numpy.fft.ifft2 (k-space) =========")
        print(f"--- Coil {args.coil} ---")
        for k, v in metrics_ifft.items():
            print(f"{k:>10}: {v}")
        print("Ops IFFT2D:", ops_ifft)
        print(f"Elapsed time: ({elapsed:.2f} s)")

if __name__ == "__main__":
    main()

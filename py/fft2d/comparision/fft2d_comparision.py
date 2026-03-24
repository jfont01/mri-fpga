
import numpy as np, matplotlib.pyplot as plt, sys
import math, os, json, time
from typing import List, Tuple

# ---------------------------------------------------------
#   FLIST
# ---------------------------------------------------------

PY_FLIST_FXP_MODEL = os.environ.get("PY_FLIST_FXP_MODEL")
PY_FLIST_FFT2D     = os.environ.get("PY_FLIST_FFT2D")
PY_NPY_DATA     = os.environ.get("PY_NPY_DATA")

PY_NPY_DATA = os.getenv("PY_NPY_DATA")

if PY_NPY_DATA is None:
    raise RuntimeError("PY_NPY_DATA no definido")

knee_path = os.path.join(PY_NPY_DATA, "MTR_030_2d_target_mag.npy")


for flist in (PY_FLIST_FXP_MODEL, PY_FLIST_FFT2D):
    if not flist:
        raise RuntimeError("[ERROR] PY_FLIST_FXP_MODEL / PY_FLIST_FFT2D no definidos")

    with open(flist) as f:
        paths = [line.strip() for line in f if line.strip()]

    path_py = [r for r in paths if r.endswith(".py")]

    for pt in path_py:
        folder = os.path.dirname(
            os.path.abspath(os.path.join(os.path.dirname(flist), pt))
        )
        if folder not in sys.path:
            sys.path.append(folder)


PY_FFT2D_CONFIG_COMPARISION_JSON = os.environ.get("PY_FFT2D_CONFIG_COMPARISION_JSON")

if not PY_FFT2D_CONFIG_COMPARISION_JSON:
    raise RuntimeError(
        "[ERROR] PY_FFT2D_CONFIG_COMPARISION_JSON environment variable is missing"
    )

from stimulus_gen import gen_tone_cos_2d, gen_concentric_rings_2d, gen_gaussian_spots_2d
from fft1d import build_twiddles
from fft2d import fft2d, ifft2d
from old.cfxp2d import CFxp2D


def run_comparision1(
    N: int,
    kx0: int,
    ky0: int,
    NB: int,
    NBF: int,
    A: float,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    outdir: str,
    debug: bool
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    # 1) estímulo: float + fxp
    img_fx, img_f = gen_tone_cos_2d(Nx, Ny, kx0, ky0, NB, NBF, A)

    # 2) twiddles y FFT2D en fijo
    W = build_twiddles(N, NB, NBF, mode)

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

    # 3) referencia en float (numpy)
    if shift_right_stage:
        X_ref = np.fft.fft2(img_f.astype(np.complex128)) / (N**2)
    else:
        X_ref = np.fft.fft2(img_f.astype(np.complex128))

    max_value_np = float(np.max(np.abs(X_ref)))
    if max_value_np <= 0.0:
        int_bits_needed_np = 0
    elif max_value_np <= 1.0:
        int_bits_needed_np = 0
    else:
        int_bits_needed_np = math.ceil(math.log2(max_value_np))


    # 4) métricas complejas
    metrics = img_fft_fx.quant_metrics_complex(X_ref)

    # ---------- Construir versiones complejas 2D de la FFT fxp ----------
    Ny, Nx = img_fft_fx.shape
    X_fxp = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            X_fxp[y, x] = img_fft_fx.data[y][x].to_complex()

    # Magnitudes con fftshift y log1p
    mag_fxp  = np.fft.fftshift(np.abs(X_fxp))
    mag_ref  = np.fft.fftshift(np.abs(X_ref))


    # ---------- Figura con 3 subplots ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Espacio (imagen float)
    im0 = axes[0].imshow(img_f, cmap="jet")
    axes[0].set_title("Fxp float")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Magnitud FFT2D fxp
    im1 = axes[1].imshow(mag_fxp, cmap="jet")
    if cast:
        axes[1].set_title(f"FFT2D fxp S({NB_round},{NBF_round})")
    else:
        axes[1].set_title(f"FFT2D fxp S({img_fft_fx.NB},{img_fft_fx.NBF})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Magnitud FFT2D numpy (referencia)
    im2 = axes[2].imshow(mag_ref, cmap="jet")
    axes[2].set_title("FFT2D np.fft.fft2")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Comparación FFT2D: N={N}, kx0={kx0}, ky0={ky0}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    fig.tight_layout()
    out_name = f"fft2d_comparision_N{N}_kx{kx0}_ky{ky0}.png"
    fig.savefig(os.path.join(outdir, out_name), dpi=300)
    plt.close(fig)

    return metrics, ops_fft, max_value_np, int_bits_needed_np

def run_comparision2(
    N: int,
    NB: int,
    NBF: int,
    A: float,
    rings_period: float,
    phase0: float,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    outdir: str,
    debug: bool
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    # 1) estímulo: float + fxp
    img_fx, img_f = gen_concentric_rings_2d(Nx, Ny, NB, NBF, A, rings_period, phase0)

    # 2) twiddles y FFT2D en fijo
    W = build_twiddles(N, NB, NBF, mode)

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

    # 3) referencia en float (numpy)
    if shift_right_stage:
        X_ref = np.fft.fft2(img_f.astype(np.complex128)) / (N**2)
    else:
        X_ref = np.fft.fft2(img_f.astype(np.complex128))

    max_value_np = float(np.max(np.abs(X_ref)))
    if max_value_np <= 0.0:
        int_bits_needed_np = 0
    elif max_value_np <= 1.0:
        int_bits_needed_np = 0
    else:
        int_bits_needed_np = math.ceil(math.log2(max_value_np))


    # 4) métricas complejas
    metrics = img_fft_fx.quant_metrics_complex(X_ref)

    # ---------- Construir versiones complejas 2D de la FFT fxp ----------
    Ny, Nx = img_fft_fx.shape
    X_fxp = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            X_fxp[y, x] = img_fft_fx.data[y][x].to_complex()

    # Magnitudes con fftshift y log1p
    mag_fxp  = np.fft.fftshift(np.abs(X_fxp))
    mag_ref  = np.fft.fftshift(np.abs(X_ref))


    # ---------- Figura con 3 subplots ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Espacio (imagen float)
    im0 = axes[0].imshow(img_f, cmap="jet")
    axes[0].set_title("Fxp float")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Magnitud FFT2D fxp
    im1 = axes[1].imshow(mag_fxp, cmap="jet")
    if cast:
        axes[1].set_title(f"FFT2D fxp S({NB_round},{NBF_round})")
    else:
        axes[1].set_title(f"FFT2D fxp S({img_fft_fx.NB},{img_fft_fx.NBF})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Magnitud FFT2D numpy (referencia)
    im2 = axes[2].imshow(mag_ref, cmap="jet")
    axes[2].set_title("FFT2D np.fft.fft2")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Comparación FFT2D: N={N}, rings_period={rings_period}, phase0={phase0}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    fig.tight_layout()
    out_name = f"fft2d_comparision_rings_N{N}_.png"
    fig.savefig(os.path.join(outdir, out_name), dpi=300)
    plt.close(fig)

    return metrics, ops_fft, max_value_np, int_bits_needed_np

def run_comparision3(
    N: int,
    NB: int,
    NBF: int,
    A: float,
    variance: float,
    centers: List[Tuple[int, int]],
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    outdir: str,
    debug: bool
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    # 1) estímulo: float + fxp
    img_fx, img_f = gen_gaussian_spots_2d(Nx, Ny, NB, NBF, variance, centers, A)

    # 2) twiddles y FFT2D en fijo
    W = build_twiddles(N, NB, NBF, mode)

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

    # 3) referencia en float (numpy)
    if shift_right_stage:
        X_ref = np.fft.fft2(img_f.astype(np.complex128)) / (N**2)
    else:
        X_ref = np.fft.fft2(img_f.astype(np.complex128))

    max_value_np = float(np.max(np.abs(X_ref)))
    if max_value_np <= 0.0:
        int_bits_needed_np = 0
    elif max_value_np <= 1.0:
        int_bits_needed_np = 0
    else:
        int_bits_needed_np = math.ceil(math.log2(max_value_np))


    # 4) métricas complejas
    metrics = img_fft_fx.quant_metrics_complex(X_ref)

    # ---------- Construir versiones complejas 2D de la FFT fxp ----------
    Ny, Nx = img_fft_fx.shape
    X_fxp = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            X_fxp[y, x] = img_fft_fx.data[y][x].to_complex()

    # Magnitudes con fftshift y log1p
    mag_fxp  = np.fft.fftshift(np.abs(X_fxp))
    mag_ref  = np.fft.fftshift(np.abs(X_ref))


    # ---------- Figura con 3 subplots ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Espacio (imagen float)
    im0 = axes[0].imshow(img_f, cmap="jet")
    axes[0].set_title("Fxp float")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Magnitud FFT2D fxp
    im1 = axes[1].imshow(mag_fxp, cmap="jet")
    if cast:
        axes[1].set_title(f"FFT2D fxp S({NB_round},{NBF_round})")
    else:
        axes[1].set_title(f"FFT2D fxp S({img_fft_fx.NB},{img_fft_fx.NBF})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Magnitud FFT2D numpy (referencia)
    im2 = axes[2].imshow(mag_ref, cmap="jet")
    axes[2].set_title("FFT2D np.fft.fft2")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Comparación FFT2D: N={N}, , "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    fig.tight_layout()
    out_name = f"fft2d_comparision_gaussian_dots_N{N}.png"
    fig.savefig(os.path.join(outdir, out_name), dpi=300)
    plt.close(fig)

    return metrics, ops_fft, max_value_np, int_bits_needed_np

def run_comparision4(
    N: int,
    NB: int,
    NBF: int,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    outdir: str,
    debug: bool
):
    """
    Versión adaptada para usar una imagen real guardada en:
        mri_fft2d_tests/skmtea_MTR_030_target_512.npy

    Hace:
      - carga de imagen espacial (float)
      - cuantización a CFxp2D
      - FFT2D fija
      - referencia np.fft.fft2
      - métricas + figura comparativa
    """

    os.makedirs(outdir, exist_ok=True)

    # -------- 1) Cargar imagen espacial desde .npy --------
    img_f = np.load(knee_path).astype(np.float64)   # (Ny, Nx)

    Ny, Nx = img_f.shape
    if Ny != Nx:
        raise ValueError(f"Imagen no cuadrada: {Ny}x{Nx}")

    # Si N viene de la config, comprobamos que coincida
    if N is not None and N != Nx:
        print(f"[WARNING] N={N} pero la imagen es {Nx}x{Ny}. Usando N={Nx}.")
    N = Nx
    Nx = Ny = N

    # 2) Imagen en fijo-punto
    img_fx = CFxp2D.from_float(img_f, NB, NBF)

    # 3) Twiddles y FFT2D en fijo
    W = build_twiddles(N, NB, NBF, mode)

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

    # 4) Referencia en float (numpy)
    if shift_right_stage:
        X_ref = np.fft.fft2(img_f.astype(np.complex128)) / (N**2)
    else:
        X_ref = np.fft.fft2(img_f.astype(np.complex128))

    max_value_np = float(np.max(np.abs(X_ref)))
    if max_value_np <= 0.0:
        int_bits_needed_np = 0
    elif max_value_np <= 1.0:
        int_bits_needed_np = 0
    else:
        int_bits_needed_np = math.ceil(math.log2(max_value_np))

    # 5) Métricas complejas (CFxp2D vs referencia compleja)
    metrics = img_fft_fx.quant_metrics_complex(X_ref)

    # 6) Construir array complejo 2D de la FFT fxp
    Ny, Nx = img_fft_fx.shape
    X_fxp = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in range(Ny):
        for x in range(Nx):
            X_fxp[y, x] = img_fft_fx.data[y][x].to_complex()

    # Magnitudes con fftshift (sin log, igual que tus otros tests)
    mag_fxp = np.fft.fftshift(np.abs(X_fxp))
    mag_ref = np.fft.fftshift(np.abs(X_ref))

    # 7) Figura con 3 subplots (espacio, fxp, numpy)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) Espacio (imagen float)
    im0 = axes[0].imshow(img_f, cmap="gray")
    axes[0].set_title("Imagen espacial (float)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Magnitud FFT2D fxp
    im1 = axes[1].imshow(mag_fxp, cmap="gray")
    if cast:
        axes[1].set_title(f"FFT2D fxp S({NB_round},{NBF_round})")
    else:
        axes[1].set_title(f"FFT2D fxp S({img_fft_fx.NB},{img_fft_fx.NBF})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Magnitud FFT2D numpy (referencia)
    im2 = axes[2].imshow(mag_ref, cmap="gray")
    axes[2].set_title("FFT2D np.fft.fft2")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Comparación FFT2D SKM-TEA: N={N}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    fig.tight_layout()
    out_name = f"fft2d_comparision_skmtea_MTR_030_N{N}.png"
    fig.savefig(os.path.join(outdir, out_name), dpi=300)
    plt.close(fig)

    return metrics, ops_fft, max_value_np, int_bits_needed_np

def run_comparision5(
    N: int,
    NB: int,
    NBF: int,
    A: float,
    rings_period: float,
    phase0: float,
    cast: bool,
    shift_right_stage: bool,
    mode: str,
    NB_round: int | None,
    NBF_round: int | None,
    outdir: str,
    debug: bool
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    # 1) estímulo: float + fxp
    img_fx, img_f = gen_concentric_rings_2d(Nx, Ny, NB, NBF, A, rings_period, phase0)

    # 2) twiddles 
    W = build_twiddles(N, NB, NBF, mode)

    # 3) FFT2D en fxp
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

    # 4) FFT2D referencia en numpy
    X_ref = np.fft.fft2(img_f.astype(np.complex128))

    # 5) IFFT2D referencia en numpy     
    img_i_f = np.fft.ifft2(X_ref).real   # (Ny, Nx), float64

    # 6) IFFT2D fxp  
    img_ifft_fx, ops_ifft = ifft2d(
        X=img_fft_fx,
        Wx=W,
        Wy=W,
        cast=cast,
        shift_right_stage=shift_right_stage,
        NB_round=NB_round,
        NBF_round=NBF_round,
        debug=debug,
        normalize=False,   # pon True si tu ifft2d internamente no escala
    )

    # 7) Pasar la imagen fija reconstruida a float para comparar (parte real)
    Ny, Nx = img_ifft_fx.shape
    img_i_fx = np.zeros((Ny, Nx), dtype=np.float64)
    for y in range(Ny):
        for x in range(Nx):
            img_i_fx[y, x] = img_ifft_fx.data[y][x].to_complex().real

    # Compensar el 1/N² del pipeline fijo
    if shift_right_stage:
        img_i_fx *= (N * N)
        
    diff = img_i_fx - img_i_f  
    mse  = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(diff))
    max_abs_err = np.max(np.abs(diff))

    # --- SNR en dB ---
    signal_power = np.mean(img_i_f**2)
    noise_power  = np.mean(diff**2)
    if noise_power > 0 and signal_power > 0:
        snr_db = 10.0 * np.log10(signal_power / noise_power)
    else:
        snr_db = np.inf  # caso límite (ruido ~0)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAX_ABS_ERR": max_abs_err,
        "SNR_dB": snr_db,
    }

    # 9) Magnitud y fase de FFT numpy con fftshift (para visualizar)
    mag_fft_np   = np.fft.fftshift(np.abs(X_ref))
    phase_fft_np = np.fft.fftshift(np.angle(X_ref))

    # 10) Fig resumen en un PNG
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Fila 1, col 1: |FFT2D numpy|
    ax = axes[0, 0]
    im0 = ax.imshow(mag_fft_np, cmap="jet")
    ax.set_title("FFT2D numpy |X|")
    ax.axis("off")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # Fila 1, col 2: fase FFT2D numpy
    ax = axes[0, 1]
    im1 = ax.imshow(phase_fft_np, cmap="jet")
    ax.set_title("FFT2D numpy phase(X)")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Fila 1, col 3: IFFT2D numpy (espacio)
    ax = axes[0, 2]
    im2 = ax.imshow(img_i_f, cmap="jet")
    ax.set_title("IFFT2D numpy")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 1: IFFT2D fxp (espacio)
    ax = axes[1, 0]
    im3 = ax.imshow(img_i_fx, cmap="jet")
    ax.set_title("IFFT2D fxp")
    ax.axis("off")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 2: |IFFT_fxp - IFFT_np|
    ax = axes[1, 1]
    im4 = ax.imshow(np.abs(diff), cmap="jet")
    ax.set_title("|IFFT_fxp - IFFT_np|")
    ax.axis("off")
    fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    # Fila 2, col 3: panel de texto (métricas + ops)
    ax_text = axes[1, 2]
    ax_text.axis("off")

    lines = []
    lines.append(f"N = {N}")
    lines.append(f"S({NB},{NBF}), cast={cast}, shift={shift_right_stage}")
    lines.append(f"mode = {mode}")
    if cast and NB_round is not None and NBF_round is not None:
        lines.append(f"round S({NB_round},{NBF_round})")
    lines.append("")
    lines.append("IFFT2D metrics (fxp vs numpy):")
    for k, v in metrics.items():
        lines.append(f"  {k}: {v:.3e}")
    lines.append("")
    lines.append("Ops FFT2D:")
    for k, v in ops_fft.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Ops IFFT2D:")
    for k, v in ops_ifft.items():
        lines.append(f"  {k}: {v}")

    ax_text.text(
        0.0, 1.0,
        "\n".join(lines),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

    fig.tight_layout()
    png_path = os.path.join(outdir, f"ifft2d_comparision_rings_N{N}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    if debug:
        print(f"[DEBUG] Figura IFFT2D guardada en: {png_path}")

    # Puedes devolver también ops_fft/ops_ifft si querés
    return metrics, ops_fft, ops_ifft


def main():

    with open(PY_FFT2D_CONFIG_COMPARISION_JSON, "r") as f:
        cfg_all = json.load(f)

    root  = cfg_all["fft2d"]
    selected   = str(root.get("test", "1"))
    cases = root.get("cases", {})

    def case(tid: str) -> dict:
        """Devuelve el diccionario de parámetros para el test tid."""
        cfg = cases.get(tid, {})
        if not cfg:
            print(f"[WARNING] No hay config para test {tid} en JSON, usando defaults.")
        return cfg

    # ------------------------------ TEST 1 ------------------------------
    if selected in ("1", "all"):
        c1 = case("1")
        N  = int(c1.get("N"))
        kx0  = int(c1.get("kx0"))
        ky0  = int(c1.get("ky0"))
        NB  = int(c1.get("NB"))
        NBF  = int(c1.get("NBF"))
        A  = float(c1.get("A"))
        outdir  = str(c1.get("outdir"))
        cast  = bool(c1.get("cast"))
        shift_right_stage  = bool(c1.get("shift_right_stage"))
        mode  = str(c1.get("mode"))
        NB_round  = int(c1.get("NB_round"))
        NBF_round  = int(c1.get("NBF_round"))
        debug  = bool(c1.get("debug"))

        print(f"========== COMPARISION 1: tono real 2D N = {N} ==========")
        t0 = time.perf_counter()
        metrics, ops_fft, max_value_np, int_bits_needed_np = run_comparision1(N ,kx0, ky0, NB, NBF, A, cast, shift_right_stage, mode, NB_round, NBF_round, outdir, debug)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("=== Métricas FFT2D vs numpy.fft.fft2 ===")
        for k, v in metrics.items():
            print(f"{k:>10}: {v}")
        print("Ops FFT2D:", ops_fft)
        print("Max value np.fft.fft2:", max_value_np)
        print("Int bits needed np.fft.fft2:", int_bits_needed_np)
        print(f"Elapsed time: ({elapsed:.2f} s)")
        print("")


    # ------------------------------ TEST 2 ------------------------------
    if selected in ("2", "all"):
        c2 = case("2")
        N  = int(c2.get("N"))
        NB  = int(c2.get("NB"))
        NBF  = int(c2.get("NBF"))
        A  = float(c2.get("A"))
        rings_period  = float(c2.get("rings_period"))
        phase0  = float(c2.get("phase0"))
        outdir  = str(c2.get("outdir"))
        cast  = bool(c2.get("cast"))
        shift_right_stage  = bool(c2.get("shift_right_stage"))
        mode  = str(c2.get("mode"))
        NB_round  = int(c2.get("NB_round"))
        NBF_round  = int(c2.get("NBF_round"))
        debug  = bool(c2.get("debug"))

        print(f"========== COMPARISION 2: concentric rings 2D N = {N} ==========")
        t0 = time.perf_counter()
        metrics, ops_fft, max_value_np, int_bits_needed_np = run_comparision2(N, NB, NBF, A, rings_period, phase0, cast, shift_right_stage,
                                                                              mode, NB_round, NBF_round, outdir, debug)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("=== Métricas FFT2D vs numpy.fft.fft2 ===")
        for k, v in metrics.items():
            print(f"{k:>10}: {v}")
        print("Ops FFT2D:", ops_fft)
        print("Max value np.fft.fft2:", max_value_np)
        print("Int bits needed np.fft.fft2:", int_bits_needed_np)
        print(f"Elapsed time: ({elapsed:.2f} s)")
        print("")


    # ------------------------------ TEST 3 ------------------------------
    if selected in ("3", "all"):
        c3 = case("3")
        N  = int(c3.get("N"))
        NB  = int(c3.get("NB"))
        NBF  = int(c3.get("NBF"))
        A  = float(c3.get("A"))
        variance  = float(c3.get("variance"))
        outdir  = str(c3.get("outdir"))
        cast  = bool(c3.get("cast"))
        shift_right_stage  = bool(c3.get("shift_right_stage"))
        mode  = str(c3.get("mode"))
        NB_round  = int(c3.get("NB_round"))
        NBF_round  = int(c3.get("NBF_round"))
        debug  = bool(c3.get("debug"))
        centers_raw = c3.get("centers", [])
        centers: List[Tuple[int, int]] = [
            (int(cx), int(cy)) for cx, cy in centers_raw
        ]

        print(f"========== COMPARISION 3: gaussian dots 2D N = {N} ==========")
        t0 = time.perf_counter()
        metrics, ops_fft, max_value_np, int_bits_needed_np = run_comparision3(N, NB, NBF, A, variance, centers, cast, shift_right_stage,
                                                                              mode, NB_round, NBF_round, outdir, debug)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("=== Métricas FFT2D vs numpy.fft.fft2 ===")
        for k, v in metrics.items():
            print(f"{k:>10}: {v}")
        print("Ops FFT2D:", ops_fft)
        print("Max value np.fft.fft2:", max_value_np)
        print("Int bits needed np.fft.fft2:", int_bits_needed_np)
        print(f"Elapsed time: ({elapsed:.2f} s)")
        print("")

    # ------------------------------ TEST 4 ------------------------------
    if selected in ("4", "all"):
        c4 = case("4")
        N  = int(c4.get("N"))
        NB  = int(c4.get("NB"))
        NBF  = int(c4.get("NBF"))
        outdir  = str(c4.get("outdir"))
        cast  = bool(c4.get("cast"))
        shift_right_stage  = bool(c4.get("shift_right_stage"))
        mode  = str(c4.get("mode"))
        NB_round  = int(c4.get("NB_round"))
        NBF_round  = int(c4.get("NBF_round"))
        debug  = bool(c4.get("debug"))

        print(f"========== COMPARISION 4: FFT2D de imagen de rodilla N = {N} ==========")
        t0 = time.perf_counter()
        metrics, ops_fft, max_value_np, int_bits_needed_np = run_comparision4(N, NB, NBF, cast, shift_right_stage, mode, NB_round,
                                                                              NBF_round, outdir, debug)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("=== Métricas FFT2D vs numpy.fft.fft2 ===")
        for k, v in metrics.items():
            print(f"{k:>10}: {v}")
        print("Ops FFT2D:", ops_fft)
        print("Max value np.fft.fft2:", max_value_np)
        print("Int bits needed np.fft.fft2:", int_bits_needed_np)
        print(f"Elapsed time: ({elapsed:.2f} s)")
        print("")


    # ------------------------------ TEST 5 ------------------------------
    if selected in ("5", "all"):
        c5 = case("5")
        N  = int(c5.get("N"))
        NB  = int(c5.get("NB"))
        NBF  = int(c5.get("NBF"))
        A = float(c5.get("A"))
        rings_period = float(c5.get("rings_period"))
        phase0 = float(c5.get("phase0"))
        outdir  = str(c5.get("outdir"))
        cast  = bool(c5.get("cast"))
        shift_right_stage  = bool(c5.get("shift_right_stage"))
        mode  = str(c5.get("mode"))
        NB_round  = int(c5.get("NB_round"))
        NBF_round  = int(c5.get("NBF_round"))
        debug  = bool(c5.get("debug"))

        print(f"========== COMPARISION 5: FFT2D e IFFT2D de concentric rings N = {N} ==========")
        t0 = time.perf_counter()
        metrics_ifft, ops_fft, ops_ifft = run_comparision5(
            N=N,
            NB=NB,
            NBF=NBF,
            A=A,
            rings_period=rings_period,
            phase0=phase0,
            cast=cast,
            shift_right_stage=shift_right_stage,
            mode=mode,
            NB_round=NB_round,
            NBF_round=NBF_round,
            outdir=outdir,
            debug=debug,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("=== Métricas IFFT2D fxp vs numpy.fft.ifft2 ===")
        for k, v in metrics_ifft.items():
            print(f"{k:>12}: {v}")
        print("Ops FFT2D:", ops_fft)
        print("Ops IFFT2D:", ops_ifft)
        print(f"Elapsed time: ({elapsed:.2f} s)")
        print("")


if __name__ == "__main__":
    main()
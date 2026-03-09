import argparse, math, sys, os, json, time
import matplotlib.pyplot as plt
from typing import List, Tuple
# ---------------------------------------------------------
#   FLIST
# ---------------------------------------------------------

PY_FLIST_FXP_MODEL = os.environ.get("PY_FLIST_FXP_MODEL")
PY_FLIST_FFT2D     = os.environ.get("PY_FLIST_FFT2D")

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


PY_FFT2D_CONFIG_TEST_FXP_JSON = os.environ.get("PY_FFT2D_CONFIG_TEST_FXP_JSON")

if not PY_FFT2D_CONFIG_TEST_FXP_JSON:
    raise RuntimeError(
        "[ERROR] PY_FFT2D_CONFIG_COMPARISION_JSON environment variable is missing"
    )
from stimulus_gen import gen_tone_cos_2d, gen_impulse_2d, gen_checkerboard_2d, gen_concentric_rings_2d, gen_gaussian_spots_2d
from fft2d import fft2d_norm
from helpers import _plot_fft2d_figure

# -------------------------------------------------------------------------------------
#   TEST 1: Generación y cuantización de single tone 2D
# -------------------------------------------------------------------------------------
def run_test1(Nx: int, Ny: int, kx0: int, ky0: int, NB: int, NBF: int, A: float, outdir: str) -> dict:
    """
    Genera una senoidal 2D, la cuantiza a CFxp2D, calcula métricas
    y guarda una figura comparando float vs cuantizada.

    Devuelve:
        metrics : dict con MSE, RMSE, MAE, MAX_ABS_ERR, SNR_dB, PSNR_dB
    """

    # img_fx: CFxp2D, img_f: np.ndarray (float)
    img_fx, img_f = gen_tone_cos_2d(Nx, Ny, kx0, ky0, NB, NBF, A=A)

    # imagen cuantizada en float
    img_q_float = img_fx.to_float()

    # métricas de cuantización respecto a la referencia float
    metrics = img_fx.quant_metrics(img_f)

    # --------- gráfica combinada ---------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(img_f, cmap="jet")
    axes[0].set_title("Float")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(img_q_float, cmap="jet")
    axes[1].set_title(f"Cuantizada S({NB},{NBF})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Senoidal 2D (Nx={Nx}, Ny={Ny}, kx0={kx0}, ky0={ky0}, A={A})")

    # Dejamos espacio abajo para el texto de métricas
    fig.subplots_adjust(bottom=0.25)

    # --------- texto de métricas en la figura ---------
    metric_keys = ["MSE", "RMSE", "MAE", "MAX_ABS_ERR", "SNR_dB", "PSNR_dB"]
    lines = []
    for k in metric_keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if math.isinf(v):
            sval = "inf"
        elif k.endswith("_dB"):
            sval = f"{v:.2f} dB"
        else:
            sval = f"{v:.3e}"
        lines.append(f"{k}: {sval}")

    metrics_text = "\n".join(lines)

    fig.text(
        0.5, 0.02, metrics_text,
        ha="center", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"sine2d_kx0_{kx0}_ky0_{ky0}_N_{Nx}.png"), dpi=300)
    plt.close(fig)

    return metrics

# -------------------------------------------------------------------------------------
#   TEST 2: FFT2D de una senoidal 2D magnitud
# -------------------------------------------------------------------------------------
def run_test2(
    N:         int,
    kx0:       float,
    ky0:       float,
    NB:        int,
    NBF:       int,
    A:         float,
    cast:      bool,
    mode:      str,
    NB_round:  int,
    NBF_round: int,
    debug:     bool,
    outdir:    str
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_tone_cos_2d(Nx, Ny, kx0, ky0, NB, NBF, A=A)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(outdir, f"sine2d_fft2d_N{N}_kx{kx0}_ky{ky0}.png")
    suptitle = (
        f"FFT2D senoidal 2D: N={N}, kx0={kx0}, ky0={ky0}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    # Solo magnitud → phase_fft=None
    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=None,
        title_prefix="Seno 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex

# -------------------------------------------------------------------------------------
#   TEST 3: FFT2D de una senoidal 2D magnitud y fase
# -------------------------------------------------------------------------------------
def run_test3(
    N:         int,
    kx0:       float,
    ky0:       float,
    NB:        int,
    NBF:       int,
    A:         float,
    cast:      bool,
    mode:      str,
    NB_round:  int,
    NBF_round: int,
    debug:     bool,
    outdir:    str,
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_tone_cos_2d(Nx, Ny, kx0, ky0, NB, NBF, A=A)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(
        outdir, f"sine2d_fft2d_mag_phase_N{N}_kx{kx0}_ky{ky0}.png"
    )
    suptitle = (
        f"FFT2D senoidal 2D: N={N}, kx0={kx0}, ky0={ky0}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=phase,
        title_prefix="Seno 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex

# -------------------------------------------------------------------------------------
#   TEST 4: FFT2D de un impulso 2D magnitud y fase
# -------------------------------------------------------------------------------------
def run_test4(
    N:         int,
    kx0:       float,
    ky0:       float,
    NB:        int,
    NBF:       int,
    A:         float,
    cast:      bool,
    mode:      str,
    NB_round:  int,
    NBF_round: int,
    debug:     bool,
    outdir:    str,
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_impulse_2d(Nx, Ny, NB, NBF, A, kx0, ky0)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(
        outdir, f"impulse2d_fft2d_mag_phase_N{N}_kx{kx0}_ky{ky0}.png"
    )
    suptitle = (
        f"FFT2D impulso 2D: N={N}, kx0={kx0}, ky0={ky0}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=phase,
        title_prefix="Impulso 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex

# -------------------------------------------------------------------------------------
#   TEST 5: FFT2D de checkerboard 2D magnitud y fase
# -------------------------------------------------------------------------------------
def run_test5(
    N:         int,
    period_x:  int,
    period_y:  int,
    NB:        int,
    NBF:       int,
    A:         float,
    cast:      bool,
    mode:      str,
    NB_round:  int,
    NBF_round: int,
    debug:     bool,
    outdir:    str,
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_checkerboard_2d(Nx, Ny, NB, NBF, A, period_x, period_y)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(outdir, f"checkerboard_fft2d_mag_phase_N{N}.png")
    suptitle = (
        f"FFT2D checkerboard 2D: N={N}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=phase,
        title_prefix="Checkerboard 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex

# -------------------------------------------------------------------------------------
#   TEST 6: FFT2D de spatial pattern anillos concentricos bar 2D magnitud y fase
# -------------------------------------------------------------------------------------
def run_test6(
    N:              int,
    rings_period:   float,
    phase0:         int,
    NB:             int,
    NBF:            int,
    A:              float,
    cast:           bool,
    mode:           str,
    NB_round:       int,
    NBF_round:      int,
    debug:          bool,
    outdir:         str,
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_concentric_rings_2d(Nx, Ny, NB, NBF, A, rings_period, phase0)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(outdir, f"rings_fft2d_mag_phase_N{N}.png")
    suptitle = (
        f"FFT2D rings 2D: N={N}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=phase,
        title_prefix="Rings 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex

# -------------------------------------------------------------------------------------
#   TEST 7: FFT2D de gaussian spots magnitud y fase
# -------------------------------------------------------------------------------------
def run_test7(
    N:              int,
    variance:       float,
    centers:        List[Tuple[int, int]],
    NB:             int,
    NBF:            int,
    A:              float,
    cast:           bool,
    mode:           str,
    NB_round:       int,
    NBF_round:      int,
    debug:          bool,
    outdir:         str,
):
    Nx = Ny = N
    os.makedirs(outdir, exist_ok=True)

    img_q, img_f = gen_gaussian_spots_2d(Nx, Ny, NB, NBF, variance, centers, A)

    mag_log, phase, total_ops, int_bits_needed, max_complex = fft2d_norm(
        img_q, img_f, N, NB, NBF,
        cast, mode, NB_round, NBF_round, debug
    )

    out_name = os.path.join(outdir, f"gaussian_spots_fft2d_mag_phase_N{N}.png")
    suptitle = (
        f"FFT2D gaussian spots 2D: N={N}, "
        f"S({NB},{NBF}), cast={cast}, mode={mode}"
    )

    _plot_fft2d_figure(
        img_f=img_f,
        mag_fft_log=mag_log,
        phase_fft=phase,
        title_prefix="Gaussian spots 2D",
        suptitle=suptitle,
        out_path=out_name,
    )

    return total_ops, int_bits_needed, max_complex


def main():
    parser = argparse.ArgumentParser(
        description="Tests de FFT2D en punto fijo"
    )
    parser.add_argument(
        "flist",
        help="Ruta al archivo .f (flist) con las rutas a los modelos",
    )
    parser.parse_args()

    with open(PY_FFT2D_CONFIG_TEST_FXP_JSON, "r") as f:
        cfg_all = json.load(f)

    fft1_root  = cfg_all["fft2d"]
    selected   = str(fft1_root.get("test", "1"))
    fft1_cases = fft1_root.get("cases", {})

    def case(tid: str) -> dict:
        """Devuelve el diccionario de parámetros para el test tid."""
        cfg = fft1_cases.get(tid, {})
        if not cfg:
            print(f"[WARNING] No hay config para test {tid} en JSON, usando defaults.")
        return cfg

    # ---------- TEST 1 ----------
    if selected in ("1", "all"):
        c1 = case("1")
        Nx  = int(c1.get("Nx"))
        Ny  = int(c1.get("Ny"))
        kx0  = int(c1.get("kx0"))
        ky0  = int(c1.get("ky0"))
        NB  = int(c1.get("NB"))
        NBF  = int(c1.get("NBF"))
        A  = float(c1.get("A"))
        outdir  = str(c1.get("output_dir"))
        print(f"========== TEST 1: tono real 2D ==========")
        run_test1(Nx, Ny, kx0, ky0, NB, NBF, A, outdir)
        print("")

    # ---------- TEST 2 ----------
    if selected in ("2", "all"):
        c2 = case("2")
        N  = int(c2.get("N"))
        kx0  = int(c2.get("kx0"))
        ky0  = int(c2.get("ky0"))
        NB  = int(c2.get("NB"))
        NBF  = int(c2.get("NBF"))
        A  = float(c2.get("A"))
        cast = bool(c2.get("cast"))
        mode = str(c2.get("mode"))
        NB_round = int(c2.get("NB_round"))
        NBF_round = int(c2.get("NBF_round"))
        debug  = bool(c2.get("debug"))
        outdir  = str(c2.get("output_dir"))

        print("========== TEST 2: FFT2D fxp magnitud ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test2(
            N, kx0, ky0, NB, NBF, A,
            cast, mode, NB_round, NBF_round, debug, outdir
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")



    if selected in ("3", "all"):
        c3 = case("3")
        N  = int(c3.get("N"))
        kx0  = int(c3.get("kx0"))
        ky0  = int(c3.get("ky0"))
        NB  = int(c3.get("NB"))
        NBF  = int(c3.get("NBF"))
        A  = float(c3.get("A"))
        cast = bool(c3.get("cast"))
        mode = str(c3.get("mode"))
        NB_round = int(c3.get("NB_round"))
        NBF_round = int(c3.get("NBF_round"))
        debug  = bool(c3.get("debug"))
        outdir  = str(c3.get("output_dir"))

        print("========== TEST 3: FFT2D seno fxp magnitud y fase ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test3(
            N, kx0, ky0, NB, NBF, A,
            cast, mode, NB_round, NBF_round, debug, outdir
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")

    if selected in ("4", "all"):
        c4 = case("4")
        N  = int(c4.get("N"))
        kx0  = int(c4.get("kx0"))
        ky0  = int(c4.get("ky0"))
        NB  = int(c4.get("NB"))
        NBF  = int(c4.get("NBF"))
        A  = float(c4.get("A"))
        cast = bool(c4.get("cast"))
        mode = str(c4.get("mode"))
        NB_round = int(c4.get("NB_round"))
        NBF_round = int(c4.get("NBF_round"))
        debug  = bool(c4.get("debug"))
        outdir  = str(c4.get("output_dir"))

        print("========== TEST 4: FFT2D impulso fxp magnitud y fase ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test4(
            N, kx0, ky0, NB, NBF, A,
            cast, mode, NB_round, NBF_round, debug, outdir
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")

    if selected in ("5", "all"):
        c5 = case("5")
        N  = int(c5.get("N"))
        period_x  = int(c5.get("period_x"))
        period_y  = int(c5.get("period_y"))
        NB  = int(c5.get("NB"))
        NBF  = int(c5.get("NBF"))
        A  = float(c5.get("A"))
        cast = bool(c5.get("cast"))
        mode = str(c5.get("mode"))
        NB_round = int(c5.get("NB_round"))
        NBF_round = int(c5.get("NBF_round"))
        debug  = bool(c5.get("debug"))
        outdir  = str(c5.get("output_dir"))

        print("========== TEST 5: FFT2D checkerboard fxp magnitud y fase ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test5(
            N, period_x, period_y, NB, NBF, A,
            cast, mode, NB_round, NBF_round, debug, outdir
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")

    if selected in ("6", "all"):
        c6 = case("6")
        N  = int(c6.get("N"))
        rings_period  = float(c6.get("rings_period"))
        phase0  = int(c6.get("phase0"))
        NB  = int(c6.get("NB"))
        NBF  = int(c6.get("NBF"))
        A  = float(c6.get("A"))
        cast = bool(c6.get("cast"))
        mode = str(c6.get("mode"))
        NB_round = int(c6.get("NB_round"))
        NBF_round = int(c6.get("NBF_round"))
        debug  = bool(c6.get("debug"))
        outdir  = str(c6.get("output_dir"))


        print("========== TEST 6: FFT2D concentric rings fxp magnitud y fase ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test6(N, rings_period, phase0, NB, NBF, 
                                                            A, cast, mode, NB_round, NBF_round, debug, outdir)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")

    if selected in ("7", "all"):
        c7 = case("7")
        N  = int(c7.get("N"))
        centers_raw = c7.get("centers", [])
        centers: List[Tuple[int, int]] = [
            (int(cx), int(cy)) for cx, cy in centers_raw
        ]
        variance  = float(c7.get("variance"))
        NB  = int(c7.get("NB"))
        NBF  = int(c7.get("NBF"))
        A  = float(c7.get("A"))
        cast = bool(c7.get("cast"))
        mode = str(c7.get("mode"))
        NB_round = int(c7.get("NB_round"))
        NBF_round = int(c7.get("NBF_round"))
        debug  = bool(c7.get("debug"))
        outdir  = str(c7.get("output_dir"))


        print("========== TEST 7: FFT2D gaussian spots fxp magnitud y fase ==========")
        t0 = time.perf_counter()
        total_ops, int_bits_needed, max_complex = run_test7(N, variance, centers, NB, NBF, 
                                                            A, cast, mode, NB_round, NBF_round, debug, outdir)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Max Value in FFT2D fxp:", max_complex)
        print("Int bits needed:", int_bits_needed)
        print("Ops FFT2D fxp:", total_ops)
        print(f"Elapsed time: ({elapsed*1000:.1f} ms)")
        

if __name__ == "__main__":
    main()





from cfxp import CFxp
from typing import Dict
import sys, os, math, numpy as np, matplotlib.pyplot as plt

def load_skm_tea(outpath):
        
    if len(sys.argv) < 2:
        raise RuntimeError("[ERROR] Flist specification is missing")

    flist_path = sys.argv[1]

    with open(flist_path) as f:
        paths = [line.strip() for line in f if line.strip()]

    path_py = [r for r in paths if r.endswith(".py")]

    for pt in path_py:
        folder = os.path.dirname(
            os.path.abspath(os.path.join(os.path.dirname(flist_path), pt))
        )
        if folder not in sys.path:
            sys.path.append(folder)



def _debug_stage_bits(x: list[CFxp], s: int, stages: int):
    """
    Imprime estadísticas de rango y 'bits necesarios' para la parte real/imag
    después del stage s.
    """
    # asumimos que todos los CFxp de x tienen mismo formato
    p0 = x[0]
    NB  = p0.re.NB
    NBF = p0.re.NBF
    int_bits_avail = NB - NBF - 1  # bits enteros útiles (sin el bit de signo)

    max_re = 0.0
    max_im = 0.0

    for z in x:
        c = z.to_complex()
        max_re = max(max_re, abs(c.real))
        max_im = max(max_im, abs(c.imag))

    max_val = max(max_re, max_im)

    if max_val > 0.0:
        # bits enteros necesarios (incluyendo bit de signo)
        bits_needed = math.ceil(math.log2(max_val)) + 1
        if bits_needed < 1:
            bits_needed = 1
    else:
        bits_needed = 1


    print(
        f"[FFT][stage {s}/{stages}] "
        f"NB={NB}, NBF={NBF}, int_bits_avail={int_bits_avail}, "
        f"max(re)={max_re:.6g}, max(im)={max_im:.6g}, "
        f"int_bits_needed≈{bits_needed-1}"
    )

def _accum_ops(total: Dict[str, int], add: Dict[str, int]) -> None:
    for k in total.keys():
        total[k] += add.get(k, 0)


def _calculate_stages(N: int) -> int:
    assert (N & (N - 1)) == 0, f"N debe ser potencia de 2"
    return int(math.log2(N))

def print_ops(N, ops_fft, ops_dft):
    print(f"\nResumen de operaciones (N={N}):")
    print(f"{'op':>8} | {'cmul':>10} | {'cadd/sub':>10} | {'mul':>10} | {'add/sub':>12}")
    print("-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 12)

    print(f"{'FFT':>8} | "
          f"{ops_fft['cmul']:10d} | "
          f"{ops_fft['caddsub']:10d} | "
          f"{ops_fft['mul']:10d} | "
          f"{ops_fft['addsub']:12d}")

    print(f"{'DFT':>8} | "
          f"{ops_dft['cmul']:10d} | "
          f"{ops_dft['caddsub']:10d} | "
          f"{ops_dft['mul']:10d} | "
          f"{ops_dft['addsub']:12d}")

def print_comparison(X_fft_fxp, X_dft, X_np):
    N = len(X_fft_fxp)

    # cabecera
    print(f"{'k':>3} | {'FFT fxp':>23} | {'DFT ref':>23} | {'numpy.fft':>23}")
    print("-" * 3 + "-+-" + "-" * 23 + "-+-" + "-" * 23 + "-+-" + "-" * 23)

    # filas
    for k in range(N):
        a = X_fft_fxp[k].to_complex()
        b = X_dft[k]
        c = X_np[k]
        print(f"{k:3d} | "
              f"{a.real:+10.6f}{a.imag:+10.6f}j | "
              f"{b.real:+10.6f}{b.imag:+10.6f}j | "
              f"{c.real:+10.6f}{c.imag:+10.6f}j")

def _plot_fft2d_figure(
    img_f: np.ndarray,
    mag_fft_log: np.ndarray,
    phase_fft: np.ndarray | None,
    title_prefix: str,
    suptitle: str,
    out_path: str,
):
    """
    Genera la figura de 2 paneles (espacio + magnitud) o 3 paneles (sumando fase),
    según si phase_fft es None o no.
    """
    if phase_fft is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax_space, ax_mag = axes
        ax_phase = None
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax_space, ax_mag, ax_phase = axes

    # 1) Espacio
    im0 = ax_space.imshow(img_f, cmap="jet")
    ax_space.set_title("Espacio (float)")
    ax_space.axis("off")
    fig.colorbar(im0, ax=ax_space, fraction=0.046, pad=0.04)

    # 2) Magnitud log(1+|X|)
    im1 = ax_mag.imshow(mag_fft_log, cmap="jet")
    ax_mag.set_title("Magnitud log(1+|X|)")
    ax_mag.axis("off")
    fig.colorbar(im1, ax=ax_mag, fraction=0.046, pad=0.04)

    # 3) Fase
    if ax_phase is not None and phase_fft is not None:
        im2 = ax_phase.imshow(phase_fft, cmap="jet")
        ax_phase.set_title("Fase ∠X [rad]")
        ax_phase.axis("off")
        fig.colorbar(im2, ax=ax_phase, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)






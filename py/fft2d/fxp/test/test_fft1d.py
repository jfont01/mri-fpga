import argparse, math, sys, os, json, matplotlib.pyplot as plt, numpy as np

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

from cfxp import CFxp
from fft1d import build_twiddles, fft, ifft, fft_norm
from dft import dft, np_fft
from helpers import print_ops, print_comparison
from stimulus_gen import gen_impulse_1d, gen_tone_complex_1d, gen_tone_cos_real_1d, gen_sine_real_1d, gen_two_sines_real_hz_1d, gen_tone_cos_2d



# ---------------------------------------------------------
#   TEST 1: Impulso
# ---------------------------------------------------------

def run_test1(N):
    NB = 6
    NBF = 4

    W = build_twiddles(N, NB, NBF)

    print("Twiddles:")
    for n, wn in enumerate(W):
        c = wn.to_complex()
        print(f"W[{n:2d}] = {c.real:+.6f}{c.imag:+.6f}j : {wn}")

    print("Twiddles simétricos:")
    for n in range(N // 2):
        assert W[n].to_complex() == -W[n + N // 2].to_complex()
        print(f"W[{n:2d}] = -W[{n + N//2:2d}]")


    print("")

    x_q = gen_impulse_1d(N, NB, NBF)
    x_f = [x.to_complex() for x in x_q]

    

    print("Input:")
    for n, xn in enumerate(x_q):
        c = xn.to_complex()
        print(f"x[{n:2d}] = {c.real:+.6f}{c.imag:+.6f}j : {xn}")
    print("")

    X_fft, ops_fft = fft(x_q, W, cast=True, NB_round=16, NBF_round=12, debug=True, shift_right_stage=False)
    X_dft, ops_dft = dft(x_f, debug=True)
    X_np = np_fft(x_f)
    print_comparison(X_fft, X_dft, X_np)
    print_ops(N, ops_fft, ops_dft)

# -------------------- TEST 2: tono complejo ---------------------------------
def run_test2(N):
    k0 = 2
    NB = 16
    NBF = 11

    W = build_twiddles(N, NB, NBF)

    x_q = gen_tone_complex_1d(N, k0, NB, NBF, A=1.0)
    x_f = [z.to_complex() for z in x_q]


    print("Input:")
    for n, xn in enumerate(x_q):
        c = xn.to_complex()
        print(f"x[{n:2d}] = {c.real:+.6f}{c.imag:+.6f}j : {xn}")
    print("")

    X_fft, ops_fft = fft(x_q, W, cast=True, NB_round=NB, NBF_round=NBF, debug=True, shift_right_stage=False)
    X_dft, ops_dft = dft(x_f, debug=True)
    X_np = np_fft(x_f)
    print_comparison(X_fft, X_dft, X_np)
    print_ops(N, ops_fft, ops_dft)

# -------------------- TEST 3: coseno real -----------------------------------
def run_test3(N):
    k0 = 2
    NB = 16
    NBF = 11

    W = build_twiddles(N, NB, NBF)

    x_q = gen_tone_cos_real_1d(N, k0, NB, NBF, A=1.0)
    x_f = [z.to_complex() for z in x_q]


    print("Input:")
    for n, xn in enumerate(x_q):
        c = xn.to_complex()
        print(f"x[{n:2d}] = {c.real:+.6f}{c.imag:+.6f}j : {xn}")
    print("")

    X_fft, ops_fft = fft(x_q, W, cast=True, NB_round=NB, NBF_round=NBF, debug=True, shift_right_stage=False)
    X_dft, ops_dft = dft(x_f, debug=True)
    X_np = np_fft(x_f)
    print_comparison(X_fft, X_dft, X_np)
    print_ops(N, ops_fft, ops_dft)

# -------------------- TEST 4: senoidal real con gráficos --------------------
def run_test4(N=256, k0=2, NB=16, NBF=11, A=0.05, outdir="."):
    # asegurar que exista el directorio
    os.makedirs(outdir, exist_ok=True)

    # 1) generar señal
    x_q, x_f = gen_sine_real_1d(N, k0, NB, NBF, A=A)

    # 2) twiddles cuantizados
    W = build_twiddles(N, NB, NBF)

    # 3) FFT en fijo
    X_fft_q, ops_fft = fft(
        x_q, W,
        cast=True,
        NB_round=NB,
        NBF_round=NBF,
        debug=True,
    )
    X_fft = np.array([z.to_complex() for z in X_fft_q])

    # 4) FFT de referencia con numpy y DFT
    X_np = np.fft.fft(np.array(x_f, dtype=np.complex128))
    X_dft, ops_dft = dft(x_f, debug=True)

    # 5) magnitud y fase
    Npoints   = len(x_f)
    k         = np.arange(Npoints)
    mag_fft   = np.abs(X_fft)
    phase_fft = np.angle(X_fft)

    n = np.arange(Npoints)

    # ---------- Figura resumen con 3 subplots ----------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Señal en el tiempo
    axes[0].stem(n, x_f)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("x[n]")
    axes[0].set_title(f"Senoidal real cuantizada (N={N}, k0={k0}, A={A})")

    # Magnitud
    axes[1].stem(k, mag_fft)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("|X[k]|")
    axes[1].set_title("Espectro de magnitud (FFT punto fijo)")

    # Fase
    axes[2].stem(k, phase_fft)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("∠X[k] [rad]")
    axes[2].set_title("Espectro de fase (FFT punto fijo)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f"Resumen FFT 1D (N={N}, k0={k0}, S({NB},{NBF}))")

    fig.savefig(os.path.join(outdir, f"sine_fft_summary_N{N}_k{k0}.png"), dpi=300)
    plt.close(fig)

    # prints de comparación numérica
    print_comparison(X_fft_q, X_dft, X_np)
    print_ops(N, ops_fft, ops_dft)


# -------------------- TEST 5: sanity test de ifft ---------------------------
def run_test5(N, NB, NBF):

    # señal cualquiera
    x_q = [CFxp.quantize(math.sin(2*math.pi*3*n/N), 0.0, NB, NBF) for n in range(N)]
    x_f = [z.to_complex() for z in x_q]

    W = build_twiddles(N, NB, NBF)

    X_fft, _ = fft(x_q, W, cast=True, NB_round=NB, NBF_round=NBF, debug=False)
    x_rec, _ = ifft(X_fft, W, cast=True, NB_round=NB, NBF_round=NBF, debug=False)

    print("n | x original       | x rec ifft")
    for n in range(N):
        print(f"{n:2d} | {x_f[n].real:+.6f} | {x_rec[n].to_complex().real:+.6f}")

# -------------------- TEST 6: fft vector test hz ----------------------------
def run_test6(N: int, Fs: float, f0: float, f1: float, A0: float , A1: float, NB: int, NBF: int, outdir: str = "."):

    x_f, x_q, time = gen_two_sines_real_hz_1d(N, Fs, f0, f1, A0, A1, NB, NBF)

    W = build_twiddles(N, NB, NBF, 'round')

    t, xq_f, hz, amps, phases  = fft_norm(N, Fs, x_q, time, W, NB, NBF, cast=True, debug=True)


    # ==== Figura con 3 gráficas ====
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))

    # 1) Señal en el tiempo (cuantizada)
    axs[0].plot(t, xq_f)
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("x(t)")
    axs[0].set_title(f"Señal cuantizada S({NB},{NBF}), Fs = {Fs}")

    # 2) Magnitud del espectro
    axs[1].stem(hz, amps)
    axs[1].set_xlabel("Hz")
    axs[1].set_ylabel("Amplitud")
    axs[1].set_title("Espectro de magnitud")

    # 3) Fase del espectro
    axs[2].stem(hz, phases)
    axs[2].set_xlabel("Hz")
    axs[2].set_ylabel("Fase [rad]")
    axs[2].set_title("Espectro de fase")

    fig.suptitle(
        f"Dos senoidales sumadas: f0={f0} Hz, f1={f1} Hz, "
        f"A0={A0}, A1={A1}, N={N}, Fs={Fs}",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(outdir, "fft_norm.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description="Tests de FFT radix-2 en punto fijo (flist + JSON de config)"
    )
    parser.add_argument(
        "flist",
        help="Ruta al archivo .f (flist) con las rutas a los modelos",
    )
    parser.parse_args()

    with open(PY_FFT2D_CONFIG_TEST_FXP_JSON, "r") as f:
        cfg_all = json.load(f)

    fft1_root  = cfg_all["fft1d"]
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
        N  = int(c1.get("N", 4))
        print(f"========== TEST 1: impulso N={N} ==========")
        run_test1(N)
        print("")

    # ---------- TEST 2 ----------
    if selected in ("2", "all"):
        c2 = case("2")
        N  = int(c2.get("N", 4))
        print(f"========== TEST 2: tono complejo N={N} ==========")
        run_test2(N)
    
    # ---------- TEST 3 ----------
    if selected in ("3", "all"):
        c3 = case("3")
        N  = int(c3.get("N", 4))
        print(f"========== TEST 3: coseno real N={N} ==========")
        run_test3(N)

    # ---------- TEST 4 ----------
    if selected in ("4", "all"):
        c4         = case("4")
        N          = int(c4.get("N", 256))
        k0         = int(c4.get("k0", 2))
        NB         = int(c4.get("NB", 16))
        NBF        = int(c4.get("NBF", 11))
        A          = float(c4.get("A", 0.05))
        output_dir = c4.get("output_dir", "plots_fft1d_t4")
        print(f"========== TEST 4: seno real con graficas N={N}, k0={k0} ==========")
        run_test4(N=N, k0=k0, NB=NB, NBF=NBF, A=A, outdir=output_dir)

    # ---------- TEST 5 ----------
    if selected in ("5", "all"):
        c5   = case("5")
        N    = int(c5.get("N", 16))
        NB   = int(c5.get("NB", 16))
        NBF  = int(c5.get("NBF", 11))
        print(f"========== TEST 5: ifft sanity test N={N}, NB={NB}, NBF={NBF} ==========")
        run_test5(N, NB, NBF)

    # ---------- TEST 6 ----------
    if selected in ("6", "all"):
        c6         = case("6")
        N          = int(c6.get("N"))
        NB         = int(c6.get("NB"))
        NBF        = int(c6.get("NBF"))
        A0         = float(c6.get("A0"))
        A1         = float(c6.get("A1"))
        Fs         = float(c6.get("Fs"))
        f0         = float(c6.get("f0"))
        f1         = float(c6.get("f1"))
        output_dir = c6.get("output_dir", "plots_fft1d")
        print(f"========== TEST 6: fft hz vector test N={N}, NB={NB}, NBF={NBF} ==========")
        run_test6(N, Fs, f0, f1, A0, A1, NB, NBF, output_dir)


if __name__ == "__main__":
    main()



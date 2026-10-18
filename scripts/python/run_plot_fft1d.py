#!/usr/bin/env python3
"""
run_plot_fft1d.py

Runner de ploteo para los modulos fft1d_* (r2, r2sdf, r22sdf). Se ejecuta SIN
argumentos desde el directorio de un modulo (igual que run_gtest.py /
run_regression_sim.py): descubre solo el modulo, lee su JSON de regresion,
ubica los .dat de cada caso y genera una figura por caso con:

  1. entrada x[n]
  2. |X[k]|
  3. fase de X[k]
  4. error contra la FFT ideal, en LSB de salida (SQNR, sesgo y maximo)

Descubrimiento automatico
--------------------------
  modulo   : via MODULES_ROOT + cwd  (detect_current_module, igual que el resto)
  JSON     : <PREFIX>_TB_REGRESSION_JSON  o  <PREFIX>_REGRESSION_JSON  o
             <module_dir>/testbench/<module>_tb_regression.json   (mismo orden
             de prioridad que run_regression_sim.py)
  casos    : campo "cases" del JSON
  .dat     : modules/<module>/build/<CASE>/simulation/vectors/
                 stimuli/in_ports/{i_re,i_im,i_valid}.dat
                 expected/out_ports/{o_re,o_im,o_valid}.dat   (--src expected, default)
                 actual/out_ports/{o_re,o_im,o_valid}.dat      (--src actual)
  salida   : modules/<module>/build/<CASE>/plots/<CASE>_<src>.png

Parametros por caso (auto)
--------------------------
  N          : del JSON (clave "N"); si falta, se infiere de la cantidad de muestras.
  Formatos   : NB/NBF del datapath (default 16/15) y, si el caso los define,
               NB_IN/NBF_IN (puerto de entrada) y NB_OUT/NBF_OUT (puerto de
               salida). Misma regla de defaults que el RTL y el modelo: el ancho
               por defecto es NB y los bits fraccionales por defecto heredan los
               bits enteros del datapath, NBF_x = NB_x - (NB - NBF).
               Los .dat de entrada estan en Q(NB_IN, NBF_IN) y los de salida en
               Q(NB_OUT, NBF_OUT), NO en el formato del datapath.
  SCALE_MASK : (default: todas las etapas escalan). La salida es la FFT por
               2^-S, con S = popcount(SCALE_MASK & (N-1)).
  K0         : solo para marcar el bin esperado. En este orden: sufijo "_k<N>"
               del nombre del caso; "k0=..." del sim_summary.rpt; STIM_K0 del
               JSON si STIM_TYPE == 0 (tono/multitono); si no, None.

Orden de la salida
------------------
  Con la referencia ideal disponible, el orden (natural o bit-reversed) se
  detecta comparando la salida contra la FFT ideal en los dos ordenes: anda
  con cualquier estimulo, no solo con tonos. PLOT_REORDER del JSON (o
  --reorder) sigue pudiendo forzarlo.

Uso
---
  python3 run_plot_fft1d.py                 # todos los casos, fuente expected
  python3 run_plot_fft1d.py --src actual    # plotea la salida del RTL
  python3 run_plot_fft1d.py --only n64_q1_15_sine_k3
  python3 run_plot_fft1d.py --src both      # una figura por expected y por actual
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reusa helpers del proyecto (mismo import que los otros runners)
from regression_common import (
    detect_current_module,
    load_json,
    resolve_path,
    extract_params,
    banner,
    kv,
)

META_KEYS = {"name", "enabled", "params", "defines", "cpp_defines",
             "sim_defines", "auto_defines", "binary_file"}


# ----------------------------------------------------------------------
# Lectura .dat crudos  (identico criterio que fixed_to_real del testbench)
# ----------------------------------------------------------------------
def load_dat_raw(path: Path) -> np.ndarray:
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line))
    return np.asarray(vals, dtype=np.int64)


def raw_to_real(raw: np.ndarray, nbf: int) -> np.ndarray:
    return raw.astype(np.float64) / float(2 ** nbf)


def load_complex(d: Path, re_name: str, im_name: str, nbf: int):
    re = raw_to_real(load_dat_raw(d / re_name), nbf)
    im = raw_to_real(load_dat_raw(d / im_name), nbf)
    n = min(len(re), len(im))
    return re[:n] + 1j * im[:n]


def filter_valid(x, valid_path: Path):
    if not valid_path.is_file():
        return x
    v = load_dat_raw(valid_path)
    m = min(len(v), len(x))
    return x[:m][v[:m].astype(bool)]


# ----------------------------------------------------------------------
# Formatos y escalado del caso
# ----------------------------------------------------------------------
def as_int(value, default):
    """Acepta enteros o strings ("784", "0x310")."""
    if value is None:
        return default
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def port_formats(params):
    """
    (NB, NBF), (NB_IN, NBF_IN), (NB_OUT, NBF_OUT) del caso, con la misma regla
    de defaults que fft1d_r22sdf.v / fft1d_r22sdf.hpp.
    """
    nb  = as_int(params.get("NB"), 16)
    nbf = as_int(params.get("NBF"), 15)
    nbi = nb - nbf                                   # bits enteros del datapath

    nb_in   = as_int(params.get("NB_IN"), nb)
    nbf_in  = as_int(params.get("NBF_IN"), nb_in - nbi)
    nb_out  = as_int(params.get("NB_OUT"), nb)
    nbf_out = as_int(params.get("NBF_OUT"), nb_out - nbi)

    return (nb, nbf), (nb_in, nbf_in), (nb_out, nbf_out)


def scale_stages(params, n):
    """S = cantidad de etapas que escalan x1/2. Default: todas (escala 1/N)."""
    log2n = int(round(np.log2(n)))
    mask = as_int(params.get("SCALE_MASK"), (1 << 32) - 1)
    return bin(mask & ((1 << log2n) - 1)).count("1"), mask


def qfmt(nb, nbf):
    return f"Q{nb - nbf}.{nbf}"


# ----------------------------------------------------------------------
# Bit-reversal y deteccion de orden
# ----------------------------------------------------------------------
def bitrev_index(i: int, log2n: int) -> int:
    r = 0
    for _ in range(log2n):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def bitrev_permute(x):
    n = len(x)
    log2n = int(round(np.log2(n)))
    idx = np.array([bitrev_index(i, log2n) for i in range(n)])
    return x[idx]


def peak_info(X):
    mag = np.abs(X)
    peak = int(np.argmax(mag))
    purity = (mag[peak] ** 2) / (np.sum(mag ** 2) + 1e-30)
    return {"peak_index": peak, "purity": purity}


def detect_order_by_reference(X_raw, X_ref):
    """
    Compara la salida cruda contra la FFT ideal en orden natural y en orden
    bit-reversed. Devuelve el orden con menor error; "indistinguible" si el
    espectro ideal es invariante a la permutacion; "desconocido" si ninguno de
    los dos es claramente mejor.
    """
    X_br = bitrev_permute(X_ref)
    scale = np.max(np.abs(X_ref)) + 1e-30
    if np.max(np.abs(X_ref - X_br)) <= 1e-9 * scale:
        # Espectro invariante a la permutacion (p. ej. impulso en n=0, DC):
        # los dos ordenes dan lo mismo y el error se puede medir igual.
        return "indistinguible"
    e_nat = np.sum(np.abs(X_raw - X_ref) ** 2)
    e_br  = np.sum(np.abs(X_raw - X_br) ** 2)
    lo, hi = min(e_nat, e_br), max(e_nat, e_br)
    if lo > 0.5 * hi:
        return "desconocido"
    return "natural" if e_nat < e_br else "bitrev"


# ----------------------------------------------------------------------
# Extraccion automatica de K0 (solo para marcar el bin esperado)
# ----------------------------------------------------------------------
def k0_from_case_name(name: str):
    m = re.search(r"_k(\d+)\b", name)
    return int(m.group(1)) if m else None


def k0_from_report(case_dir: Path):
    rpt = case_dir / "reports" / "sim_summary.rpt"
    if not rpt.is_file():
        return None
    try:
        txt = rpt.read_text(errors="replace")
        m = re.search(r"k0\s*=\s*(\d+)", txt)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def k0_from_params(params):
    # Solo tono/multitono (STIM_TYPE 0): STIM_K0 es el tono principal.
    if as_int(params.get("STIM_TYPE"), -1) == 0 and "STIM_K0" in params:
        return as_int(params.get("STIM_K0"), None)
    return None


def resolve_k0(case_name: str, case_dir: Path, params, override):
    if override is not None:
        return override, "cli"
    k = k0_from_case_name(case_name)
    if k is not None:
        return k, "nombre_caso"
    k = k0_from_report(case_dir)
    if k is not None:
        return k, "sim_summary.rpt"
    k = k0_from_params(params)
    if k is not None:
        return k, "STIM_K0"
    return None, "no_encontrado"


# ----------------------------------------------------------------------
# Error contra la referencia
# ----------------------------------------------------------------------
def error_stats(X_out, X_ref, lsb):
    """Error en LSB de salida, SQNR (dB), sesgo por componente y maximo."""
    e = (X_out - X_ref) / lsb
    p_sig = np.sum(np.abs(X_ref) ** 2)
    p_err = np.sum(np.abs(X_out - X_ref) ** 2)
    if p_err == 0.0:
        sqnr = np.inf
    elif p_sig == 0.0:
        sqnr = -np.inf
    else:
        sqnr = 10.0 * np.log10(p_sig / p_err)
    return {
        "e": e,
        "sqnr_db": sqnr,
        "bias_re": float(np.mean(e.real)),
        "bias_im": float(np.mean(e.imag)),
        "rms": float(np.sqrt(np.mean(np.abs(e) ** 2))),
        "max": float(np.max(np.abs(e))),
    }


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_figure(x_in, X_out, N, order, info, title, out_png, k0=None, err=None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), constrained_layout=True)

    ax = axes[0]
    ni = np.arange(len(x_in))
    ax.plot(ni, x_in.real, marker=".", ms=4, lw=1, label="Re x[n]")
    ax.plot(ni, x_in.imag, marker=".", ms=4, lw=1, label="Im x[n]", alpha=0.8)
    ax.set_title("Entrada  x[n]  (dominio del tiempo)")
    ax.set_xlabel("indice de muestra  n"); ax.set_ylabel("amplitud")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    k = np.arange(len(X_out)); mag = np.abs(X_out)
    ax.stem(k, mag, basefmt=" ")
    p = info["peak_index"]
    ax.plot(p, mag[p], "rv", ms=9, label=f"pico k={p}")
    if k0 is not None:
        ax.axvline(k0 % N, color="green", ls="--", lw=1, alpha=0.7,
                   label=f"k0={k0 % N} (esperado)")
    ax.set_title(f"FFT  |X[k]|   —   orden: {order}   (pureza pico={info['purity']:.3f})")
    ax.set_xlabel("indice de frecuencia  k"); ax.set_ylabel("|X[k]|")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    phase = np.angle(X_out)
    thr = 0.02 * mag.max() if mag.max() > 0 else 0
    m = mag > thr
    ax.plot(k, phase, color="0.8", lw=1, zorder=1)
    ax.plot(k[m], phase[m], "o", ms=5, color="tab:purple", zorder=2,
            label=f"fase donde |X|>{thr:.3g}")
    ax.set_title("FFT  fase  ∠X[k]  (rad)")
    ax.set_xlabel("indice de frecuencia  k"); ax.set_ylabel("fase [rad]")
    ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    if err is None:
        ax.text(0.5, 0.5, f"sin panel de error: la salida ploteada esta en orden '{order}' "
                          "(el error se mide en orden natural)",
                ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
    else:
        e = err["e"]
        ax.plot(k, e.real, ".", ms=3, label="Re e[k]")
        ax.plot(k, e.imag, ".", ms=3, label="Im e[k]", alpha=0.7)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(err["bias_re"], color="tab:blue", ls="--", lw=1,
                   label=f"media Re = {err['bias_re']:+.3f}")
        ax.axhline(err["bias_im"], color="tab:orange", ls="--", lw=1,
                   label=f"media Im = {err['bias_im']:+.3f}")
        sq = err["sqnr_db"]
        sq_txt = "inf" if np.isinf(sq) and sq > 0 else f"{sq:.1f}"
        ax.set_title(f"Error vs FFT ideal  (X_ideal = FFT(x)·2^-S, misma entrada)   "
                     f"SQNR = {sq_txt} dB   rms = {err['rms']:.3f}   max = {err['max']:.3f} LSB")
        ax.set_xlabel("indice de frecuencia  k"); ax.set_ylabel("error [LSB de salida]")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Procesar un caso para una fuente (expected|actual)
# ----------------------------------------------------------------------
def plot_one(module_name, case_name, case_dir, params, src, k0_override, reorder_override):
    vroot = case_dir / "simulation" / "vectors"
    in_dir = vroot / "stimuli" / "in_ports"
    out_dir = vroot / src / "out_ports"

    for d in (in_dir, out_dir):
        if not d.is_dir():
            print(f"  [SKIP] {case_name} ({src}): no existe {d}")
            return False

    (nb, nbf), (nb_in, nbf_in), (nb_out, nbf_out) = port_formats(params)

    try:
        # Cada puerto en SU formato: la entrada en Q(NB_IN,NBF_IN), la salida
        # en Q(NB_OUT,NBF_OUT). Con NB=20/NBF=19 y puertos de 16 bits, usar el
        # NBF del datapath achicaba todo 16 veces.
        x_in = load_complex(in_dir, "i_re.dat", "i_im.dat", nbf_in)
        X_raw = load_complex(out_dir, "o_re.dat", "o_im.dat", nbf_out)
    except FileNotFoundError as e:
        print(f"  [SKIP] {case_name} ({src}): {e}")
        return False

    # Filtrar AMBOS por su senal de validez. Con N_CYCLES > T_frame el TB
    # sigue escribiendo despues del frame (relleno con valid=0): si no se
    # filtra, la entrada aparece como la senoidal seguida de una cola de
    # ceros, y la salida mezcla basura de estados intermedios con la FFT.
    x_in  = filter_valid(x_in,  in_dir  / "i_valid.dat")
    X_raw = filter_valid(X_raw, out_dir / "o_valid.dat")
    if len(X_raw) == 0:
        print(f"  [SKIP] {case_name} ({src}): salida vacia (o_valid nunca 1?)")
        return False

    N = int(params.get("N", len(X_raw)))
    # Red de seguridad: quedarse con UN frame de N muestras aunque el .dat
    # traiga mas (p.ej. si o_valid quedara alto de mas, o multiples frames).
    x_in  = x_in[:N]
    X_raw = X_raw[:N]

    if len(x_in) < N:
        print(f"  [WARN] {case_name} ({src}): entrada valida={len(x_in)} < N={N}")
    if len(X_raw) < N:
        print(f"  [WARN] {case_name} ({src}): salida valida={len(X_raw)} < N={N} "
              f"(frame incompleto: subir N_CYCLES?)")

    # Referencia ideal: FFT de la entrada TAL COMO LA VIO EL DUT (ya cuantizada,
    # leida del .dat) escalada por 2^-S. Mide solo el error de la FFT.
    S, mask = scale_stages(params, N)
    complete = (len(x_in) == N and len(X_raw) == N)
    X_ref = np.fft.fft(x_in) * 2.0 ** (-S) if complete else None

    k0, k0_src = resolve_k0(case_name, case_dir, params, k0_override)

    # Orden de la salida. Prioridad: CLI > PLOT_REORDER del JSON > 'auto'.
    # En 'auto' se detecta contra la referencia (anda con cualquier estimulo).
    reorder = reorder_override or str(params.get("PLOT_REORDER", "auto")).lower()
    detected = detect_order_by_reference(X_raw, X_ref) if X_ref is not None else "desconocido"

    X_out, applied = X_raw, "none"
    if reorder == "bitrev":
        X_out = bitrev_permute(X_raw)
        applied = "bitrev(forzado)"
    elif reorder == "none":
        applied = "none(forzado)"
    elif detected == "bitrev":
        X_out = bitrev_permute(X_raw)
        applied = "bitrev(auto)"

    # Orden de lo que se plotea, segun la referencia
    order = detect_order_by_reference(X_out, X_ref) if X_ref is not None else "desconocido"
    measurable = order in ("natural", "indistinguible")
    if order == "desconocido" and applied == "bitrev(forzado)" and X_ref is not None:
        # Salida muy degradada (p. ej. saturada): la referencia no alcanza para
        # confirmar el orden, pero PLOT_REORDER=bitrev lo fija. Se mide igual.
        order = "natural (forzado, sin confirmar)"
        measurable = True
    info = peak_info(X_out)

    err = None
    if X_ref is not None and measurable:
        err = error_stats(X_out, X_ref, 2.0 ** (-nbf_out))

    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"{case_name}_{src}.png"

    title = (f"{module_name}  |  {case_name}  |  src={src}\n"
             f"N={N}   datapath {qfmt(nb, nbf)}   in {qfmt(nb_in, nbf_in)}   "
             f"out {qfmt(nb_out, nbf_out)}   SCALE_MASK={mask & ((1 << int(round(np.log2(N)))) - 1)} (S={S})")
    if k0 is not None:
        title += f"   k0={k0}"
    make_figure(x_in, X_out, N, order, info, title, str(out_png), k0=k0, err=err)

    err_txt = (f" SQNR={err['sqnr_db']:.1f}dB sesgo=({err['bias_re']:+.3f},{err['bias_im']:+.3f}) "
               f"max={err['max']:.3f}LSB") if err else f" (sin panel de error: orden {order})"
    print(f"  [OK]   {case_name} ({src}): N={N} in={qfmt(nb_in, nbf_in)} out={qfmt(nb_out, nbf_out)} "
          f"S={S} k0={k0}({k0_src}) orden={order} reorden={applied}{err_txt} -> {out_png}")
    return True


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plotea entrada+FFT+error de cada caso de un modulo fft1d_*.")
    ap.add_argument("--src", choices=["expected", "actual", "both"], default="expected",
                    help="fuente de la salida FFT (default expected = modelo C++)")
    ap.add_argument("--only", type=str, default=None, help="plotear solo este caso")
    ap.add_argument("--k0", type=int, default=None, help="forzar k0 (sobreescribe autodeteccion)")
    ap.add_argument("--json", type=Path, default=None, help="ruta al JSON de regresion (override)")
    ap.add_argument("--reorder", choices=["auto", "none", "bitrev"], default=None,
                    help="reordenamiento de la salida: auto (detecta contra la FFT ideal), "
                         "bitrev (fuerza bit-reversed->natural), none (grafica cruda). "
                         "Si se omite, usa PLOT_REORDER del caso en el JSON, o 'auto'.")
    args = ap.parse_args()

    module_name, module_dir, prefix = detect_current_module()
    banner(f"run_plot_fft1d  [{module_name}]")

    # ubicar JSON (mismo orden de prioridad que run_regression_sim)
    if args.json:
        regression_json = resolve_path(args.json, module_dir)
    else:
        env = (os.environ.get(f"{prefix}_TB_REGRESSION_JSON")
               or os.environ.get(f"{prefix}_REGRESSION_JSON"))
        regression_json = (resolve_path(env, module_dir) if env
                           else module_dir / "testbench" / f"{module_name}_tb_regression.json")

    kv("regression_json", regression_json)
    data = load_json(regression_json, "plot regression JSON")

    regression_dir = resolve_path(data.get("regression_dir", "build"), module_dir)
    defaults = data.get("defaults", {})
    default_params = extract_params(defaults, META_KEYS)

    cases = data.get("cases")
    if not cases:
        print("[ERROR] JSON sin 'cases'", file=sys.stderr)
        sys.exit(1)

    srcs = ["expected", "actual"] if args.src == "both" else [args.src]

    total, done = 0, 0
    for raw_case in cases:
        name = raw_case.get("name")
        if not name:
            continue
        if args.only and name != args.only:
            continue
        if raw_case.get("enabled", True) is False:
            continue

        params = dict(default_params)
        params.update(extract_params(raw_case, META_KEYS))
        case_dir = regression_dir / name

        if not case_dir.is_dir():
            print(f"  [SKIP] {name}: no existe build dir {case_dir}")
            continue

        for src in srcs:
            total += 1
            if plot_one(module_name, name, case_dir, params, src, args.k0, args.reorder):
                done += 1

    print(f"\n[run_plot_fft1d] figuras generadas: {done}/{total}")
    sys.exit(0 if done > 0 else 1)


if __name__ == "__main__":
    main()
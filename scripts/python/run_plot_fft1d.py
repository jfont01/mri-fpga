#!/usr/bin/env python3
"""
run_plot_fft1d.py

Runner de ploteo para el modulo fft1d_r2. Se ejecuta SIN argumentos desde el
directorio de un modulo (igual que run_gtest.py / run_regression_sim.py):
descubre solo el modulo, lee su JSON de regresion, ubica los .dat de cada caso
y genera una figura (entrada + FFT magnitud/fase) por caso.

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
  N    : del JSON (clave "N"); si falta, se infiere de la cantidad de muestras.
  NBF  : del JSON (clave "NBF"); default 15.
  K0   : en este orden -> del sufijo "_k<N>" en el nombre del caso;
         si no, del sim_summary.rpt ("complex sine k0=..."); si no, None.
         Con K0 conocido, la deteccion natural/bitrev es exacta.

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


def detect_order(X, k0=None):
    n = len(X)
    log2n = int(round(np.log2(n)))
    mag = np.abs(X)
    peak = int(np.argmax(mag))
    purity = (mag[peak] ** 2) / (np.sum(mag ** 2) + 1e-30)
    info = {"peak_index": peak, "purity": purity}
    if k0 is not None:
        k0 %= n
        if peak == k0:
            return "natural", info
        if peak == bitrev_index(k0, log2n):
            return "bitrev", info
        return "desconocido", info
    return "desconocido", info


# ----------------------------------------------------------------------
# Extraccion automatica de K0
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


def resolve_k0(case_name: str, case_dir: Path, override):
    if override is not None:
        return override, "cli"
    k = k0_from_case_name(case_name)
    if k is not None:
        return k, "nombre_caso"
    k = k0_from_report(case_dir)
    if k is not None:
        return k, "sim_summary.rpt"
    return None, "no_encontrado"


# ----------------------------------------------------------------------
# Plot (identico a plot_fft.py, condensado)
# ----------------------------------------------------------------------
def make_figure(x_in, X_out, N, order, info, title, out_png, k0=None):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)

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

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Procesar un caso para una fuente (expected|actual)
# ----------------------------------------------------------------------
def plot_one(case_name, case_dir, params, src, k0_override):
    vroot = case_dir / "simulation" / "vectors"
    in_dir = vroot / "stimuli" / "in_ports"
    out_dir = vroot / src / "out_ports"

    for d in (in_dir, out_dir):
        if not d.is_dir():
            print(f"  [SKIP] {case_name} ({src}): no existe {d}")
            return False

    nbf = int(params.get("NBF", 15))

    try:
        x_in = load_complex(in_dir, "i_re.dat", "i_im.dat", nbf)
        X_raw = load_complex(out_dir, "o_re.dat", "o_im.dat", nbf)
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

    k0, k0_src = resolve_k0(case_name, case_dir, k0_override)
    order, info = detect_order(X_raw, k0=k0)

    X_out, applied = X_raw, "none"
    if order == "bitrev":
        X_out = bitrev_permute(X_raw)
        order, info = detect_order(X_out, k0=k0)
        applied = "bitrev(auto)"

    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"{case_name}_{src}.png"

    title = f"fft1d_r2  |  {case_name}  |  N={N}  Q1.{nbf}  src={src}"
    if k0 is not None:
        title += f"  k0={k0}"
    make_figure(x_in, X_out, N, order, info, title, str(out_png), k0=k0)

    print(f"  [OK]   {case_name} ({src}): N={N} nbf={nbf} k0={k0}({k0_src}) "
          f"orden={order} reorden={applied} -> {out_png}")
    return True


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plotea entrada+FFT de cada caso de fft1d_r2.")
    ap.add_argument("--src", choices=["expected", "actual", "both"], default="expected",
                    help="fuente de la salida FFT (default expected = modelo C++)")
    ap.add_argument("--only", type=str, default=None, help="plotear solo este caso")
    ap.add_argument("--k0", type=int, default=None, help="forzar k0 (sobreescribe autodeteccion)")
    ap.add_argument("--json", type=Path, default=None, help="ruta al JSON de regresion (override)")
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
            if plot_one(name, case_dir, params, src, args.k0):
                done += 1

    print(f"\n[run_plot_fft1d] figuras generadas: {done}/{total}")
    sys.exit(0 if done > 0 else 1)


if __name__ == "__main__":
    main()
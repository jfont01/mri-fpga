#!/usr/bin/env python3
"""
run_plot_fft2d.py

Runner de ploteo de la fft2d. Igual que run_plot_fft1d.py: se ejecuta desde la
carpeta del modulo, lee su JSON de regresion (el mismo de gtest y sim), toma
los .dat de cada caso y genera una figura por caso y por bobina:

  fila 1  (a) |K| del k-space de entrada, en dB          (k' vertical, r horizontal)
          (b) |A| de la salida, ya en orden natural, en dB  (p vertical, x horizontal)
          (c) |A| de referencia: IDFT 2D ideal de la MISMA entrada por 2^-(S1+S2)
          (d) |error| en LSB de salida, con SQNR, rms, maximo y sesgo
  fila 2  solo con el fantoma (STIM_TYPE 3 con su .json de gen_fft2d_stim.py):
          (e) fantoma original |m|, (f) imagen desplegada |m_hat| con la salida
          del modelo y U = (S^H S)^-1 S^H, (g) |m_hat - m| en dB, con la SNR

El eje horizontal es siempre el readout (r en el k-space, x en la imagen).

Modo imagen (INVERSE = 0, p. ej. las imagenes de gen_fft2d_stim.py): la
entrada es una imagen y la salida su espectro, asi que la fila 1 pasa a ser
  (a) la imagen de entrada (lineal, en grises)
  (b) |X| de la salida y (c) |X| de la FFT 2D ideal, en dB y centrados
      (fftshift: frecuencia 0 en el medio)
  (d) |error| en LSB, tambien centrado

Modo ida y vuelta (STIM_FILE de gen_fft2d_stim.py con kind = roundtrip): la
entrada es el espectro bit a bit de una imagen (pasadas 1 y 2) y la fft2d hace
la vuelta (pasadas 3 y 4). La fila 1 es la de siempre (contra la IDFT 2D de la
misma entrada) y se agrega
  fila 2  (e) la imagen original, (f) la reconstruida (parte real),
          (g) |error| contra la original en dB, (h) SQNR de la ida y vuelta,
          error en el pixel (0, 0) y dos chequeos: el estimulo es el espectro
          regenerado y la salida es, bit a bit, la de la cadena de Python
          (gen_fft2d_stim.RoundTrip, con el ROUND_CONV del caso)

En los casos con INVERSE = 1 la linea de texto agrega el sesgo de la fila p = 0
y el del resto: el sesgo half-up se concentra en p = 0 (bitacora 4.6) y con
ROUND_CONV = 1 tiene que desaparecer.

Orden de la salida
------------------
La fft2d entrega o = t1 * N2 + t2 = A[p = br(t2), x = br(t1)] (bit-reversed en
las dos pasadas, bitacora seccion 3.8). Aca se deshace para plotear en orden
natural; si la salida no coincide con la referencia en ese orden, el error lo
muestra.

Descubrimiento
--------------
  modulo : MODULES_ROOT + cwd (detect_current_module)
  JSON   : <PREFIX>_TB_REGRESSION_JSON o testbench/<module>_tb_regression.json
  .dat   : build/<CASE>/simulation/vectors/stimuli/in_ports/{i_valid,i_re,i_im}.dat
           build/<CASE>/simulation/vectors/<src>/out_ports/{o_valid,o_last,o_re,o_im}.dat
  salida : build/<CASE>/plots/<CASE>_<src>_coil<l>.png

Parametros por caso (mismos defaults que fft2d.hpp y fft2d_stimulus.hpp)
-------------------------------------------------------------------------
  N1 1024, N2 512, SCALE_MASK1 801, SCALE_MASK2 384, INVERSE 1, N_COILS 3,
  NB 20 / NBF 19, NB_IN 16 y NB_OUT 16 (NBF_x = NB_x - (NB - NBF), o sea Q1.15).

Uso (desde modules/fft2d)
-------------------------
  python3 ../../scripts/python/run_plot_fft2d.py                    # todos, bobina 0
  python3 ../../scripts/python/run_plot_fft2d.py --only n64x32_phantom --coil 3
  python3 ../../scripts/python/run_plot_fft2d.py --all-coils
  python3 ../../scripts/python/run_plot_fft2d.py --src actual       # salida del RTL, cuando exista
  python3 ../../scripts/python/run_plot_fft2d.py --full-check       # build/full_check de fft2d_check_full.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regression_common import (
    banner,
    detect_current_module,
    extract_params,
    kv,
    load_json,
    resolve_path,
)

META_KEYS = {"name", "enabled", "params", "defines", "cpp_defines",
             "sim_defines", "auto_defines", "binary_file", "modes", "gtest"}

DB_FLOOR = -80.0


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------
def load_dat_raw(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def load_valid(d: Path, prefix: str, nbf: int):
    v = load_dat_raw(d / f"{prefix}_valid.dat").astype(bool)
    re = load_dat_raw(d / f"{prefix}_re.dat")
    im = load_dat_raw(d / f"{prefix}_im.dat")
    n = min(len(v), len(re), len(im))
    v, re, im = v[:n], re[:n], im[:n]
    return re[v], im[v], (re[v] + 1j * im[v]) / float(2 ** nbf)


def as_int(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def case_config(params):
    nb, nbf = as_int(params.get("NB"), 20), as_int(params.get("NBF"), 19)
    nbi = nb - nbf
    nb_in = as_int(params.get("NB_IN"), 16)
    nb_out = as_int(params.get("NB_OUT"), 16)
    n1, n2 = as_int(params.get("N1"), 1024), as_int(params.get("N2"), 512)
    l1, l2 = int(np.log2(n1)), int(np.log2(n2))
    m1 = as_int(params.get("SCALE_MASK1"), 801) & ((1 << l1) - 1)
    m2 = as_int(params.get("SCALE_MASK2"), 384) & ((1 << l2) - 1)
    return {
        "n1": n1, "n2": n2, "l1": l1, "l2": l2, "m1": m1, "m2": m2,
        "s1": bin(m1).count("1"), "s2": bin(m2).count("1"),
        "inverse": as_int(params.get("INVERSE"), 1) == 1,
        "round_conv": as_int(params.get("ROUND_CONV"), 0) == 1,
        "coils": as_int(params.get("N_COILS"), 3),
        "nbf_in": as_int(params.get("NBF_IN"), nb_in - nbi),
        "nbf_out": as_int(params.get("NBF_OUT"), nb_out - nbi),
        "nb_in": nb_in, "nb_out": nb_out,
    }


def bitrev_perm(n: int) -> np.ndarray:
    bits = int(np.log2(n))
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])


def hw_to_natural(a_hw, cfg):
    """a_hw[l, t1, t2] (orden del stream) -> A[l, p, x] en orden natural."""
    br1, br2 = bitrev_perm(cfg["n1"]), bitrev_perm(cfg["n2"])
    a_xp = a_hw[:, br1, :][:, :, br2]           # [l, x, p]
    return np.swapaxes(a_xp, 1, 2)              # [l, p, x]


def reference(k, cfg):
    """IDFT 2D sin normalizar (o FFT si INVERSE = 0) por 2^-(S1+S2): [l, p, x]."""
    n1, n2 = cfg["n1"], cfg["n2"]
    if cfg["inverse"]:
        y = np.fft.ifft2(k, axes=(-2, -1)) * (n1 * n2)
    else:
        y = np.fft.fft2(k, axes=(-2, -1))
    return y * 2.0 ** (-(cfg["s1"] + cfg["s2"]))


def error_stats(out, ref, lsb):
    e = (out - ref) / lsb
    p_sig, p_err = np.sum(np.abs(ref) ** 2), np.sum(np.abs(out - ref) ** 2)
    sqnr = np.inf if p_err == 0 else 10 * np.log10(p_sig / p_err)
    return {"e": e, "sqnr": sqnr, "rms": float(np.sqrt(np.mean(np.abs(e) ** 2))),
            "max": float(np.max(np.abs(e))),
            "bias": (float(np.mean(e.real)), float(np.mean(e.imag)))}


def db(x, ref_max):
    return 20 * np.log10(np.abs(x) / ref_max + 1e-30).clip(DB_FLOOR, None)


# ----------------------------------------------------------------------
# Fantoma: rehacerlo con los mismos generadores y desplegar
# ----------------------------------------------------------------------
def model_dir():
    root = os.environ.get("PROJECT_ROOT")
    return Path(root) / "model" if root else None


def phantom_unfold(meta, qr_in, qi_in, a_nat, cfg):
    """
    Rehace fantoma y mapas con gen_fft2d_stim.phantom_kspace (los mismos
    parametros del .json), chequea que el estimulo sea ese y despliega.
    a_nat: [l, p, x] en Q1.15. Devuelve dict o None (con motivo).
    """
    md = model_dir()
    if md is None or not (md / "gen_fft2d_stim.py").is_file():
        return None, "no se encuentra model/gen_fft2d_stim.py (PROJECT_ROOT)"
    sys.path.insert(0, str(md))
    import fft2d_ref as R                                   # noqa: E402
    from gen_fft2d_stim import phantom_kspace               # noqa: E402

    m, smaps, y_true, _, g, qr, qi = phantom_kspace(md / "gen", meta["N"], meta["L"],
                                                    meta["af"], meta["norm"])
    same = (qr.shape == qr_in.shape) and np.array_equal(qr, qr_in) and np.array_equal(qi, qi_in)
    # A en Q1.15 -> unidades fisicas de y_alias (como Chain.phys), [l, x, p]
    a_xp = np.swapaxes(a_nat, 1, 2)
    y = a_xp * 2.0 ** (cfg["s1"] + cfg["s2"] - cfg["l1"] - cfg["l2"]) / g[:, None, None]
    U = R.unfolding_matrix(smaps, meta["af"])
    mh = R.unfold(y, U, meta["af"])                          # [x, y]
    snr = 10 * np.log10(np.sum(np.abs(m) ** 2) / np.sum(np.abs(mh - m) ** 2))
    snr_alias = 10 * np.log10(np.sum(np.abs(y_true) ** 2) / np.sum(np.abs(y - y_true) ** 2))
    return {"m": m, "mh": mh, "snr": snr, "snr_alias": snr_alias, "same_stim": same}, ""


# ----------------------------------------------------------------------
# Ida y vuelta: rehacer imagen y espectro y comparar contra la original
# ----------------------------------------------------------------------
def roundtrip_check(meta, qr_in, qi_in, a_nat, cfg):
    """
    Rehace la imagen y las 4 pasadas con gen_fft2d_stim.roundtrip_chain (el
    stim_gen guardado en el .json y las mascaras de este caso como vuelta).
    a_nat: [1, p, x] en Q1.15. Devuelve dict o None (con motivo).
    """
    md = model_dir()
    if md is None or not (md / "gen_fft2d_stim.py").is_file():
        return None, "no se encuentra model/gen_fft2d_stim.py (PROJECT_ROOT)"
    sys.path.insert(0, str(md))
    from gen_fft2d_stim import roundtrip_chain              # noqa: E402

    r = roundtrip_chain(md / "gen", meta["spec"], (cfg["m1"], cfg["m2"]), round_conv=int(cfg["round_conv"]))
    X = r["X"]
    same_stim = (qr_in.shape[1:] == X.shape and np.array_equal(qr_in[0], X.real)
                 and np.array_equal(qi_in[0], X.imag))
    y = a_nat[0] * 2.0 ** cfg["nbf_out"]                    # LSB, exacto
    x = r["x"]
    e = y - x
    sqnr = 10 * np.log10(np.sum(np.abs(x) ** 2) / max(np.sum(np.abs(e) ** 2), 1e-300))
    return {"x": x, "y": y, "e": e, "sqnr": sqnr, "e00": complex(e[0, 0]),
            "same_stim": same_stim, "same_chain": bool(np.array_equal(y, r["y"])),
            "sats": r["sats"], "masks": r["masks"], "image": meta["spec"]["image"]}, ""


# ----------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------
def make_figure(k_l, a_l, ref_l, st, ph, title, out_png, cfg, coil, rt=None):
    rows = 2 if (ph is not None or rt is not None) else 1
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4.6 * rows + 0.6), squeeze=False,
                             constrained_layout=True)
    ext_k = dict(aspect="auto", origin="lower", interpolation="nearest")

    ax = axes[0, 0]
    kmax = np.abs(k_l).max() or 1.0
    # piso de -100 dB: 1 LSB de Q1.15 esta ~90 dB debajo del pico, asi se ve
    # que parte del k-space la cuantizacion de entrada deja en cero
    kdb = (20 * np.log10(np.abs(k_l) / kmax + 1e-30)).clip(-100, None)
    im = ax.imshow(kdb, cmap="magma", vmin=-100, vmax=0, **ext_k)
    ax.set_title(f"(a) |K| entrada, bobina {coil}  [dB re max]")
    ax.set_xlabel("readout r"); ax.set_ylabel("fila k'  (phase-encode)")
    fig.colorbar(im, ax=ax, shrink=0.85)

    rmax = np.abs(ref_l).max() or 1.0
    for ax, z, name in ((axes[0, 1], a_l, "(b) |A| salida del modelo"),
                        (axes[0, 2], ref_l, "(c) |A| referencia ideal")):
        im = ax.imshow(db(z, rmax), cmap="viridis", vmin=DB_FLOOR, vmax=0, **ext_k)
        ax.set_title(f"{name}  [dB re max ref]")
        ax.set_xlabel("x  (readout)"); ax.set_ylabel("p  (phase-encode plegado)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[0, 3]
    im = ax.imshow(np.abs(st["e"]), cmap="inferno", **ext_k)
    sq = "inf" if np.isinf(st["sqnr"]) else f"{st['sqnr']:.1f}"
    ax.set_title(f"(d) |error| [LSB]  SQNR {sq} dB\nrms {st['rms']:.3f}  max {st['max']:.2f}  "
                 f"sesgo ({st['bias'][0]:+.3f}, {st['bias'][1]:+.3f})")
    ax.set_xlabel("x  (readout)"); ax.set_ylabel("p")
    fig.colorbar(im, ax=ax, shrink=0.85)

    if ph is not None:
        img = dict(aspect="equal", origin="lower", interpolation="nearest", cmap="gray")
        mmax = np.abs(ph["m"]).max() or 1.0
        ax = axes[1, 0]
        ax.imshow(np.abs(ph["m"]).T, vmin=0, vmax=mmax, **img)
        ax.set_title("(e) fantoma original |m|"); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax = axes[1, 1]
        ax.imshow(np.abs(ph["mh"]).T, vmin=0, vmax=mmax, **img)
        ax.set_title(f"(f) |m_hat| desplegada (salida del modelo)\nSNR imagen {ph['snr']:.2f} dB, "
                     f"plegada {ph['snr_alias']:.2f} dB")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax = axes[1, 2]
        # piso mas bajo que el resto: con SNR de 50-70 dB el error queda cerca de -80
        im = ax.imshow((20 * np.log10(np.abs(ph["mh"] - ph["m"]) / mmax + 1e-30)).clip(-120, None).T,
                       cmap="magma", vmin=-120, vmax=-20,
                       aspect="equal", origin="lower", interpolation="nearest")
        ax.set_title("(g) |m_hat - m|  [dB re max |m|]"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax = axes[1, 3]
        ax.set_axis_off()
        txt = ("estimulo = fantoma regenerado: " + ("si (bit a bit)" if ph["same_stim"] else "NO") +
               f"\nS1 = {cfg['s1']}, S2 = {cfg['s2']}\n"
               f"SNR imagen {ph['snr']:.2f} dB")
        ax.text(0.02, 0.6, txt, fontsize=12, va="center", family="monospace")

    if rt is not None:
        img = dict(aspect="equal", origin="upper", interpolation="nearest")
        xmax = np.abs(rt["x"]).max() or 1.0
        for ax, z, name in ((axes[1, 0], rt["x"].real, f"(e) original: {rt['image']}"),
                            (axes[1, 1], rt["y"].real, "(f) reconstruida (parte real)")):
            im = ax.imshow(z / 2.0 ** cfg["nbf_out"], cmap="gray", vmin=-xmax / 2.0 ** cfg["nbf_out"],
                           vmax=xmax / 2.0 ** cfg["nbf_out"], **img)
            ax.set_title(name); ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.85)
        ax = axes[1, 2]
        im = ax.imshow((20 * np.log10(np.abs(rt["e"]) / xmax + 1e-30)).clip(-120, None),
                       cmap="magma", vmin=-120, vmax=-20, **img)
        ax.set_title("(g) |reconstruida - original|  [dB re max]"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax = axes[1, 3]
        ax.set_axis_off()
        m = rt["masks"]
        txt = (f"ida  m1 = {m[0]}, m2 = {m[1]}\nvuelta  m3 = {m[2]}, m4 = {m[3]}\n"
               f"S = {sum(bin(v).count('1') for v in m)}, sat por pasada {rt['sats']}\n\n"
               f"SQNR ida y vuelta {rt['sqnr']:.2f} dB\n"
               f"error (0, 0) = {rt['e00'].real:+.0f} {rt['e00'].imag:+.0f}j LSB\n\n"
               "estimulo = espectro regenerado: " + ("si" if rt["same_stim"] else "NO") +
               "\nsalida = cadena Python: " + ("si (bit a bit)" if rt["same_chain"] else "NO"))
        ax.text(0.02, 0.5, txt, fontsize=12, va="center", family="monospace")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def make_figure_image(x_l, a_l, ref_l, st, title, out_png, cfg, meta):
    """Modo imagen: entrada en el espacio, salida = espectro (centrado con fftshift)."""
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6), squeeze=False, constrained_layout=True)
    n2, n1 = x_l.shape
    ax = axes[0, 0]
    img = x_l.real if np.abs(x_l.imag).max() == 0 else np.abs(x_l)
    im = ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
    what = meta.get("image", "imagen") if meta else "imagen"
    ax.set_title(f"(a) entrada: {what}  ({n1} x {n2}, Q1.15)")
    ax.set_xlabel("x  (columna, readout)"); ax.set_ylabel("y  (fila)")
    fig.colorbar(im, ax=ax, shrink=0.85)

    shift = lambda z: np.fft.fftshift(z)
    ext = [-n1 // 2 - 0.5, n1 // 2 - 0.5, -n2 // 2 - 0.5, n2 // 2 - 0.5]
    spec = dict(origin="lower", interpolation="nearest", extent=ext, aspect="equal")
    rmax = np.abs(ref_l).max() or 1.0
    floor = -100.0
    for ax, z, name in ((axes[0, 1], a_l, "(b) |X| salida del modelo"),
                        (axes[0, 2], ref_l, "(c) |X| FFT 2D ideal")):
        zdb = (20 * np.log10(np.abs(shift(z)) / rmax + 1e-30)).clip(floor, None)
        im = ax.imshow(zdb, cmap="inferno", vmin=floor, vmax=0, **spec)
        ax.set_title(f"{name}  [dB re max]")
        ax.set_xlabel("kx  (ciclos por imagen)"); ax.set_ylabel("ky")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[0, 3]
    im = ax.imshow(np.abs(shift(st["e"])), cmap="magma", **spec)
    sq = "inf" if np.isinf(st["sqnr"]) else f"{st['sqnr']:.1f}"
    ax.set_title(f"(d) |error| [LSB]  SQNR {sq} dB\nrms {st['rms']:.3f}  max {st['max']:.2f}  "
                 f"sesgo ({st['bias'][0]:+.3f}, {st['bias'][1]:+.3f})")
    ax.set_xlabel("kx"); ax.set_ylabel("ky")
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_case(label, k_all, a_hw, n_last, cfg, coils, out_dir, phantom_meta, raw_in, image_meta=None,
              rt_meta=None):
    """k_all: [L, N2, N1] complejo (Q1.15 -> real). a_hw: [L, N1, N2] complejo."""
    a_nat = hw_to_natural(a_hw, cfg)
    ref = reference(k_all, cfg)
    lsb = 2.0 ** (-cfg["nbf_out"])

    ph, why = None, ""
    if phantom_meta is not None:
        qr_in, qi_in = raw_in
        ph, why = phantom_unfold(phantom_meta, qr_in, qi_in, a_nat, cfg)
        if ph is None:
            print(f"  [WARN] {label}: sin fila del fantoma ({why})")
    rt = None
    if rt_meta is not None:
        qr_in, qi_in = raw_in
        rt, why = roundtrip_check(rt_meta, qr_in, qi_in, a_nat, cfg)
        if rt is None:
            print(f"  [WARN] {label}: sin fila de la ida y vuelta ({why})")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_all = [error_stats(a_nat[l], ref[l], lsb) for l in range(a_nat.shape[0])]
    sq = ", ".join("inf" if np.isinf(s["sqnr"]) else f"{s['sqnr']:.1f}" for s in stats_all)
    extra = ""
    if ph is not None:
        extra = (f"; SNR imagen {ph['snr']:.2f} dB; estimulo = fantoma: "
                 f"{'si' if ph['same_stim'] else 'NO'}")
    if rt is not None:
        extra = (f"; ida y vuelta {rt['sqnr']:.2f} dB, |error(0,0)| {abs(rt['e00']):.0f} LSB; "
                 f"estimulo = espectro: {'si' if rt['same_stim'] else 'NO'}; "
                 f"salida = cadena Python: {'si' if rt['same_chain'] else 'NO'}")
    if cfg["inverse"]:
        e_all = np.stack([s["e"] for s in stats_all])                          # [l, p, x], LSB
        extra += (f"; sesgo fila p=0 {e_all[:, 0].real.mean():+.3f}/{e_all[:, 0].imag.mean():+.3f}, "
                  f"resto {e_all[:, 1:].real.mean():+.4f}/{e_all[:, 1:].imag.mean():+.4f}")
    print(f"  [OK]   {label}: {cfg['n1']}x{cfg['n2']}, {a_nat.shape[0]} bobinas, o_last {n_last}, "
          f"S1={cfg['s1']} S2={cfg['s2']}; SQNR por bobina [{sq}] dB; "
          f"max {max(s['max'] for s in stats_all):.2f} LSB{extra}")

    files = []
    for l in coils:
        title = (f"fft2d  |  {label}  |  bobina {l}{'' if cfg['inverse'] else '   (FFT directa)'}\n"
                 f"N1={cfg['n1']} (mask {cfg['m1']}, S1={cfg['s1']})   "
                 f"N2={cfg['n2']} (mask {cfg['m2']}, S2={cfg['s2']})   "
                 f"in Q{cfg['nb_in'] - cfg['nbf_in']}.{cfg['nbf_in']}   "
                 f"out Q{cfg['nb_out'] - cfg['nbf_out']}.{cfg['nbf_out']}"
                 f"{'   INVERSA' if cfg['inverse'] else ''}{'   CONV' if cfg['round_conv'] else ''}")
        png = out_dir / f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_coil{l}.png"
        if cfg["inverse"]:
            make_figure(k_all[l], a_nat[l], ref[l], stats_all[l], ph, title, png, cfg, l, rt)
        else:
            make_figure_image(k_all[l], a_nat[l], ref[l], stats_all[l], title.replace("bobina", "canal"),
                              png, cfg, image_meta)
        files.append(png)
    return files


# ----------------------------------------------------------------------
# Fuentes de datos
# ----------------------------------------------------------------------
def from_case(module_dir, case_dir, name, params, src, coil, all_coils):
    cfg = case_config(params)
    vroot = case_dir / "simulation" / "vectors"
    in_dir, out_dir = vroot / "stimuli" / "in_ports", vroot / src / "out_ports"
    for d in (in_dir, out_dir):
        if not d.is_dir():
            print(f"  [SKIP] {name} ({src}): no existe {d}")
            return False

    qr, qi, k = load_valid(in_dir, "i", cfg["nbf_in"])
    _, _, a = load_valid(out_dir, "o", cfg["nbf_out"])
    t = cfg["n1"] * cfg["n2"]
    L = min(len(k), len(a)) // t
    if L == 0:
        print(f"  [SKIP] {name} ({src}): menos de una bobina completa (entrada {len(k)}, salida {len(a)})")
        return False
    if len(a) != len(k):
        print(f"  [WARN] {name} ({src}): {len(k)} muestras de entrada y {len(a)} de salida")
    last_path = out_dir / "o_last.dat"
    n_last = int(load_dat_raw(last_path).sum()) if last_path.is_file() else -1

    k_all = k[:L * t].reshape(L, cfg["n2"], cfg["n1"])
    a_hw = a[:L * t].reshape(L, cfg["n1"], cfg["n2"])
    raw_in = (qr[:L * t].reshape(L, cfg["n2"], cfg["n1"]), qi[:L * t].reshape(L, cfg["n2"], cfg["n1"]))

    meta, image_meta, rt_meta = None, None, None
    if as_int(params.get("STIM_TYPE"), 0) == 3:
        sidecar = (module_dir / str(params.get("STIM_FILE", ""))).with_suffix(".json")
        if sidecar.is_file():
            m = json.loads(sidecar.read_text())
            if m.get("kind") == "phantom":
                meta = m
            elif m.get("kind") == "image":
                image_meta = m
            elif m.get("kind") == "roundtrip":
                rt_meta = m

    coils = range(L) if all_coils else [min(coil, L - 1)]
    plot_case(f"{name}_{src}", k_all, a_hw, n_last, cfg, coils, case_dir / "plots", meta, raw_in, image_meta,
              rt_meta)
    return True


def from_full_check(module_dir, coil, all_coils):
    d = module_dir / "build" / "full_check"
    meta_path = d / "case.json"
    if not meta_path.is_file():
        print(f"[ERROR] no existe {meta_path}: correr antes model/fft2d_check_full.py")
        return False
    meta = json.loads(meta_path.read_text())
    params = {"N1": meta["N1"], "N2": meta["N2"], "SCALE_MASK1": meta["mask1"],
              "SCALE_MASK2": meta["mask2"], "N_COILS": meta["L"]}
    cfg = case_config(params)
    L, n1, n2 = meta["L"], meta["N1"], meta["N2"]
    kin = np.frombuffer((d / "kspace.bin").read_bytes(), dtype="<i4").reshape(L, n2, n1, 2)
    out = np.frombuffer((d / "out.bin").read_bytes(), dtype="<i4").reshape(L, n1, n2, 2)
    k_all = (kin[..., 0] + 1j * kin[..., 1]) / 2.0 ** cfg["nbf_in"]
    a_hw = (out[..., 0] + 1j * out[..., 1]) / 2.0 ** cfg["nbf_out"]
    coils = range(L) if all_coils else [min(coil, L - 1)]
    plot_case("full_check", k_all, a_hw, meta.get("o_last", -1), cfg, coils, d / "plots",
              meta, (kin[..., 0].astype(np.int64), kin[..., 1].astype(np.int64)))
    return True


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plotea entrada, salida, referencia y error de cada caso de la fft2d.")
    ap.add_argument("--src", choices=["expected", "actual", "both"], default="expected",
                    help="salida a plotear (default expected = modelo C++)")
    ap.add_argument("--only", type=str, default=None, help="plotear solo este caso")
    ap.add_argument("--coil", type=int, default=0, help="bobina a plotear (default 0)")
    ap.add_argument("--all-coils", action="store_true", help="una figura por bobina")
    ap.add_argument("--json", type=Path, default=None, help="JSON de regresion (override)")
    ap.add_argument("--full-check", action="store_true",
                    help="plotear build/full_check (model/fft2d_check_full.py) en vez de los casos")
    args = ap.parse_args()

    module_name, module_dir, prefix = detect_current_module()
    banner(f"run_plot_fft2d  [{module_name}]")

    if args.full_check:
        sys.exit(0 if from_full_check(module_dir, args.coil, args.all_coils) else 1)

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
    default_params = extract_params(data.get("defaults", {}), META_KEYS)
    srcs = ["expected", "actual"] if args.src == "both" else [args.src]

    total, done = 0, 0
    for raw_case in data.get("cases", []):
        name = raw_case.get("name")
        if not name or (args.only and name != args.only) or raw_case.get("enabled", True) is False:
            continue
        params = dict(default_params)
        params.update(extract_params(raw_case, META_KEYS))
        case_dir = regression_dir / name
        if not case_dir.is_dir():
            print(f"  [SKIP] {name}: no existe {case_dir} (correr sim)")
            continue
        for src in srcs:
            total += 1
            done += from_case(module_dir, case_dir, name, params, src, args.coil, args.all_coils)

    print(f"\n[run_plot_fft2d] casos ploteados: {done}/{total}")
    sys.exit(0 if done > 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fft1d_scaling_bound.py -- cota de los picos internos de una pasada segun donde
estan las etapas que dividen (bitacora 4.12). Estudio, no regresion.

Con las S etapas que dividen AL PRINCIPIO de la pasada (mascara 2^S - 1):

  pico de cualquier etapa interna <= max(pico de la entrada, pico de la salida)

en modulo, salvo el redondeo. Por que:
  * etapas que dividen: (a +- b) w / 2 con |w| = 1 es un promedio, no supera al
    mayor de los dos;
  * etapas que no dividen: despues de k etapas de una DIF quedan 2^k DFT
    independientes de N / 2^k puntos; cada valor intermedio es la IDFT (con su
    1/M) de un bloque de la salida, asi que no supera el pico de la salida.
Consecuencia: alcanza con chequear las fronteras entre pasadas. Dividiendo al
final no hay garantia: las primeras etapas suman sin dividir.

Dos chequeos:
  float   radix-2 DIF en NumPy, N = 16 a 1024, 8 senales, todos los S
  fxp     modelo bit a bit fft1d_r22sdf_fxp.py (20/18/16), N = 64, 512 (tail) y
          1024, Shepp / ruido / chirp, con el S minimo para que la salida quede
          < 1: al principio contra al final

Uso (desde la raiz del repo):
  python3 model/fft1d_scaling_bound.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "gen")]

import fft1d_r22sdf_fxp as F                    # noqa: E402
from gen_phantom import gen_shepp_logan_2d      # noqa: E402

TW_ROOT = HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"


def dif_stage_peaks(x, divide):
    """Radix-2 DIF en float, entrada en orden natural; divide[k]: la etapa k divide
    por 2. Devuelve el pico en modulo despues de cada etapa (sobre todas las filas)."""
    x = np.array(x, dtype=complex)
    n = x.shape[-1]
    peaks, span = [], n
    for k in range(n.bit_length() - 1):
        h = span // 2
        y = x.reshape(*x.shape[:-1], n // span, 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        s, d = a + b, (a - b) * np.exp(-2j * np.pi * np.arange(h) / span)
        if divide[k]:
            s, d = s / 2, d / 2
        x = np.stack([s, d], axis=-2).reshape(x.shape)
        peaks.append(np.abs(x).max())
        span = h
    return peaks


def float_signals(n, rng):
    img = gen_shepp_logan_2d(n, 0.9)
    t = np.arange(n)
    spec = np.fft.fft(img, axis=1)
    yield "shepp filas", img
    yield "shepp columnas", img.T
    yield "constante", np.full((4, n), 0.9)
    yield "ruido", 0.9 * (rng.uniform(-1, 1, (64, n)) + 1j * rng.uniform(-1, 1, (64, n))) / math.sqrt(2)
    yield "chirp", 0.9 * np.exp(1j * np.pi * t * t / n)[None]
    yield "impulso", np.eye(1, n) * 0.9
    yield "tono", 0.9 * np.exp(2j * np.pi * 5 * t / n)[None]
    yield "espectro de shepp", spec / 2 ** math.ceil(math.log2(np.abs(spec).max()))


def check_float(rng) -> float:
    worst = 0.0
    for n in (16, 64, 256, 1024):
        L = n.bit_length() - 1
        for name, x in float_signals(n, rng):
            pin, pout = np.abs(x).max(), np.abs(np.fft.fft(x, axis=-1)).max()
            for S in range(L + 1):
                pe = dif_stage_peaks(x, [k < S for k in range(L)])
                worst = max(worst, max(pe) / max(pin, pout / 2 ** S))
                if n == 256 and S in (4, 6) and name in ("shepp filas", "ruido", "chirp"):
                    pl = dif_stage_peaks(x, [k >= L - S for k in range(L)])
                    print(f"  N = {n} {name:12s} S = {S}: cota {max(pin, pout / 2 ** S):.3f}, "
                          f"pico interno al principio {max(pe):.3f}, al final {max(pl):.3f}")
    return worst


def q15(z):
    return (np.clip(np.round(z.real * 32768), -32768, 32767).astype(np.int64),
            np.clip(np.round(z.imag * 32768), -32768, 32767).astype(np.int64))


def check_fxp(rng) -> float:
    worst = 0.0
    for n in (64, 512, 1024):
        L = n.bit_length() - 1
        tw = F.load_twiddles(TW_ROOT / f"n{n}_tw18", n, 18)
        t = np.arange(n)
        sigs = {"shepp filas": gen_shepp_logan_2d(n, 0.9).astype(complex),
                "ruido": 0.63 * (rng.uniform(-1, 1, (32, n)) + 1j * rng.uniform(-1, 1, (32, n))),
                "chirp": 0.9 * np.exp(1j * np.pi * t * t / n)[None]}
        for name, z in sigs.items():
            xr, xi = q15(z)
            x = (xr + 1j * xi) / 32768
            pin, pout = np.abs(x).max(), np.abs(np.fft.fft(x, axis=-1)).max()
            S = max(0, math.floor(math.log2(pout)) + 1)       # minimo para que la salida quede < 1
            bound = max(pin, pout / 2 ** S)
            row = []
            for where, mask in (("principio", (1 << S) - 1), ("final", ((1 << S) - 1) << (L - S))):
                st = F.Stats()
                cfg = F.Cfg(N=n, NB=20, NBF=19, NB_TW=18, NBF_TW=17, NB_IN=16, NBF_IN=15,
                            NB_OUT=16, NBF_OUT=15, SCALE_MASK=mask, INVERSE=0)
                F.fft1d_r22sdf_fxp(xr, xi, cfg, *tw, st)
                inner = max(v for k, v in st.peak.items() if k != "salida")
                if where == "principio":
                    worst = max(worst, inner / bound)
                row.append(f"al {where} (mascara {mask}) pico {inner:.4f}, sat {st.total_sat}")
            print(f"  N = {n:4d} {name:11s} S = {S}, cota {bound:.4f}: " + "; ".join(row))
    return worst


def main() -> int:
    rng = np.random.default_rng(1)
    print("float (radix-2 DIF):")
    wf = check_float(rng)
    print(f"  max(pico interno / cota) dividiendo al principio = {wf:.6f}")
    print("\nbit a bit (fft1d_r22sdf_fxp.py, 20/18/16; pico interno por componente):")
    wx = check_fxp(rng)
    print(f"  max(pico interno / cota) dividiendo al principio = {wx:.6f}")
    ok = wf <= 1 + 1e-12 and wx <= 1 + 1e-12
    print("\nCOTA OK" if ok else "\nCOTA VIOLADA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

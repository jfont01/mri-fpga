#!/usr/bin/env python3
"""
fft_rounding_study.py -- half-up contra redondeo convergente (round-half-even) en
el modelo bit a bit, por punto de redondeo (bitacora 4.6, paso B). Estudio, no
regresion: cambia _cast de fft1d_r22sdf_fxp.py en memoria.

Al descartar d bits de x, con m = floor(x / 2^d) y r = x - m 2^d:
  half-up      m + (r >= 2^(d-1))
  convergente  m + (r > 2^(d-1)) + (r == 2^(d-1)) * lsb(m)

Puntos de redondeo del nucleo (20/18/16): mariposas que escalan (d = 1), cast
del cmul (d = 17) y cast de salida (d = 4). Modos comparados: half-up en todo
(hoy), convergente solo en las mariposas, en mariposas y salida, en todo.

Mide:
  1D      N = 1024, mascara completa, ruido: sesgo y rms contra la FFT ideal
  ida y   reparto minimo al principio (4.12), N = 64 y 256: SQNR, error en el
  vuelta  pixel (0, 0) y saturaciones; con --stoch, ademas redondeo estocastico
          (sin sesgo e independiente de la senal) y la media del error del
          espectro despues de la ida, que explica el error en (0, 0)
  2D      fantoma 1024 x 512, 8 bobinas, 801 / 384: sesgo en la fila p = 0 y en
          el resto, SNR de la imagen desplegada (unos 20 s)

Uso (desde la raiz del repo):
  python3 model/fft_rounding_study.py            # todo
  python3 model/fft_rounding_study.py --no-2d --stoch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft1d_r22sdf_fxp as F                    # noqa: E402
import fft2d_mask_study as MS                   # noqa: E402
import fft2d_ref as R                           # noqa: E402
import gen_fft2d_stim as G                      # noqa: E402
from fft2d_roundtrip_study import IMAGES        # noqa: E402

I64 = np.int64
TW_ROOT = HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"
_cast_halfup = F._cast
MODE = {"bfly": False, "cmul": False, "out": False, "stoch": False}
RNG = np.random.default_rng(11)

MODES = [("half-up (hoy)", dict(bfly=False, cmul=False, out=False)),
         ("conv. mariposas", dict(bfly=True, cmul=False, out=False)),
         ("conv. mariposas + salida", dict(bfly=True, cmul=False, out=True)),
         ("conv. en todo", dict(bfly=True, cmul=True, out=True))]


def round_conv(x, d):
    q = x >> d
    r = x - (q << d)
    h = I64(1) << (d - 1)
    return q + ((r > h) | ((r == h) & ((q & 1) == 1)))


def round_stoch(x, d):
    q = x >> d
    r = x - (q << d)
    return q + (RNG.integers(0, 1 << d, size=np.shape(x)) < r)


def cast_sel(x, drop, nb, stats=None, name=""):
    """_cast del modelo con el redondeo elegido por punto (por el nombre de la etapa)."""
    point = "bfly" if name.startswith("etapa") else "cmul" if name.startswith("cmul") else "out"
    if drop > 0 and MODE["stoch"] and point != "cmul":
        return _cast_halfup(round_stoch(np.asarray(x, dtype=I64), drop), 0, nb, stats, name)
    if drop > 0 and MODE[point]:
        return _cast_halfup(round_conv(np.asarray(x, dtype=I64), drop), 0, nb, stats, name)
    return _cast_halfup(x, drop, nb, stats, name)


def set_mode(m, stoch=False):
    MODE.update(m)
    MODE["stoch"] = stoch


def self_check():
    v = np.arange(-4096, 4096, dtype=I64)
    for d in (1, 2, 4, 17):
        ref = np.array([int(np.round(t / 2 ** d)) for t in v], dtype=I64)   # np.round: al par
        assert np.array_equal(round_conv(v, d), ref), d


def study_1d():
    n = 1024
    tw = F.load_twiddles(TW_ROOT / f"n{n}_tw18", n, 18)
    rng = np.random.default_rng(3)
    xr, xi = MS.q15(0.3 * (rng.standard_normal((256, n)) + 1j * rng.standard_normal((256, n))))
    ref = np.fft.fft(xr + 1j * xi, axis=-1) / 2 ** 10
    br = G.bitrev(n)
    print("1D, N = 1024, 20/18/16, mascara completa, ruido: error contra la FFT ideal / 2^10 [LSB]")
    for name, m in MODES:
        set_mode(m)
        cfg = F.Cfg(N=n, NB=20, NBF=19, NB_TW=18, NB_IN=16, NB_OUT=16, SCALE_MASK=n - 1, INVERSE=0)
        yr, yi = F.fft1d_r22sdf_fxp(xr, xi, cfg, *tw)
        y = np.empty_like(ref)
        y[..., br] = yr + 1j * yi
        e = y - ref
        mse = np.mean(np.abs(e) ** 2)
        b = (e.real.mean(), e.imag.mean())
        print(f"  {name:26s} sesgo {b[0]:+.4f} / {b[1]:+.4f}   rms {np.sqrt(mse):.4f}   "
              f"sesgo^2 / MSE {100 * (b[0] ** 2 + b[1] ** 2) / mse:4.1f} %")


def roundtrip_setup(n, kind):
    rt = G.RoundTrip(n)
    qr, qi = G.q15(G.make_image(HERE / "gen", kind, n, IMAGES[kind]).astype(np.complex128))
    x = qr + 1j * qi
    s, _ = G.minimal_split(G.boundary_peaks(x, n), rt.log)
    return rt, qr, qi, x, s, [rt.early(v) for v in s]


def study_roundtrip(stoch: bool):
    print("\nIda y vuelta, reparto minimo al principio: SQNR [dB] / |error(0, 0)| [LSB] / saturaciones")
    print("  " + " " * 15 + " | ".join(f"{name:>24s}" for name, _ in MODES))
    for n in (64, 256):
        for kind in IMAGES:
            rt, qr, qi, x, _, masks = roundtrip_setup(n, kind)
            row = []
            for _, m in MODES:
                set_mode(m)
                y, sats, _ = rt.run(qr, qi, masks)
                e = y - x
                sq = 10 * np.log10(np.sum(np.abs(x) ** 2) / np.sum(np.abs(e) ** 2))
                row.append(f"{sq:6.2f} / {abs(e[0, 0]):5.0f} / {sum(sats)}")
            print(f"  N = {n:3d} {kind:9s} " + " | ".join(f"{c:>24s}" for c in row))
    if not stoch:
        return
    print("\n  N = 256, de donde sale el error en (0, 0) (convergente en mariposas y salida):")
    for kind in IMAGES:
        rt, qr, qi, x, s, masks = roundtrip_setup(256, kind)
        set_mode(MODES[2][1])
        X, _, _ = rt.run(qr, qi, masks[:2])
        e_x = X - np.fft.fft2(x) / 2 ** (s[0] + s[1])
        y, _, _ = rt.run(qr, qi, masks)
        e00 = (y - x)[0, 0]
        set_mode(MODES[0][1], stoch=True)
        st = [abs((rt.run(qr, qi, masks)[0] - x)[0, 0]) for _ in range(5)]
        print(f"    {kind:9s} media del error del espectro {e_x.mean().real:+.4f} LSB "
              f"-> suma / 2^(S3+S4) = {np.sum(e_x).real / 2 ** (s[2] + s[3]):+6.0f};  "
              f"error (0, 0) {e00.real:+6.0f};  estocastico {np.mean(st):4.0f} ({min(st):.0f} a {max(st):.0f})")
    set_mode(MODES[0][1])


def study_2d():
    m, smaps, y_true, k_hw = MS.build_phantom(HERE / "gen", 1024, 8, 2)
    ch = MS.Chain(k_hw, smaps, m, 2, TW_ROOT, "bobina")
    ch.set_truth(y_true)
    s1, s2 = bin(801).count("1"), bin(384).count("1")
    gold = R.fft2d_golden((ch.qr + 1j * ch.qi) / MS.FS, s1, s2) * MS.FS          # [l, p, x], LSB
    print("\nFantoma 1024 x 512, 8 bobinas, 801 / 384: error contra la IDFT 2D ideal [LSB]")
    for name, mm in MODES:
        set_mode(mm)
        r = ch.evaluate(801, 384)
        hr, hi = ch.pass1(801)
        ar, ai = ch.pass2(hr, hi, 384)
        a = (ar + 1j * ai)[:, ch.ib1][:, :, ch.ib2]                                 # [l, x, p]
        e = np.swapaxes(a, 1, 2) - gold
        print(f"  {name:26s} fila p = 0: {e[:, 0].real.mean():+.3f} / {e[:, 0].imag.mean():+.3f}   "
              f"resto: {e[:, 1:].real.mean():+.4f} / {e[:, 1:].imag.mean():+.4f}   "
              f"SNR imagen {r['snr_img']:.2f} dB   sat {r['sat1'] + r['sat2']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-2d", action="store_true", help="saltear el fantoma de tamano completo")
    ap.add_argument("--stoch", action="store_true", help="agregar redondeo estocastico y la media del espectro")
    a = ap.parse_args()
    self_check()
    F._cast = cast_sel
    study_1d()
    study_roundtrip(a.stoch)
    if not a.no_2d:
        study_2d()
    return 0


if __name__ == "__main__":
    sys.exit(main())

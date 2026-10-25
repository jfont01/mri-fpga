#!/usr/bin/env python3
"""
fft2d_roundtrip_study.py -- ida y vuelta FFT 2D -> IFFT 2D con el modelo bit a bit
(bitacora §4.12). Estudio, no regresion: no escribe archivos.

Cuatro pasadas del nucleo R2²SDF (fft1d_r22sdf_fxp.py), con interfaces en orden
natural (el reordenamiento entre pasadas es ideal):

  pasada 1  FFT  de filas      (eje x)   mascara m1, S1 etapas que dividen
  pasada 2  FFT  de columnas   (eje y)   mascara m2, S2
  pasada 3  IFFT de filas      (eje x)   mascara m3, S3   (INVERSE = 1)
  pasada 4  IFFT de columnas   (eje y)   mascara m4, S4

Lo que hay en cada frontera se conoce exacto (sin contar el redondeo):

  despues de 1:  FFT de cada fila             / 2^S1
  despues de 2:  FFT 2D                       / 2^(S1+S2)
  despues de 3:  N * FFT de cada columna      / 2^(S1+S2+S3)
  despues de 4:  N^2 * imagen                 / 2^(S1+S2+S3+S4)

Para recuperar la imagen: S1+S2+S3+S4 = 2 log2 N. Para no desbordar, el pico de
cada frontera tiene que quedar < 1 -> minimo acumulado de S en cada frontera.

Compara tres repartos:
  completa / sin escalar   ida con todas las etapas, vuelta sin ninguna
  minimo, al principio     S minimos, las etapas que dividen primero en cada pasada
  minimo, al final         S minimos, las etapas que dividen al final

Uso (desde la raiz del repo):
  python3 model/fft2d_roundtrip_study.py            # N = 256, las 4 imagenes
  python3 model/fft2d_roundtrip_study.py -N 512 --images shepp rings
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft1d_r22sdf_fxp as F                    # noqa: E402
import gen_fft2d_stim as G                      # noqa: E402

GEN_DIR = HERE / "gen"
TW_ROOT = HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"
FS = 32768.0                                     # fondo de escala Q1.15

IMAGES = {
    "shepp": {},
    "sine": {"f": 20, "angle": 25},
    "rings": {"period": 16},
    "zoneplate": {},
}


def bitrev(n: int) -> np.ndarray:
    L = n.bit_length() - 1
    return np.array([int(format(i, f"0{L}b")[::-1], 2) for i in range(n)])


class RoundTrip:
    def __init__(self, n: int, nb: int = 20, nb_tw: int = 18, nb_io: int = 16):
        self.n, self.log = n, n.bit_length() - 1
        self.fmt = dict(N=n, NB=nb, NBF=nb - 1, NB_TW=nb_tw, NBF_TW=nb_tw - 1,
                        NB_IN=nb_io, NBF_IN=nb_io - 1, NB_OUT=nb_io, NBF_OUT=nb_io - 1)
        self.tw = F.load_twiddles(TW_ROOT / f"n{n}_tw{nb_tw}", n, nb_tw)
        self.br = bitrev(n)                      # involucion: br(br(i)) = i

    def _pass(self, xr, xi, mask, inverse, stats):
        """Una pasada sobre el ultimo eje; entrada y salida en orden natural."""
        cfg = F.Cfg(**self.fmt, SCALE_MASK=mask, INVERSE=inverse)
        yr, yi = F.fft1d_r22sdf_fxp(xr, xi, cfg, *self.tw, stats)
        return yr[..., self.br], yi[..., self.br]

    def run(self, qr, qi, masks):
        """Devuelve (salida compleja en LSB, saturaciones por pasada, picos en las 3 fronteras)."""
        sats, peaks = [], []
        r, i = qr, qi
        for k, (mask, inv) in enumerate(zip(masks, (0, 0, 1, 1))):
            st = F.Stats()
            r, i = self._pass(r.T.copy(), i.T.copy(), mask, inv, st) if k % 2 else \
                self._pass(r, i, mask, inv, st)
            if k % 2:
                r, i = r.T, i.T
            sats.append(st.total_sat)
            if k < 3:
                peaks.append(np.abs(r + 1j * i).max() / FS)
        return r + 1j * i, sats, peaks

    # máscaras: bit i = etapa i en orden de tiempo
    def early(self, s: int) -> int:
        return (1 << s) - 1

    def late(self, s: int) -> int:
        return ((1 << s) - 1) << (self.log - s)


def boundary_peaks(x: np.ndarray, n: int):
    """Picos sin escalar en las tres fronteras, en fondo de escala."""
    xf = x / FS
    return (np.abs(np.fft.fft(xf, axis=1)).max(),
            np.abs(np.fft.fft2(xf)).max(),
            n * np.abs(np.fft.fft(xf, axis=0)).max())


def minimal_split(peaks, log: int):
    """S1..S4 minimos para que cada frontera quede < 1 y el total sea 2 log2 N."""
    acc = [math.floor(math.log2(p)) + 1 if p >= 1 else 0 for p in peaks]
    acc = [max(acc[0], 0), max(acc[1], acc[0]), max(acc[2], acc[1])]
    s = [acc[0], acc[1] - acc[0], acc[2] - acc[1], 2 * log - acc[2]]
    ok = all(0 <= v <= log for v in s)
    return s, ok


def sqnr(ref, got):
    return 10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs(got - ref) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-N", type=int, default=256)
    ap.add_argument("--images", nargs="+", default=list(IMAGES), choices=list(IMAGES))
    a = ap.parse_args()

    rt = RoundTrip(a.N)
    full = (1 << rt.log) - 1
    print(f"N = {a.N} ({rt.log} etapas por pasada), formatos 20/18/16, total a dividir = {2 * rt.log}")
    for kind in a.images:
        img = G.make_image(GEN_DIR, kind, a.N, IMAGES[kind])
        qr, qi = G.q15(img.astype(np.complex128))
        x = qr + 1j * qi
        p = boundary_peaks(x, a.N)
        s, ok = minimal_split(p, rt.log)
        print(f"\n{kind}: log2 de los picos sin escalar = "
              + ", ".join(f"{math.log2(v):.2f}" for v in p) + f"  ->  S1..S4 minimos = {s}"
              + ("" if ok else "  (NO ENTRA en una pasada)"))
        pred = [p[0] / 2 ** s[0], p[1] / 2 ** (s[0] + s[1]), p[2] / 2 ** (s[0] + s[1] + s[2])]
        print("   picos predichos en las fronteras: " + ", ".join(f"{v:.3f}" for v in pred))
        configs = [("completa / sin escalar", [full, full, 0, 0])]
        if ok:
            configs += [("minimo, al principio", [rt.early(v) for v in s]),
                        ("minimo, al final", [rt.late(v) for v in s])]
        for name, m in configs:
            y, sats, pk = rt.run(qr, qi, m)
            e00 = abs(y[0, 0] - x[0, 0])
            print(f"   {name:24s} mascaras {[hex(v) for v in m]}  sat {sats}  "
                  f"picos " + ", ".join(f"{v:.3f}" for v in pk)
                  + f"  SQNR {sqnr(x, y):6.2f} dB  |error(0,0)| {e00:.0f} LSB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

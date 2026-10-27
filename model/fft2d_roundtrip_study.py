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

La cadena bit a bit (RoundTrip, boundary_peaks, minimal_split) vive en
gen_fft2d_stim.py, que la usa para los casos roundtrip de la fft2d.

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

import gen_fft2d_stim as G                      # noqa: E402
from gen_fft2d_stim import RoundTrip, boundary_peaks, minimal_split   # noqa: E402

GEN_DIR = HERE / "gen"

IMAGES = {
    "shepp": {},
    "sine": {"f": 20, "angle": 25},
    "rings": {"period": 16},
    "zoneplate": {},
}


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

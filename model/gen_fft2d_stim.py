#!/usr/bin/env python3
"""
gen_fft2d_stim.py -- k-space de estimulo para la fft2d (STIM_TYPE = 3).

Hoy genera el fantoma del paso 4: Shepp-Logan N x N, L mapas de sensibilidad,
k-space submuestreado en phase-encode (Af), con los mismos generadores que
fft2d_mask_study.py. Lo normaliza (por bobina o comun, pico 0,9) y lo cuantiza
a Q1.15.

Escribe dos archivos:
  <out>.bin   int32 little-endian [L][N/Af][N][re, im]: enteros crudos Q1.15,
              en el orden en que entra al hardware (fila k', readout contiguo).
              Es lo que lee fft2d_stimulus.hpp.
  <out>.json  como se genero (N, L, Af, normalizacion, ganancias g_l), para que
              run_plot_fft2d.py pueda rehacer el fantoma y los mapas, chequear
              que el estimulo coincide y desplegar la imagen.

Uso (desde la raiz del repo):
    python3 model/gen_fft2d_stim.py -N 64 -L 8 --out modules/fft2d/testbench/stim/phantom_n64_l8.bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fft2d_mask_study import build_phantom, comp_peak, q15   # noqa: E402


def phantom_kspace(gen_dir: Path, n: int, L: int, af: int, norm: str):
    """(m, smaps, y_true, k_hw, g, qr, qi): lo mismo que arma Chain."""
    m, smaps, y_true, k_hw = build_phantom(gen_dir, n, L, af)
    if norm == "comun":
        g = np.full(L, 0.9 / comp_peak(k_hw))
    else:
        g = np.array([0.9 / comp_peak(k_hw[l]) for l in range(L)])
    qr, qi = q15(k_hw * g[:, None, None])
    return m, smaps, y_true, k_hw, g, qr, qi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-N", type=int, default=64, help="lado de la imagen (N1 = N, N2 = N / Af)")
    ap.add_argument("-L", type=int, default=8, help="bobinas")
    ap.add_argument("--af", type=int, default=2)
    ap.add_argument("--norm", choices=("bobina", "comun"), default="bobina")
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("--out", type=Path, required=True, help="archivo .bin de salida")
    a = ap.parse_args()

    _, _, _, k_hw, g, qr, qi = phantom_kspace(a.gen_dir, a.N, a.L, a.af, a.norm)
    L, n2, n1 = k_hw.shape

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(np.stack([qr, qi], axis=-1).astype("<i4").tobytes())
    meta = {
        "kind": "phantom",
        "N": a.N, "L": L, "af": a.af, "norm": a.norm,
        "N1": n1, "N2": n2,
        "gains": [float(v) for v in g],
        "format": "int32 little-endian [L][N2][N1][re, im], Q1.15",
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{a.out}: {L} bobinas x {n2} filas x {n1}, Q1.15, normalizacion {a.norm}; "
          f"pico {max(np.abs(qr).max(), np.abs(qi).max())} LSB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

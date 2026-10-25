#!/usr/bin/env python3
"""
gen_fft2d_stim.py -- archivos de estimulo de la fft2d (STIM_TYPE = 3).

Dos familias:

  phantom  k-space de SENSE del paso 4: Shepp-Logan N x N, L mapas de
           sensibilidad, submuestreado en phase-encode (Af), normalizado por
           bobina (o comun) a pico 0,9. L bobinas de N/Af filas x N. Va con la
           IFFT2D (INVERSE = 1).
  imagen   una imagen cuadrada N x N, real, para la FFT2D directa (INVERSE = 0):
             shepp      Shepp-Logan (gen/gen_phantom.py, el de scikit-image)
             sine       A cos(2 pi (fx x + fy y) / N): fx, fy en ciclos por
                        imagen; con --f y --angle queda rotada (fx = f cos,
                        fy = f sin). Si fx y fy son enteros el espectro son dos
                        deltas; si no, aparece el derrame (leakage)
             rings      ondas circulares de periodo fijo (gen_concentric_rings_2d):
                        A/2 (1 + cos(2 pi r / periodo)); el espectro es un anillo
             zoneplate  A cos(pi r^2 / N): la frecuencia local crece con r y
                        llega a Nyquist en el borde; el espectro es otra zone plate
           x es la columna (readout, contiguo) e y la fila; r se mide desde el
           centro (N/2, N/2).

Escribe:
  <out>.bin   int32 little-endian [L][filas][columnas][re, im], enteros crudos
              Q1.15 en el orden en que entra al hardware. Lo lee
              fft2d_stimulus.hpp.
  <out>.json  como se genero, para que run_plot_fft2d.py sepa que es.

Uso (desde la raiz del repo):
  # todo lo que pide el JSON de la fft2d (claves "stim_gen" de cada caso)
  python3 model/gen_fft2d_stim.py --from-json modules/fft2d/testbench/fft2d_tb_regression.json
  # uno suelto
  python3 model/gen_fft2d_stim.py --image sine -N 256 --f 20 --angle 30 --out modules/fft2d/testbench/stim/x.bin
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

import fft2d_ref as R                                        # noqa: E402
from fft2d_mask_study import build_phantom, comp_peak, q15   # noqa: E402

AMP = 0.9          # pico de las imagenes, en fondo de escala


# ---------------------------------------------------------------- fantoma SENSE
def phantom_kspace(gen_dir: Path, n: int, L: int, af: int, norm: str):
    """(m, smaps, y_true, k_hw, g, qr, qi): lo mismo que arma Chain."""
    m, smaps, y_true, k_hw = build_phantom(gen_dir, n, L, af)
    if norm == "comun":
        g = np.full(L, 0.9 / comp_peak(k_hw))
    else:
        g = np.array([0.9 / comp_peak(k_hw[l]) for l in range(L)])
    qr, qi = q15(k_hw * g[:, None, None])
    return m, smaps, y_true, k_hw, g, qr, qi


# ---------------------------------------------------------------------- imagenes
def make_image(gen_dir: Path, kind: str, n: int, p: dict) -> np.ndarray:
    """Imagen real [fila y, columna x] con pico <= AMP."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    r = np.hypot(x - n / 2.0, y - n / 2.0)
    if kind == "shepp":
        gp = R._load(gen_dir, "gen_phantom")
        return gp.gen_shepp_logan_2d(N=n, A=AMP)
    if kind == "sine":
        if "f" in p:
            ang = np.deg2rad(float(p.get("angle", 0.0)))
            fx, fy = float(p["f"]) * np.cos(ang), float(p["f"]) * np.sin(ang)
        else:
            fx, fy = float(p.get("fx", 8)), float(p.get("fy", 0))
        return AMP * np.cos(2 * np.pi * (fx * x + fy * y) / n)
    if kind == "rings":
        gp = R._load(gen_dir, "gen_phantom")
        return gp.gen_concentric_rings_2d(N=n, A=AMP, rings_period=float(p.get("period", 16)))
    if kind == "zoneplate":
        return AMP * np.cos(np.pi * r ** 2 / n)
    raise ValueError(f"imagen desconocida: {kind}")


def write(out: Path, qr: np.ndarray, qi: np.ndarray, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(np.stack([qr, qi], axis=-1).astype("<i4").tobytes())
    meta["format"] = "int32 little-endian [L][filas][columnas][re, im], Q1.15"
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    L, rows, cols = qr.shape
    print(f"  {out}: {meta['kind']}"
          f"{' ' + meta['image'] if meta['kind'] == 'image' else ''}, {L} x {rows} x {cols}, "
          f"pico {max(np.abs(qr).max(), np.abs(qi).max())} LSB")


def generate(gen_dir: Path, spec: dict, out: Path) -> None:
    kind = spec["kind"]
    if kind == "phantom":
        n, L, af, norm = int(spec.get("N", 64)), int(spec.get("L", 8)), int(spec.get("af", 2)), spec.get("norm", "bobina")
        _, _, _, k_hw, g, qr, qi = phantom_kspace(gen_dir, n, L, af, norm)
        write(out, qr, qi, {"kind": "phantom", "N": n, "L": L, "af": af, "norm": norm,
                            "N1": k_hw.shape[2], "N2": k_hw.shape[1], "gains": [float(v) for v in g]})
        return
    n = int(spec["N"])
    params = {k: v for k, v in spec.items() if k not in ("kind", "N")}
    img = make_image(gen_dir, kind, n, params)
    qr, _ = q15(img.astype(np.complex128))
    qi = np.zeros_like(qr)
    write(out, qr[None], qi[None], {"kind": "image", "image": kind, "N": n, "A": AMP, **params})


def from_json(gen_dir: Path, json_path: Path) -> int:
    module_dir = json_path.resolve().parent.parent
    data = json.loads(json_path.read_text())
    count = 0
    for case in data.get("cases", []):
        spec = case.get("stim_gen")
        if case.get("enabled", True) is False or spec is None:
            continue
        generate(gen_dir, spec, module_dir / case["STIM_FILE"])
        count += 1
    print(f"{count} archivos de estimulo generados")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", type=Path, default=None, help="generar todo lo que pide un JSON de regresion")
    ap.add_argument("--image", choices=("shepp", "sine", "rings", "zoneplate"), default=None)
    ap.add_argument("-N", type=int, default=64, help="lado de la imagen")
    ap.add_argument("-L", type=int, default=8, help="bobinas (fantoma)")
    ap.add_argument("--af", type=int, default=2)
    ap.add_argument("--norm", choices=("bobina", "comun"), default="bobina")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--f", type=float, default=None, help="frecuencia (ciclos por imagen), con --angle")
    ap.add_argument("--angle", type=float, default=0.0, help="grados")
    ap.add_argument("--period", type=float, default=16.0, help="periodo de los anillos (pixeles)")
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("--out", type=Path, default=None, help="archivo .bin de salida")
    a = ap.parse_args()

    if a.from_json:
        return from_json(a.gen_dir, a.from_json)
    if a.out is None:
        ap.error("falta --out (o --from-json)")
    if a.image is None:
        spec = {"kind": "phantom", "N": a.N, "L": a.L, "af": a.af, "norm": a.norm}
    else:
        spec = {"kind": a.image, "N": a.N}
        if a.image == "sine":
            spec.update({"f": a.f, "angle": a.angle} if a.f is not None else {"fx": a.fx or 8, "fy": a.fy or 0})
        if a.image == "rings":
            spec["period"] = a.period
    generate(a.gen_dir, spec, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

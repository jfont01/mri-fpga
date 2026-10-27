#!/usr/bin/env python3
"""
gen_fft2d_stim.py -- archivos de estimulo de la fft2d (STIM_TYPE = 3).

Tres familias:

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
  roundtrip  la vuelta de la ida y vuelta FFT 2D -> IFFT 2D (bitacora 4.12):
           una de las imagenes de arriba, sus pasadas 1 y 2 (FFT de filas y
           de columnas) bit a bit con fwd_masks = [m1, m2], y el espectro en
           orden natural [ky][kx] como estimulo de una fft2d INVERSE = 1 (las
           pasadas 3 y 4, con SCALE_MASK1 = m3 y SCALE_MASK2 = m4 del caso).
           Con las mascaras del caso corre las 4 pasadas y exige 0
           saturaciones y S1 + S2 + S3 + S4 = 2 log2 N (vuelve la imagen con
           ganancia 1). Avisa si las mascaras no son las de la regla "lo
           justo, al principio de cada pasada". Con ROUND_CONV = 1 en el caso,
           las 4 pasadas (tambien la ida, que hace Python) redondean al par.

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
  python3 model/gen_fft2d_stim.py --image shepp -N 64 --fwd-masks 15 31 --inv-masks 1 3 --out modules/fft2d/testbench/stim/x.bin
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft1d_r22sdf_fxp as F                                 # noqa: E402
import fft2d_ref as R                                        # noqa: E402
from fft2d_mask_study import build_phantom, comp_peak, q15   # noqa: E402

AMP = 0.9          # pico de las imagenes, en fondo de escala
FS = 32768.0       # fondo de escala Q1.15
TW_ROOT = HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"


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


# ----------------------------------------------------------------- ida y vuelta
def bitrev(n: int) -> np.ndarray:
    L = n.bit_length() - 1
    return np.array([int(format(i, f"0{L}b")[::-1], 2) for i in range(n)])


class RoundTrip:
    """Las 4 pasadas del nucleo R2²SDF (fft1d_r22sdf_fxp.py) con interfaces en orden
    natural (reordenamiento ideal entre pasadas). Pasadas 1 y 3 por filas, 2 y 4 por
    columnas; 1 y 2 directas, 3 y 4 inversas (INVERSE = 1)."""

    def __init__(self, n: int, nb: int = 20, nb_tw: int = 18, nb_io: int = 16, round_conv: int = 0):
        self.n, self.log = n, n.bit_length() - 1
        self.fmt = dict(N=n, NB=nb, NBF=nb - 1, NB_TW=nb_tw, NBF_TW=nb_tw - 1,
                        NB_IN=nb_io, NBF_IN=nb_io - 1, NB_OUT=nb_io, NBF_OUT=nb_io - 1,
                        ROUND_CONV=round_conv)
        self.tw = F.load_twiddles(TW_ROOT / f"n{n}_tw{nb_tw}", n, nb_tw)
        self.br = bitrev(n)                      # involucion: br(br(i)) = i

    def _pass(self, xr, xi, mask, inverse, stats):
        """Una pasada sobre el ultimo eje; entrada y salida en orden natural."""
        cfg = F.Cfg(**self.fmt, SCALE_MASK=mask, INVERSE=inverse)
        yr, yi = F.fft1d_r22sdf_fxp(xr, xi, cfg, *self.tw, stats)
        return yr[..., self.br], yi[..., self.br]

    def run(self, qr, qi, masks):
        """Corre tantas pasadas como mascaras (1 a 4). Devuelve (salida compleja en LSB,
        saturaciones por pasada, picos en las fronteras 1 a 3 alcanzadas)."""
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


def roundtrip_chain(gen_dir: Path, spec: dict, inv_masks=None, round_conv: int = 0) -> dict:
    """
    Ida y vuelta bit a bit de la imagen de spec (kind = roundtrip), todo en LSB de
    Q1.15 y en orden natural:
      x   imagen [y, x]
      X   espectro despues de las pasadas 1 y 2 [ky, kx]: el estimulo
      y   salida de las 4 pasadas, si se dan las mascaras de la vuelta [m3, m4]
    rule_masks son las de la regla (lo justo, al principio de cada pasada).
    round_conv = 1: redondeo convergente en las 4 pasadas (ROUND_CONV del nucleo).
    """
    n = int(spec["N"])
    m1, m2 = (_int(v) for v in spec["fwd_masks"])
    params = {k: v for k, v in spec.items() if k not in ("kind", "image", "N", "fwd_masks")}
    rt = RoundTrip(n, round_conv=round_conv)
    qr, qi = q15(make_image(gen_dir, spec["image"], n, params).astype(np.complex128))
    x = qr + 1j * qi
    X, _, _ = rt.run(qr, qi, [m1, m2])
    s, ok = minimal_split(boundary_peaks(x, n), rt.log)
    res = {"n": n, "log": rt.log, "params": params, "x": x, "X": X, "fwd_masks": [m1, m2],
           "rule_masks": [rt.early(v) for v in s] if ok else None}
    if inv_masks is not None:
        masks = [m1, m2] + [_int(v) for v in inv_masks]
        if any(m >> rt.log for m in masks):
            raise ValueError(f"mascaras {masks}: N = {n} tiene {rt.log} etapas por pasada")
        y, sats, peaks = rt.run(qr, qi, masks)
        p_err = np.sum(np.abs(y - x) ** 2)
        res.update(y=y, masks=masks, sats=sats, peaks=peaks,
                   s_total=sum(bin(m).count("1") for m in masks),
                   sqnr_db=float(10 * np.log10(np.sum(np.abs(x) ** 2) / max(p_err, 1e-300))),
                   e00=complex(y[0, 0] - x[0, 0]))
    return res


def _int(v) -> int:
    return int(v, 0) if isinstance(v, str) else int(v)


def write(out: Path, qr: np.ndarray, qi: np.ndarray, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(np.stack([qr, qi], axis=-1).astype("<i4").tobytes())
    meta["format"] = "int32 little-endian [L][filas][columnas][re, im], Q1.15"
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    L, rows, cols = qr.shape
    print(f"  {out}: {meta['kind']}"
          f"{' ' + meta['image'] if 'image' in meta else ''}, {L} x {rows} x {cols}, "
          f"pico {max(np.abs(qr).max(), np.abs(qi).max())} LSB")


def generate_roundtrip(gen_dir: Path, spec: dict, out: Path, case: dict) -> bool:
    """Estimulo de la vuelta y chequeo de las 4 pasadas con las mascaras del caso."""
    name, n = case.get("name", out.stem), int(spec["N"])
    if "SCALE_MASK1" not in case or "SCALE_MASK2" not in case:
        raise ValueError(f"{name}: un caso roundtrip tiene que dar SCALE_MASK1 y SCALE_MASK2 (m3 y m4)")
    if _int(case.get("N1", n)) != n or _int(case.get("N2", n)) != n or _int(case.get("INVERSE", 1)) != 1:
        raise ValueError(f"{name}: un caso roundtrip va con N1 = N2 = {n} e INVERSE = 1")
    rc = _int(case.get("ROUND_CONV", 0))
    r = roundtrip_chain(gen_dir, spec, (case["SCALE_MASK1"], case["SCALE_MASK2"]), round_conv=rc)
    X = r["X"]
    write(out, X.real.astype(np.int64)[None], X.imag.astype(np.int64)[None],
          {"kind": "roundtrip", "image": spec["image"], "N": n, "A": AMP, **r["params"],
           "fwd_masks": r["fwd_masks"], "inv_masks": r["masks"][2:], "sat": r["sats"],
           "peaks": [round(p, 4) for p in r["peaks"]], "sqnr_db": round(r["sqnr_db"], 2),
           "e00_lsb": [r["e00"].real, r["e00"].imag], **({"round_conv": rc} if rc else {}), "spec": spec})
    print(f"      ida y vuelta{' (convergente)' if rc else ''}: mascaras {r['masks']}, S = {r['s_total']}, sat {r['sats']}, "
          f"picos " + ", ".join(f"{p:.3f}" for p in r["peaks"])
          + f", SQNR {r['sqnr_db']:.2f} dB, |error(0,0)| {abs(r['e00']):.0f} LSB")
    if r["rule_masks"] is not None and r["masks"] != r["rule_masks"]:
        print(f"      [INFO] no es la regla 'lo justo, al principio': serian {r['rule_masks']}")
    ok = True
    if r["s_total"] != 2 * r["log"]:
        print(f"      [ERROR] {name}: S1 + S2 + S3 + S4 = {r['s_total']}, tiene que ser {2 * r['log']}")
        ok = False
    if sum(r["sats"]):
        print(f"      [ERROR] {name}: satura (por pasada: {r['sats']})")
        ok = False
    return ok


def generate(gen_dir: Path, spec: dict, out: Path, case: dict | None = None) -> bool:
    kind = spec["kind"]
    if kind == "phantom":
        n, L, af, norm = int(spec.get("N", 64)), int(spec.get("L", 8)), int(spec.get("af", 2)), spec.get("norm", "bobina")
        _, _, _, k_hw, g, qr, qi = phantom_kspace(gen_dir, n, L, af, norm)
        write(out, qr, qi, {"kind": "phantom", "N": n, "L": L, "af": af, "norm": norm,
                            "N1": k_hw.shape[2], "N2": k_hw.shape[1], "gains": [float(v) for v in g]})
        return True
    if kind == "roundtrip":
        return generate_roundtrip(gen_dir, spec, out, case or {})
    n = int(spec["N"])
    params = {k: v for k, v in spec.items() if k not in ("kind", "N")}
    img = make_image(gen_dir, kind, n, params)
    qr, _ = q15(img.astype(np.complex128))
    qi = np.zeros_like(qr)
    write(out, qr[None], qi[None], {"kind": "image", "image": kind, "N": n, "A": AMP, **params})
    return True


def from_json(gen_dir: Path, json_path: Path) -> int:
    module_dir = json_path.resolve().parent.parent
    data = json.loads(json_path.read_text())
    count, failed = 0, []
    for case in data.get("cases", []):
        spec = case.get("stim_gen")
        if case.get("enabled", True) is False or spec is None:
            continue
        if not generate(gen_dir, spec, module_dir / case["STIM_FILE"], case):
            failed.append(case["name"])
        count += 1
    print(f"{count} archivos de estimulo generados")
    if failed:
        print(f"[ERROR] no pasan el chequeo: {', '.join(failed)}")
        return 1
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
    ap.add_argument("--fwd-masks", type=int, nargs=2, default=None, metavar=("M1", "M2"),
                    help="con --image: estimulo roundtrip (espectro con estas mascaras de ida)")
    ap.add_argument("--inv-masks", type=int, nargs=2, default=None, metavar=("M3", "M4"),
                    help="mascaras de la vuelta, para el chequeo (con --fwd-masks)")
    ap.add_argument("--round-conv", action="store_true",
                    help="roundtrip con redondeo convergente en las 4 pasadas (ROUND_CONV = 1)")
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
    case = None
    if a.fwd_masks is not None:
        if a.image is None or a.inv_masks is None:
            ap.error("--fwd-masks va con --image y --inv-masks")
        spec = {"kind": "roundtrip", "image": spec.pop("kind"), **spec, "fwd_masks": a.fwd_masks}
        case = {"N1": a.N, "N2": a.N, "SCALE_MASK1": a.inv_masks[0], "SCALE_MASK2": a.inv_masks[1],
                "ROUND_CONV": int(a.round_conv)}
    return 0 if generate(a.gen_dir, spec, a.out, case) else 1


if __name__ == "__main__":
    sys.exit(main())

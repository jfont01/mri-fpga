#!/usr/bin/env python3
"""
gen_ps_sense_vectors.py -- vectores de referencia de modules/ps_sense (calculo de
U'' en el PS), calculados con model/unfold_ref.py.

Cada caso del JSON de regresion lleva:
    "VEC_DIR": carpeta de salida, relativa a la carpeta del modulo
    "vec_gen": como generarlo
        maps     rand | rand_disk | phantom | phantom_mask
                   rand          complejos normales al azar (seed)
                   rand_disk     rand por un disco (afuera, mapas en 0)
                   phantom       mapas sinteticos de gen/gen_smaps.py (los del fantoma)
                   phantom_mask  phantom por el soporte del Shepp-Logan dilatado
                                 'dilate' pixeles (tipo ESPIRiT)
        n_ro, n_pe, L, af        dimensiones (n_ro y n_pe/af potencias de 2)
        lam      numero (absoluto) o "med": mu * mediana de diag(S^H S) sobre las
                 posiciones con todos los pivotes > 0 (mu = 1e-4 por default)
        k_iters  iteraciones del CORDIC cuya ganancia se pliega (null = no plegar)
        nb       bits de U''_q (16)
        layout   pos_coil_a | coil_pos_a
        g        linspace (g_l de 2e-4 a 3e-4) | sat (g elegido para que el
                 maximo cuantice a 2^(nb-1) y sature: ver make_reference)
        seed     semilla de los mapas al azar

Escribe en VEC_DIR (little-endian, el layout que lee ps_sense_gtest.cpp):
    case.txt       clave valor: dimensiones, config y resultados esperados
                   (los double en hex: float.hex, exactos)
    smaps.bin      complex128 [L][n_ro][n_pe]          (entrada)
    g.bin          float64 [L]                          (entrada)
    u_ref.bin      complex128 [x][p][a][l]              U crudo (NaN en las invalidas)
    d_ref.bin      float64 [x][p][a]                    pivotes
    valid_ref.bin  uint8 [x][p]                         L D L^H valida
    buf_ref.bin    uint32 [buffer_words]                U'' empaquetado

Las posiciones invalidas van con U = 0 antes de scale_quantize (la misma regla
que ps_sense::build_buffer; si no, el NaN entra al maximo).

Uso:
    python3 model/gen_ps_sense_vectors.py --from-json modules/ps_sense/testbench/ps_sense_tb_regression.json
    ... --only r2_l8,ph1024        (solo esos casos)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft2d_ref as R                                                   # noqa: E402
from unfold_ref import cordic_gain, gram, ldlh, ldlh_ok, pack_hw, scale_quantize, solve_U  # noqa: E402

LAYOUTS = ("pos_coil_a", "coil_pos_a")


def dilate(mask: np.ndarray, it: int) -> np.ndarray:
    """Dilatacion binaria con vecindad de 4 (igual que mac_output_study.dilate)."""
    out = mask.copy()
    for _ in range(it):
        prev = out.copy()
        out[1:, :] |= prev[:-1, :]
        out[:-1, :] |= prev[1:, :]
        out[:, 1:] |= prev[:, :-1]
        out[:, :-1] |= prev[:, 1:]
    return out


def make_maps(spec: dict, gen_dir: Path) -> np.ndarray:
    kind = spec["maps"]
    L, n_ro, n_pe = spec["L"], spec["n_ro"], spec["n_pe"]
    if kind in ("rand", "rand_disk"):
        rng = np.random.default_rng(spec.get("seed", 1))
        sm = rng.standard_normal((L, n_ro, n_pe)) + 1j * rng.standard_normal((L, n_ro, n_pe))
        if kind == "rand_disk":
            yy, xx = np.mgrid[:n_ro, :n_pe]
            disk = ((xx - n_pe / 2) / (0.35 * n_pe)) ** 2 + ((yy - n_ro / 2) / (0.45 * n_ro)) ** 2 <= 1
            sm = sm * disk[None]
        return np.ascontiguousarray(sm)
    if kind in ("phantom", "phantom_mask"):
        if n_ro != n_pe:
            raise ValueError("los mapas del fantoma son cuadrados (n_ro = n_pe)")
        gs = R._load(gen_dir, "gen_smaps")
        sm = gs.gen_sensitivity_maps_2d(n_ro, L, radius_factor=1.0, sigma_factor=0.8,
                                        phase_scale=1.0, normalize=True).astype(np.complex128)
        if kind == "phantom_mask":
            gp = R._load(gen_dir, "gen_phantom")
            m = gp.gen_shepp_logan_2d(N=n_ro, A=1.0)
            sm = sm * dilate(np.abs(m) > 0, spec.get("dilate", 8))[None]
        return np.ascontiguousarray(sm)
    raise ValueError(f"maps desconocido: {kind}")


def unfold(sm: np.ndarray, af: int, lam: float):
    """unfold_ref con mapas rectangulares: S[x, p, l, a] = sm[l, x, a*n2 + p]."""
    L, n_ro, n_pe = sm.shape
    S = sm.reshape(L, n_ro, af, n_pe // af).transpose(1, 3, 0, 2)
    Sr, Si = np.ascontiguousarray(S.real), np.ascontiguousarray(S.imag)
    Ar, Ai = gram(Sr, Si, lam)
    Lr, Li, d, dinv = ldlh(Ar, Ai)
    with np.errstate(invalid="ignore", over="ignore"):
        Ur, Ui = solve_U(Lr, Li, dinv, Sr, Si)
    return Ur, Ui, d, ldlh_ok(d), Ar


def make_reference(spec: dict, gen_dir: Path) -> dict:
    L, af, nb = spec["L"], spec["af"], spec.get("nb", 16)
    layout = spec.get("layout", "pos_coil_a")
    if layout not in LAYOUTS:
        raise ValueError(f"layout desconocido: {layout}")
    k_iters = spec.get("k_iters", 12)
    K = cordic_gain(k_iters) if k_iters is not None else None
    sm = make_maps(spec, gen_dir)

    lam = spec.get("lam", 0.0)
    if lam == "med":
        _, _, _, _, A0r = unfold(sm, af, 0.0)
        diag = A0r[..., np.arange(af), np.arange(af)]
        lam = spec.get("mu", 1e-4) * float(np.median(A0r[..., 0, 0][(diag > 0).all(-1)]))
    lam = float(lam)

    Ur, Ui, d, valid, _ = unfold(sm, af, lam)
    vm = valid[..., None, None]
    Ur0, Ui0 = np.where(vm, Ur, 0.0), np.where(vm, Ui, 0.0)

    g = np.linspace(2e-4, 3e-4, L)
    if spec.get("g", "linspace") == "sat":
        # El maximo |componente| de U' tiene que ser POSITIVO (el negativo llega a
        # -2^(nb-1) sin saturar). S -> -S da U -> -U exacto (la negacion conmuta con
        # el redondeo), asi que si es negativo se niegan los mapas. Despues se elige g
        # para que el maximo quede en 1 - 2^-17: E = 0 y rint(max * 2^(nb-1)) = 2^(nb-1).
        _, _, _, c0 = scale_quantize(Ur0, Ui0, g, K, nb)
        both = np.concatenate([(Ur0 * c0).ravel(), (Ui0 * c0).ravel()])
        if both[np.argmax(np.abs(both))] < 0:
            sm = -sm
            Ur, Ui, d, valid, _ = unfold(sm, af, lam)
            vm = valid[..., None, None]
            Ur0, Ui0 = np.where(vm, Ur, 0.0), np.where(vm, Ui, 0.0)
        m0 = float(np.abs(both).max())
        g = g * (m0 / (1.0 - 2.0 ** -17))

    qr, qi, E, c = scale_quantize(Ur0, Ui0, g, K, nb)
    buf = pack_hw(qr, qi, layout)
    Upr, Upi = Ur0 * c, Ui0 * c
    maxc = float(max(np.abs(Upr).max(), np.abs(Upi).max()))
    raw = np.concatenate([np.rint(np.ldexp(Upr, nb - 1 - E)).ravel(),
                          np.rint(np.ldexp(Upi, nb - 1 - E)).ravel()])
    n_sat = int(((raw > (1 << (nb - 1)) - 1) | (raw < -(1 << (nb - 1)))).sum())
    if spec.get("g") == "sat" and n_sat == 0:
        raise RuntimeError("caso sat: no saturo ninguna componente")
    dv = d[valid]
    return dict(sm=sm, g=g, Ur=Ur, Ui=Ui, d=d, valid=valid, buf=buf, lam=lam, K=K,
                k_iters=k_iters, nb=nb, layout=LAYOUTS.index(layout), E=int(E), maxc=maxc,
                d_min=float(dv.min()) if dv.size else 0.0, d_max=float(dv.max()) if dv.size else 0.0,
                n_invalid=int((~valid).sum()), n_sat=n_sat)


def write_case(out: Path, spec: dict, ref: dict) -> int:
    out.mkdir(parents=True, exist_ok=True)
    L, n_ro, n_pe, af = spec["L"], spec["n_ro"], spec["n_pe"], spec["af"]
    lines = [
        f"maps {spec['maps']}",
        f"n_ro {n_ro}", f"n_pe {n_pe}", f"L {L}", f"af {af}",
        f"lam {ref['lam'].hex()}",
        f"k_iters {ref['k_iters'] if ref['k_iters'] is not None else 0}",
        f"k_cordic {(ref['K'] if ref['K'] is not None else 0.0).hex()}",
        f"nb {ref['nb']}", f"layout {ref['layout']}",
        f"E {ref['E']}", f"max_abs {ref['maxc'].hex()}",
        f"d_min {ref['d_min'].hex()}", f"d_max {ref['d_max'].hex()}",
        f"n_invalid {ref['n_invalid']}", f"n_sat {ref['n_sat']}",
        f"words {ref['buf'].size}",
    ]
    (out / "case.txt").write_text("\n".join(lines) + "\n")
    np.ascontiguousarray(ref["sm"], dtype=np.complex128).tofile(out / "smaps.bin")
    np.ascontiguousarray(ref["g"], dtype=np.float64).tofile(out / "g.bin")
    u = np.empty(ref["Ur"].shape, dtype=np.complex128)        # [x, p, a, l]
    u.real, u.imag = ref["Ur"], ref["Ui"]                     # sin pasar por Ur + 1j*Ui
    u.tofile(out / "u_ref.bin")
    np.ascontiguousarray(ref["d"], dtype=np.float64).tofile(out / "d_ref.bin")
    ref["valid"].astype(np.uint8).tofile(out / "valid_ref.bin")
    ref["buf"].astype("<u4").tofile(out / "buf_ref.bin")
    return sum(f.stat().st_size for f in out.iterdir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", type=Path, required=True, help="JSON de regresion del modulo")
    ap.add_argument("--only", default="", help="casos separados por coma")
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    args = ap.parse_args()

    module_dir = args.from_json.resolve().parent.parent
    data = json.loads(args.from_json.read_text())
    only = {s for s in args.only.split(",") if s}
    n = 0
    for case in data.get("cases", []):
        name = case.get("name")
        if only and name not in only:
            continue
        if "vec_gen" not in case or "VEC_DIR" not in case:
            continue
        spec = dict(case["vec_gen"])
        t0 = time.time()
        ref = make_reference(spec, args.gen_dir)
        size = write_case(module_dir / case["VEC_DIR"], spec, ref)
        n += 1
        print(f"  {name:16s} -> {case['VEC_DIR']:28s} Af={spec['af']} L={spec['L']} "
              f"{spec['n_ro']}x{spec['n_pe']} lam={ref['lam']:.3g} E={ref['E']} "
              f"invalidas={ref['n_invalid']} sat={ref['n_sat']} {size / 2**20:.2f} MiB "
              f"({time.time() - t0:.1f} s)", flush=True)
    if only and n != len(only):
        print(f"[ERROR] casos pedidos {sorted(only)}, generados {n}", file=sys.stderr)
        return 1
    print(f"{n} casos")
    return 0


if __name__ == "__main__":
    sys.exit(main())

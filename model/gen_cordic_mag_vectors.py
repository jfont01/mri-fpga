#!/usr/bin/env python3
"""
gen_cordic_mag_vectors.py -- vectores de referencia de modules/cordic_mag (paso
D2): magnitud por CORDIC y redondeo de salida, calculados con
model/mac_cordic_fxp.py (m_to_out), bit a bit.

Parametros del diseno. Salen de las claves en MAYUSCULAS del caso (con las de
"defaults" del JSON), las mismas que run_gtest y run_regression_sim pasan como
-DCORDIC_MAG_<KEY>. Las que faltan toman el default de mac_cordic_fxp.OutCfg,
que es el mismo del modelo C++:
    NB_OUT      16      (OutCfg.W_OUT)
    G           5
    N_IT        12
    NB_M        NB_OUT + 1 + G (OutCfg.W_M = None)
    ROUND_CONV  0       (OutCfg.ROUND_CONV_OUT)
case.txt guarda los que se usaron y el gtest los compara con los compilados: si
alguien cambia un default de un lado solo, o el JSON sin regenerar, falla ahi.

Cada caso del JSON lleva ademas:
    "VEC_DIR": carpeta de salida, relativa a la carpeta del modulo
    "vec_gen": que valores de M generar
        seed     semilla (numpy default_rng); el caso es determinista
        parts    lista de "tipo" o "tipo:n", en este orden:
                   rand:n      re, im uniformes en todo el rango de NB_M bits
                               (con NB_M por default, ~71 % satura la salida)
                   nosat:n     |M| uniforme en [0, R_sat), angulo uniforme: el
                               rango que no satura, con todas las salidas
                   grid        todos los pares de {min, min + 1, -2^(NB_M-2), -1,
                               0, 1, 2^(NB_M-2), max - 1, max}: las cuatro
                               esquinas, el minimo en un solo eje, el cero
                   lines       ejes y diagonales: (v, 0), (0, v), (v, v), (v, -v)
                               y sus opuestos, con v = 2^k - 1, 2^k, 2^k + 1 y max
                   small       todos los pares con |re|, |im| <= 2^G (incluye
                               |M| < 2^G: salida 0 o 1)
                   sat_edge:n  el borde de saturacion: |M| = R_sat +- 4 LSB de
                               salida, angulo al azar, mas los ejes en
                               R_sat +- 3 LSB de salida
                   ties:n      empates del redondeo de salida (bits bajos de x_n
                               = 2^(G-1)), mitad con x_n >> G par (ahi difieren
                               half-up y convergente) y mitad impar; necesita G >= 1
        shuffle  mezclar el orden (default true; con la misma semilla)

    R_sat = X_sat / K_n, con X_sat = 2^(NB_OUT+G) - floor(2^(G-1)) el x_n mas
    chico cuya salida satura (en los dos modos de redondeo: el empate de
    X_sat tiene x_n >> G = 2^NB_OUT - 1, impar, y sube tambien en convergente).

Las claves GAP_PCT y GAP_SEED (huecos en i_valid) no le importan al generador:
el estimulo ciclo a ciclo lo arma testbench/cordic_mag_stimulus.hpp.

Escribe en VEC_DIR (little-endian, lo lee cordic_mag_stimulus.hpp):
    case.txt      clave valor: parametros, n, semilla, partes y contadores
                  (x_sat, n_sat, n_ties, n_ties_even, n_zero)
    m.bin         int32 [n][re, im]   entrada (NB_M bits con signo)
    xn_ref.bin    int32 [n]           x_n (antes del redondeo de salida)
    out_ref.bin   int32 [n]           o_mag (NB_OUT bits sin signo)

Uso:
    python3 model/gen_cordic_mag_vectors.py --from-json modules/cordic_mag/testbench/cordic_mag_tb_regression.json
    ... --only def_rand,conv_ties        (solo esos casos)
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

from mac_cordic_fxp import OutCfg, cordic_gain, m_to_out, round_shift  # noqa: E402

I64 = np.int64
PART_KINDS = ("rand", "nosat", "grid", "lines", "small", "sat_edge", "ties")
SIZED = ("rand", "nosat", "sat_edge", "ties")          # los que llevan ":n"


def design_params(defaults: dict, case: dict) -> dict:
    """NB_OUT, G, N_IT, NB_M y ROUND_CONV del caso, con los defaults de OutCfg."""
    p = {k: v for k, v in defaults.items() if k.isupper()}
    p.update({k: v for k, v in case.items() if k.isupper()})
    base = OutCfg(D=0)
    rc = int(p.get("ROUND_CONV", int(base.ROUND_CONV_OUT)))
    if rc not in (0, 1):
        raise ValueError(f"ROUND_CONV = {rc}: tiene que ser 0 o 1")
    cfg = OutCfg(D=0, G=int(p.get("G", base.G)), N_IT=int(p.get("N_IT", base.N_IT)),
                 W_OUT=int(p.get("NB_OUT", base.W_OUT)),
                 W_M=int(p["NB_M"]) if "NB_M" in p else None, ROUND_CONV_OUT=bool(rc))
    # los limites del modelo C++ (static_assert), recortados para que M, x_n
    # (NB_M + 1 bits) y la salida entren en los int32 de los .bin
    if not (1 <= cfg.W_OUT <= 31 and 0 <= cfg.G <= 30 and 1 <= cfg.N_IT <= 32 and 2 <= cfg.W_M <= 30):
        raise ValueError(f"parametros fuera de rango para vectores int32: {cfg}")
    return dict(nb_out=cfg.W_OUT, g=cfg.G, n_it=cfg.N_IT, nb_m=cfg.W_M, round_conv=int(cfg.ROUND_CONV_OUT))


def x_sat(prm: dict) -> int:
    """x_n mas chico cuya salida satura (verificado con round_shift en los dos modos)."""
    g, nb_out = prm["g"], prm["nb_out"]
    xs = (1 << (nb_out + g)) - ((1 << (g - 1)) if g >= 1 else 0)
    umax = (1 << nb_out) - 1
    for conv in (False, True):
        lo, hi = round_shift(np.array([xs - 1, xs], dtype=I64), g, conv)
        if not (int(lo) <= umax < int(hi)):
            raise AssertionError(f"x_sat = {xs} mal calculado (conv = {conv})")
    return xs


def parse_parts(parts: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in parts:
        kind, _, num = str(item).partition(":")
        if kind not in PART_KINDS:
            raise ValueError(f"parte desconocida: {item!r} (validas: {', '.join(PART_KINDS)})")
        if (kind in SIZED) != bool(num):
            raise ValueError(f"parte {item!r}: " + ("falta ':n'" if kind in SIZED else "no lleva ':n'"))
        out.append((kind, int(num) if num else 0))
    return out


def polar(rng, r, n):
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.rint(r * np.cos(th)).astype(I64), np.rint(r * np.sin(th)).astype(I64)


def make_part(kind: str, n: int, prm: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    nb_m, g, n_it = prm["nb_m"], prm["g"], prm["n_it"]
    hm = 1 << (nb_m - 1)
    smin, smax = -hm, hm - 1
    kn = cordic_gain(n_it)
    r_sat = x_sat(prm) / kn
    if kind == "rand":
        return rng.integers(smin, smax + 1, n, dtype=I64), rng.integers(smin, smax + 1, n, dtype=I64)
    if kind == "nosat":
        return polar(rng, rng.uniform(0.0, r_sat, n), n)
    if kind == "grid":
        vals = sorted({smin, smin + 1, -(hm >> 1), -1, 0, 1, hm >> 1, smax - 1, smax})
        re, im = np.meshgrid(np.array(vals, dtype=I64), np.array(vals, dtype=I64), indexing="ij")
        return re.ravel(), im.ravel()
    if kind == "lines":
        vpos = sorted({v for k in range(nb_m - 1) for v in ((1 << k) - 1, 1 << k, (1 << k) + 1)
                       if 0 < v <= smax} | {smax})
        pts = []
        for v in vpos:
            for a, b in ((v, 0), (0, v), (v, v), (v, -v)):
                pts += [(a, b), (-a, -b)]
        arr = np.array(pts, dtype=I64)
        return arr[:, 0], arr[:, 1]
    if kind == "small":
        v = np.arange(-(1 << g), (1 << g) + 1, dtype=I64)
        re, im = np.meshgrid(v, v, indexing="ij")
        return re.ravel(), im.ravel()
    if kind == "sat_edge":
        w = 4.0 * (1 << g) / kn
        re, im = polar(rng, r_sat + rng.uniform(-w, w, n), n)
        win = np.arange(int(np.floor(r_sat - 0.75 * w)), int(np.ceil(r_sat + 0.75 * w)) + 1, dtype=I64)
        z = np.zeros_like(win)
        re = np.concatenate([re, win, z, -win, z])
        im = np.concatenate([im, z, win, z, -win])
        return np.clip(re, smin, smax), np.clip(im, smin, smax)
    if kind == "ties":
        if g < 1:
            raise ValueError("ties: con G = 0 no hay redondeo de salida (no hay empates)")
        mask, half = (1 << g) - 1, 1 << (g - 1)
        want = {0: n // 2, 1: n - n // 2}                 # x_n >> G par / impar
        got = {0: [], 1: []}
        for _ in range(1000):
            if all(len(got[p]) >= want[p] for p in got):
                break
            cr, ci = polar(rng, rng.uniform(0.0, r_sat, 1 << 16), 1 << 16)
            xn = m_to_out(cr, ci, n_it, g, prm["nb_out"], bool(prm["round_conv"]))[1]
            tie = (xn & mask) == half
            for p in (0, 1):
                sel = tie & (((xn >> g) & 1) == p)
                got[p] += list(zip(cr[sel].tolist(), ci[sel].tolist()))
        if any(len(got[p]) < want[p] for p in got):
            raise RuntimeError("ties: no se encontraron suficientes empates")
        pts = np.array(got[0][:want[0]] + got[1][:want[1]], dtype=I64).reshape(-1, 2)
        return pts[:, 0], pts[:, 1]
    raise ValueError(kind)


def make_reference(spec: dict, prm: dict) -> dict:
    rng = np.random.default_rng(int(spec.get("seed", 1)))
    parts = parse_parts(spec.get("parts", []))
    if not parts:
        raise ValueError("vec_gen.parts vacio")
    res, ims = [], []
    for kind, n in parts:
        re, im = make_part(kind, n, prm, rng)
        res.append(re)
        ims.append(im)
    mr, mi = np.concatenate(res), np.concatenate(ims)
    if spec.get("shuffle", True):
        perm = rng.permutation(mr.size)
        mr, mi = mr[perm], mi[perm]

    hm = 1 << (prm["nb_m"] - 1)
    if mr.min() < -hm or mi.min() < -hm or mr.max() >= hm or mi.max() >= hm:
        raise AssertionError("M fuera de NB_M bits")
    out, xn = m_to_out(mr, mi, prm["n_it"], prm["g"], prm["nb_out"], bool(prm["round_conv"]))
    out, xn = np.asarray(out, dtype=I64), np.asarray(xn, dtype=I64)
    if out.min() < 0 or out.max() > (1 << prm["nb_out"]) - 1:
        raise AssertionError("salida fuera de NB_OUT bits")
    if xn.min() < 0 or xn.max() >= 1 << (prm["nb_m"] + 1):
        raise AssertionError("x_n fuera de NB_M + 1 bits")

    xs = x_sat(prm)
    g = prm["g"]
    if g >= 1:
        tie = (xn & ((1 << g) - 1)) == (1 << (g - 1))
        n_ties, n_ties_even = int(tie.sum()), int((tie & (((xn >> g) & 1) == 0)).sum())
    else:
        n_ties = n_ties_even = 0
    stats = dict(x_sat=xs, n_sat=int((xn >= xs).sum()), n_ties=n_ties, n_ties_even=n_ties_even,
                 n_zero=int((out == 0).sum()))

    kinds = {k for k, _ in parts}
    if "sat_edge" in kinds:
        # a los dos lados del borde, y el borde exacto de x_n (X_sat - 1 no satura, X_sat si)
        if not ((xn == xs - 1).any() and (xn == xs).any()):
            raise RuntimeError("sat_edge: no aparecen x_n = X_sat - 1 y x_n = X_sat")
        if not ((xn < xs) & (out == (1 << prm["nb_out"]) - 1)).any():
            raise RuntimeError("sat_edge: no aparece la salida maxima sin saturar")
    if "ties" in kinds and n_ties_even == 0:
        raise RuntimeError("ties: sin empates con x_n >> G par")
    return dict(mr=mr, mi=mi, xn=xn, out=out, stats=stats)


def write_case(out_dir: Path, spec: dict, prm: dict, ref: dict) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = ref["stats"]
    lines = [
        f"nb_out {prm['nb_out']}", f"g {prm['g']}", f"n_it {prm['n_it']}",
        f"nb_m {prm['nb_m']}", f"round_conv {prm['round_conv']}",
        f"n {ref['mr'].size}", f"seed {int(spec.get('seed', 1))}",
        f"parts {','.join(str(p) for p in spec['parts'])}",
        f"shuffle {int(bool(spec.get('shuffle', True)))}",
        f"x_sat {s['x_sat']}", f"n_sat {s['n_sat']}", f"n_ties {s['n_ties']}",
        f"n_ties_even {s['n_ties_even']}", f"n_zero {s['n_zero']}",
    ]
    (out_dir / "case.txt").write_text("\n".join(lines) + "\n")
    m = np.empty((ref["mr"].size, 2), dtype="<i4")
    m[:, 0], m[:, 1] = ref["mr"], ref["mi"]
    m.tofile(out_dir / "m.bin")
    ref["xn"].astype("<i4").tofile(out_dir / "xn_ref.bin")
    ref["out"].astype("<i4").tofile(out_dir / "out_ref.bin")
    return sum(f.stat().st_size for f in out_dir.iterdir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", type=Path, required=True, help="JSON de regresion del modulo")
    ap.add_argument("--only", default="", help="casos separados por coma")
    args = ap.parse_args()

    module_dir = args.from_json.resolve().parent.parent
    data = json.loads(args.from_json.read_text())
    defaults = data.get("defaults", {})
    only = {s for s in args.only.split(",") if s}
    n = 0
    for case in data.get("cases", []):
        name = case.get("name")
        if only and name not in only:
            continue
        if "vec_gen" not in case or "VEC_DIR" not in case:
            continue
        spec = dict(case["vec_gen"])
        prm = design_params(defaults, case)
        t0 = time.time()
        ref = make_reference(spec, prm)
        size = write_case(module_dir / case["VEC_DIR"], spec, prm, ref)
        n += 1
        s = ref["stats"]
        print(f"  {name:12s} -> {case['VEC_DIR']:28s} NB_OUT={prm['nb_out']} G={prm['g']} "
              f"N_IT={prm['n_it']} NB_M={prm['nb_m']} RC={prm['round_conv']}  n={ref['mr'].size} "
              f"sat={s['n_sat']} empates={s['n_ties']} (pares {s['n_ties_even']}) ceros={s['n_zero']} "
              f"{size / 1024:.0f} KiB ({time.time() - t0:.1f} s)", flush=True)
    if only and n != len(only):
        print(f"[ERROR] casos pedidos {sorted(only)}, generados {n}", file=sys.stderr)
        return 1
    print(f"{n} casos")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
gen_unfold_vectors.py -- vectores de referencia de modules/unfold (paso D2): MAC
de desplegado y ajuste de P a M, calculados con model/mac_cordic_fxp.py (mac y
p_to_m), bit a bit, y con la regla de cierre de posicion de la decision 13.

Parametros del diseno. Salen de las claves en MAYUSCULAS del caso (con las de
"defaults" del JSON), las mismas que run_gtest y run_regression_sim pasan como
-DUNFOLD_<KEY>. Las que faltan toman el default del modelo C++ (G y NB_M, de
mac_cordic_fxp.OutCfg, que es el mismo):
    L           8
    AF          2
    NB_A        16
    NB_U        16
    G           5       (OutCfg.G)
    NB_M        17 + G  (OutCfg.W_M con W_OUT = 16)
    ROUND_CONV  0       (OutCfg.ROUND_CONV_M)
    NB_D        6
    E2E         0       (1: caso punta a punta, ver "phantom")
case.txt guarda los que se usaron (y NB_P = NB_A + NB_U + 1 + ceil(log2 L)) y
el gtest los compara con los compilados: si alguien cambia un default de un
lado solo, o el JSON sin regenerar, falla ahi.

El productor. Cada posicion plegada son L muestras (bobinas l = 0 ... L - 1),
cada una con A_l (re, im), sus AF coeficientes U''_{a,l} (re, im), i_last = 1
solo en la bobina L - 1 e i_d = D de la posicion. Despues, opcionalmente, se
pierden o se repiten muestras en el camino ("faults"), con el i_last que puso el
productor. Lo esperado se calcula con la MISMA regla de cierre que el modelo:
un contador l de muestras validas; la posicion se cierra con la muestra que trae
i_last o con la numero L - 1, lo que pase primero; err = las dos condiciones no
coinciden; D = i_d de la muestra que cierra; P = suma de U'' * A de las muestras
del grupo, con los coeficientes que viajan con cada muestra.

Cada caso del JSON lleva ademas:
    "VEC_DIR": carpeta de salida, relativa a la carpeta del modulo
    "vec_gen": que posiciones generar
        seed     semilla (numpy default_rng); el caso es determinista
        parts    lista de "tipo" o "tipo:n" (n = posiciones), en este orden:
                   rand:n     A y U'' uniformes en todo su rango
                   scaled:n   A y U'' con un ancho al azar por posicion (1 ... NB
                              bits): P de 2^2 a 2^(NB_P-1), para que todo s
                              tenga resultados sin saturar
                   corner     los 16 vertices (ur, ui, ar, ai) en {min, max}, los
                              mismos en las L bobinas (el coeficiente de a usa el
                              vertice (c + 5 a) mod 16); incluye la esquina de H31,
                              todo en el minimo: P_re = 0, P_im = +L 2^(NB_A+NB_U-1)
                   dsweep:k   k posiciones "scaled" por cada D de 0 a 2^NB_D - 1
                              (s = D - G de -G a 2^NB_D - 1 - G, con el 0)
                   sat:n      P armado para caer cerca de los bordes de saturacion
                              de M (+-2^(NB_M-1)), adentro y afuera, y muy afuera;
                              D al azar entre los que alcanzan el borde
                   ties:n     empates de P -> M: bits bajos de P = 2^(s-1), mitad
                              con floor(P / 2^s) par (ahi difieren half-up y
                              convergente) y mitad impar; s >= 1 al azar
                   phantom    caso punta a punta (E2E = 1): la cadena del fantoma
                              de mac_output_study.py, seccion D (ver abajo)
                 sat y ties arman P exacto con una bobina "fina" (A = 1) y L - 1
                 "gruesas" (A = -2^k, k = min(NB_A, NB_U) - 1), en orden al
                 azar: necesitan L >= 2.
        d        D de las posiciones de rand, scaled y corner: un entero o "rand"
                 (uniforme en [0, 2^NB_D - 1] por posicion). Default 13 (el D
                 del fantoma).
        d_junk   true: las muestras validas sin i_last llevan un i_d al azar (el
                 modelo lo tiene que tomar solo de la que cierra). Default false.
        faults   lista de "drop:n" (se pierde una muestra) y "dup:n" (se repite
                 una muestra); alternan una bobina que no es la ultima y la
                 ultima (la del i_last). Separadas por al menos 3 posiciones y
                 nunca en las 2 ultimas, asi el stream termina alineado.
        shuffle  mezclar el orden de las posiciones (default true; con la misma
                 semilla), antes de las fallas
        rows     (phantom) filas x de la imagen a recortar: cada una son N2 = 512
                 posiciones (t1 = br(x), t2 = 0 ... 511, en el orden del stream)

    phantom: fantoma 1024 x 1024, L = 8, Af = 2, mascaras 801 / 384, A_l a la
    salida de la fft2d bit a bit (cadena de Python, la misma de los estudios),
    U por L D L^H con lambda = 0 y U'' cuantizado como en el PS
    (scale_quantize con 1/K_n, n = N_IT de cordic_mag, 16 bits, layout
    pos_coil_a), D = 49 - S - E - F_OUT con F_OUT = 15. Ademas de lo de unfold,
    escribe la magnitud de mac_cordic_fxp.output_stage (OutCfg(D, G, N_IT,
    NB_M, NB_OUT = 16)) para el test punta a punta del gtest.

Las claves GAP_PCT y GAP_SEED (huecos en i_valid) no le importan al generador:
el estimulo ciclo a ciclo lo arma testbench/unfold_stimulus.hpp.

Escribe en VEC_DIR (little-endian, lo lee unfold_stimulus.hpp):
    case.txt       clave valor: parametros, n, n_out, semilla, partes y contadores
    stream.bin     int32 [n][2 + 2 AF + 2]: a_re, a_im, u_re[0], u_im[0], ...,
                   u_re[AF-1], u_im[AF-1], last, d (las muestras validas, en orden)
    p_ref.bin      int64 [n_out][AF][2]   P (re, im) de cada posicion cerrada
    m_ref.bin      int32 [n_out][AF][2]   M (re, im)
    close_ref.bin  int32 [n_out][2]       indice de la muestra que cierra, err
  con E2E = 1, tambien:
    mag_ref.bin    int32 [n_out][AF]      salida de cordic_mag (output_stage)
    xn_ref.bin     int32 [n_out][AF]      x_n del CORDIC

Uso:
    python3 model/gen_unfold_vectors.py --from-json modules/unfold/testbench/unfold_tb_regression.json
    ... --only def_rand,faults        (solo esos casos)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mac_cordic_fxp import OutCfg, mac, output_stage, p_to_m, round_shift  # noqa: E402

I64 = np.int64
PART_KINDS = ("rand", "scaled", "corner", "dsweep", "sat", "ties", "phantom")
SIZED = ("rand", "scaled", "dsweep", "sat", "ties")          # los que llevan ":n"
DEFAULTS = dict(L=8, AF=2, NB_A=16, NB_U=16, NB_D=6)          # los de OutCfg aparte
F_OUT = 15                                                    # phantom: bits fraccionales de la salida
PHANTOM = dict(n=1024, L=8, af=2, m1=801, m2=384)


# ----------------------------------------------------------------- parametros
def design_params(defaults: dict, case: dict) -> dict:
    """Parametros del caso con los defaults del modelo C++, y los derivados."""
    p = {k: v for k, v in defaults.items() if k.isupper()}
    p.update({k: v for k, v in case.items() if k.isupper()})
    g = int(p.get("G", OutCfg(D=0).G))
    base = OutCfg(D=0, G=g)
    prm = dict(l=int(p.get("L", DEFAULTS["L"])), af=int(p.get("AF", DEFAULTS["AF"])),
               nb_a=int(p.get("NB_A", DEFAULTS["NB_A"])), nb_u=int(p.get("NB_U", DEFAULTS["NB_U"])),
               g=g, nb_m=int(p.get("NB_M", base.W_M)),
               round_conv=int(p.get("ROUND_CONV", int(base.ROUND_CONV_M))),
               nb_d=int(p.get("NB_D", DEFAULTS["NB_D"])), e2e=int(p.get("E2E", 0)))
    prm["nb_p"] = prm["nb_a"] + prm["nb_u"] + 1 + math.ceil(math.log2(prm["l"]))
    prm["s_min"] = -prm["g"]
    prm["s_max"] = (1 << prm["nb_d"]) - 1 - prm["g"]
    # los limites del modelo C++ (static_assert), con NB_M <= 32 para los int32 de m_ref.bin
    ok = (1 <= prm["l"] <= 256 and 1 <= prm["af"] <= 16 and 2 <= prm["nb_a"] <= 31 and 2 <= prm["nb_u"] <= 31
          and 0 <= prm["g"] <= 30 and 2 <= prm["nb_m"] <= 32 and prm["round_conv"] in (0, 1)
          and 1 <= prm["nb_d"] <= 7 and prm["nb_p"] + prm["g"] <= 63 and prm["s_max"] <= 62
          and prm["e2e"] in (0, 1))
    if not ok:
        raise ValueError(f"parametros fuera de rango: {prm}")
    return prm


def parse_list(items: list[str], kinds: tuple[str, ...], sized: tuple[str, ...], what: str) -> list[tuple[str, int]]:
    out = []
    for item in items:
        kind, _, num = str(item).partition(":")
        if kind not in kinds:
            raise ValueError(f"{what} desconocida: {item!r} (validas: {', '.join(kinds)})")
        if (kind in sized) != bool(num):
            raise ValueError(f"{what} {item!r}: " + ("falta ':n'" if kind in sized else "no lleva ':n'"))
        out.append((kind, int(num) if num else 0))
    return out


# ----------------------------------------------------------------- posiciones
# Cada parte devuelve (A [npos, L, 2], U [npos, L, AF, 2], D [npos]), enteros.

def srange(nb: int) -> tuple[int, int]:
    return -(1 << (nb - 1)), (1 << (nb - 1)) - 1


def d_values(spec_d, n: int, prm: dict, rng) -> np.ndarray:
    if spec_d == "rand":
        return rng.integers(0, 1 << prm["nb_d"], n, dtype=I64)
    d = int(spec_d)
    if not 0 <= d < (1 << prm["nb_d"]):
        raise ValueError(f"d = {d} fuera de NB_D = {prm['nb_d']} bits")
    return np.full(n, d, dtype=I64)


def part_rand(n, prm, rng):
    L, af = prm["l"], prm["af"]
    lo_a, hi_a = srange(prm["nb_a"])
    lo_u, hi_u = srange(prm["nb_u"])
    A = rng.integers(lo_a, hi_a + 1, (n, L, 2), dtype=I64)
    U = rng.integers(lo_u, hi_u + 1, (n, L, af, 2), dtype=I64)
    return A, U


def part_scaled(n, prm, rng):
    """Un ancho al azar por posicion para A y otro para U''."""
    L, af = prm["l"], prm["af"]
    wa = rng.integers(1, prm["nb_a"] + 1, n)
    wu = rng.integers(1, prm["nb_u"] + 1, n)
    A = np.empty((n, L, 2), dtype=I64)
    U = np.empty((n, L, af, 2), dtype=I64)
    for i in range(n):
        lo, hi = srange(int(wa[i]))                        # con 1 bit: {-1, 0}
        A[i] = rng.integers(lo, hi + 1, (L, 2), dtype=I64)
        lo, hi = srange(int(wu[i]))
        U[i] = rng.integers(lo, hi + 1, (L, af, 2), dtype=I64)
    return A, U


def part_corner(prm):
    L, af = prm["l"], prm["af"]
    ea, eu = srange(prm["nb_a"]), srange(prm["nb_u"])
    A = np.empty((16, L, 2), dtype=I64)
    U = np.empty((16, L, af, 2), dtype=I64)
    for c in range(16):
        A[c, :, 0], A[c, :, 1] = ea[(c >> 1) & 1], ea[c & 1]           # ar, ai
        for a in range(af):
            v = (c + 5 * a) % 16
            U[c, :, a, 0], U[c, :, a, 1] = eu[(v >> 3) & 1], eu[(v >> 2) & 1]   # ur, ui
    return A, U


def coarse_limit(prm) -> tuple[int, int]:
    """(k, limite): A grueso = -2^k y el |P| mas grande que se arma exacto."""
    k = min(prm["nb_a"], prm["nb_u"]) - 1
    m = prm["l"] - 1
    return k, m * ((1 << (prm["nb_u"] - 1)) - 1) * (1 << k) - (1 << k)


def build_exact(T: np.ndarray, prm: dict, rng):
    """
    P objetivo T [npos, AF, 2] -> (A, U) con sum_l U'' A = T exacto. Bobina fina:
    A = 1 + 0j, U = f (los k bits bajos de T, en [0, 2^k)). Bobinas gruesas: A =
    -2^k + 0j, U = -c_l con sum c_l = (T - f) / 2^k, repartido parejo. Con A real,
    re de P sale de ur e im de ui. Orden de las bobinas al azar por posicion.
    """
    L, af = prm["l"], prm["af"]
    if L < 2:
        raise ValueError("sat / ties necesitan L >= 2 (una bobina fina y al menos una gruesa)")
    k, lim = coarse_limit(prm)
    if np.abs(T).max() > lim:
        raise AssertionError("build_exact: objetivo fuera del limite")
    npos = T.shape[0]
    m = L - 1
    A = np.zeros((npos, L, 2), dtype=I64)
    U = np.zeros((npos, L, af, 2), dtype=I64)
    for i in range(npos):
        perm = rng.permutation(L)
        A[i, perm[0], 0] = 1
        A[i, perm[1:], 0] = -(1 << k)
        for a in range(af):
            for c in range(2):
                t = int(T[i, a, c])
                f = t % (1 << k)
                s_coarse = -((t - f) >> k)                  # sum de U gruesos
                base, rem = divmod(s_coarse, m)
                us = [base + (1 if j < rem else 0) for j in range(m)]
                U[i, perm[0], a, c] = f
                U[i, perm[1:], a, c] = us
    lo_u, hi_u = srange(prm["nb_u"])
    if U.min() < lo_u or U.max() > hi_u:
        raise AssertionError("build_exact: U fuera de NB_U bits")
    P = (U[..., 0] * A[:, :, None, 0]).sum(axis=1), (U[..., 1] * A[:, :, None, 0]).sum(axis=1)
    if not (np.array_equal(P[0], T[..., 0]) and np.array_equal(P[1], T[..., 1])):
        raise AssertionError("build_exact: P != objetivo")
    return A, U


def feasible_s_max(prm) -> int:
    """s mas grande con el que el borde de saturacion de M (2^(NB_M-1) 2^s, con margen) se arma exacto."""
    _, lim = coarse_limit(prm)
    return min(prm["s_max"], int(math.floor(math.log2(lim))) - prm["nb_m"] - 1)


def part_sat(n, prm, rng):
    af, nb_m = prm["af"], prm["nb_m"]
    lo_m, hi_m = srange(nb_m)
    _, lim = coarse_limit(prm)
    s_hi = feasible_s_max(prm)
    if s_hi < prm["s_min"]:
        raise ValueError("sat: con estos L, NB_A y NB_U no se llega al borde de saturacion")
    s = rng.integers(prm["s_min"], s_hi + 1, n)
    T = np.empty((n, af, 2), dtype=I64)
    targets = (hi_m - 1, hi_m, hi_m + 1, hi_m + 2, "big+", lo_m + 1, lo_m, lo_m - 1, lo_m - 2, "big-")
    for i in range(n):
        si = int(s[i])
        for a in range(af):
            for c in range(2):
                mt = targets[int(rng.integers(len(targets)))]
                if mt in ("big+", "big-"):
                    span = max((lim >> max(si, 0)) - hi_m - 2, 1)
                    mt = (hi_m + 2 + int(rng.integers(span))) * (1 if mt == "big+" else -1)
                if si > 0:
                    h = 1 << (si - 1)
                    r = rng.random()
                    if r < 0.3:
                        delta = int(rng.choice([-h, -h + 1, h - 1, h]))
                    else:
                        delta = int(rng.integers(-h, h + 1))
                    t = mt * (1 << si) + delta
                else:
                    t = (mt >> -si) + int(rng.integers(-1, 2))
                T[i, a, c] = max(-lim, min(lim, t))
    A, U = build_exact(T, prm, rng)
    return A, U, s + prm["g"]


def part_ties(n, prm, rng):
    af, nb_m = prm["af"], prm["nb_m"]
    lo_m, hi_m = srange(nb_m)
    _, lim = coarse_limit(prm)
    s_hi = min(prm["s_max"], int(math.floor(math.log2(lim))) - 1)
    if s_hi < 1:
        raise ValueError("ties: no hay s >= 1 con empates armables")
    s = rng.integers(1, s_hi + 1, n)
    T = np.empty((n, af, 2), dtype=I64)
    j = 0
    for i in range(n):
        si = int(s[i])
        step, h = 1 << si, 1 << (si - 1)
        kmin = max(lo_m, -((lim + h) // step))            # ceil((-lim - h) / step)
        kmax = min(hi_m - 1, (lim - h) // step)
        for a in range(af):
            for c in range(2):
                want = j % 2                               # paridad de floor(P / 2^s)
                j += 1
                k = int(rng.integers(kmin, kmax + 1))
                if (k & 1) != want:
                    k = k + 1 if k + 1 <= kmax else k - 1
                T[i, a, c] = k * step + h
    A, U = build_exact(T, prm, rng)
    return A, U, s + prm["g"]


# ------------------------------------------------------------------- phantom
def phantom_positions(spec: dict, prm: dict):
    """La cadena de mac_output_study.py (seccion D), recortada a las filas pedidas."""
    from fft2d_mask_study import Chain, build_phantom, popcount         # noqa: E402
    from unfold_ref import (bitrev_perm, cordic_gain, gram, ldlh, pack_hw,  # noqa: E402
                            scale_quantize, smaps_to_S, solve_U, unpack_hw)

    ph = PHANTOM
    want = dict(l=ph["L"], af=ph["af"], nb_a=16, nb_u=16)
    if any(prm[k] != v for k, v in want.items()):
        raise ValueError(f"phantom: necesita {want}")
    n_it = OutCfg(D=0).N_IT                    # el default de cordic_mag
    n, L, af = ph["n"], ph["L"], ph["af"]
    n1, n2 = n, n // af
    T = n1 * n2
    tw_dir = HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"
    m, smaps, _, k_hw = build_phantom(HERE / "gen", n, L, af)
    ch = Chain(k_hw, smaps, m, af, tw_dir, "bobina")
    hr, hi = ch.pass1(ph["m1"])
    ar, ai = ch.pass2(hr, hi, ph["m2"])                   # [l, t1, t2]: el stream de la fft2d
    S = popcount(ph["m1"], ch.L1) + popcount(ph["m2"], ch.L2)
    A_r, A_i = ar.reshape(L, T), ai.reshape(L, T)         # [l, o]
    Sr, Si = smaps_to_S(smaps, af)
    Lr, Li, _, dinv = ldlh(*gram(Sr, Si, 0.0))
    Ur, Ui = solve_U(Lr, Li, dinv, Sr, Si)
    qr, qi, E, _ = scale_quantize(Ur, Ui, ch.g, cordic_gain(n_it), 16)
    Ur_s, Ui_s = unpack_hw(pack_hw(qr, qi), n1, n2, af, L)   # [o, l, a]
    D = 49 - S - E - F_OUT
    if not 0 <= D < (1 << prm["nb_d"]):
        raise ValueError(f"phantom: D = {D} fuera de NB_D bits")
    inv1 = np.argsort(bitrev_perm(n1))                    # x -> t1
    rows = [int(x) for x in spec.get("rows", [])]
    if not rows:
        raise ValueError("phantom: falta vec_gen.rows")
    o = np.concatenate([inv1[x] * n2 + np.arange(n2) for x in rows])
    A = np.stack([A_r[:, o].T, A_i[:, o].T], axis=-1).astype(I64)          # [npos, L, 2]
    U = np.stack([Ur_s[o], Ui_s[o]], axis=-1).astype(I64)                  # [npos, L, AF, 2]
    info = dict(n_it=n_it, nb_out=OutCfg(D=0).W_OUT, round_conv_out=int(OutCfg(D=0).ROUND_CONV_OUT),
                f_out=F_OUT, s_fft=S, e_u=E, d_phantom=D, rows=",".join(str(x) for x in rows))
    return A, U, np.full(o.size, D, dtype=I64), info


# -------------------------------------------------------------------- stream
def make_faults(items: list[str], npos: int, L: int, rng) -> dict[int, tuple[str, int]]:
    """{posicion: (tipo, bobina)}, alternando una bobina que no es la ultima y la ultima."""
    want = parse_list(items, ("drop", "dup"), ("drop", "dup"), "falla")
    kinds = []
    for kind, cnt in want:
        kinds += [(kind, j) for j in range(cnt)]
    if not kinds:
        return {}
    if L < 2:
        raise ValueError("faults: con L = 1 cada muestra es una posicion entera (no hay error que ver)")
    cand = np.arange(1, npos - 2)
    for _ in range(1000):
        pos = np.sort(rng.choice(cand, len(kinds), replace=False))
        if np.all(np.diff(pos) >= 3):
            break
    else:
        raise ValueError("faults: no entran separadas por 3 posiciones")
    out = {}
    for (kind, j), p in zip(kinds, pos.tolist()):
        coil = L - 1 if j % 2 == 1 else int(rng.integers(0, L - 1))
        out[int(p)] = (kind, coil)
    return out


def flatten(A, U, D, prm, faults, d_junk, rng):
    """Posiciones del productor -> muestras validas (con las fallas)."""
    L, af = prm["l"], prm["af"]
    rows = []
    for p in range(A.shape[0]):
        for l in range(L):
            last = int(l == L - 1)
            d = int(D[p]) if (last or not d_junk) else int(rng.integers(0, 1 << prm["nb_d"]))
            row = [int(A[p, l, 0]), int(A[p, l, 1])]
            for a in range(af):
                row += [int(U[p, l, a, 0]), int(U[p, l, a, 1])]
            row += [last, d]
            f = faults.get(p)
            if f is not None and f[1] == l:
                if f[0] == "drop":
                    continue
                rows.append(list(row))                    # dup: la misma muestra dos veces
            rows.append(row)
    return np.array(rows, dtype=I64)


def close_groups(stream: np.ndarray, L: int) -> list[tuple[list[int], int]]:
    """Regla de cierre (decision 13): [(muestras del grupo, err)]; un grupo sin cerrar al final no sale."""
    groups, cur, cnt = [], [], 0
    last_col = stream.shape[1] - 2
    for k in range(stream.shape[0]):
        cur.append(k)
        last = bool(stream[k, last_col])
        count = cnt == L - 1
        if last or count:
            groups.append((cur, int(last != count)))
            cur, cnt = [], 0
        else:
            cnt += 1
    return groups


def make_reference(spec: dict, prm: dict) -> dict:
    rng = np.random.default_rng(int(spec.get("seed", 1)))
    parts = parse_list(spec.get("parts", []), PART_KINDS, SIZED, "parte")
    if not parts:
        raise ValueError("vec_gen.parts vacio")
    kinds = {k for k, _ in parts}
    if ("phantom" in kinds) != bool(prm["e2e"]) or ("phantom" in kinds and len(parts) != 1):
        raise ValueError("phantom va solo, y solo con E2E = 1 (y E2E = 1 solo con phantom)")
    spec_d = spec.get("d", 13)
    As, Us, Ds = [], [], []
    info = {}
    for kind, n in parts:
        if kind == "rand":
            A, U = part_rand(n, prm, rng)
            D = d_values(spec_d, n, prm, rng)
        elif kind == "scaled":
            A, U = part_scaled(n, prm, rng)
            D = d_values(spec_d, n, prm, rng)
        elif kind == "corner":
            A, U = part_corner(prm)
            D = d_values(spec_d, A.shape[0], prm, rng)
        elif kind == "dsweep":
            nd = 1 << prm["nb_d"]
            A, U = part_scaled(n * nd, prm, rng)
            D = np.repeat(np.arange(nd, dtype=I64), n)
        elif kind == "sat":
            A, U, D = part_sat(n, prm, rng)
        elif kind == "ties":
            A, U, D = part_ties(n, prm, rng)
        else:
            A, U, D, info = phantom_positions(spec, prm)
        As.append(A)
        Us.append(U)
        Ds.append(np.asarray(D, dtype=I64))
    A, U, D = np.concatenate(As), np.concatenate(Us), np.concatenate(Ds)
    if spec.get("shuffle", "phantom" not in kinds):
        perm = rng.permutation(A.shape[0])
        A, U, D = A[perm], U[perm], D[perm]
    npos = A.shape[0]

    faults = make_faults(spec.get("faults", []), npos, prm["l"], rng)
    stream = flatten(A, U, D, prm, faults, bool(spec.get("d_junk", False)), rng)
    groups = close_groups(stream, prm["l"])

    # P de cada grupo con mac (rellenando con A = 0 hasta L muestras) y M con p_to_m (por s)
    L, af = prm["l"], prm["af"]
    n_out = len(groups)
    Ar = np.zeros((L, n_out), dtype=I64)
    Ai = np.zeros((L, n_out), dtype=I64)
    Ur = np.zeros((n_out, L, af), dtype=I64)
    Ui = np.zeros((n_out, L, af), dtype=I64)
    close = np.empty((n_out, 2), dtype=I64)
    s_out = np.empty(n_out, dtype=I64)
    for g, (ks, err) in enumerate(groups):
        for j, k in enumerate(ks):
            Ar[j, g], Ai[j, g] = stream[k, 0], stream[k, 1]
            Ur[g, j, :] = stream[k, 2:2 + 2 * af:2]
            Ui[g, j, :] = stream[k, 3:3 + 2 * af:2]
        close[g] = (ks[-1], err)
        s_out[g] = stream[ks[-1], -1] - prm["g"]
    Pr, Pi = mac(Ar, Ai, Ur, Ui)                            # [n_out, AF]
    Mr, Mi = np.empty_like(Pr), np.empty_like(Pi)
    conv = bool(prm["round_conv"])
    for s in np.unique(s_out).tolist():
        sel = s_out == s
        Mr[sel], Mi[sel] = p_to_m(Pr[sel], Pi[sel], int(s), prm["nb_m"], conv)

    lo_p, hi_p = srange(prm["nb_p"])
    if min(Pr.min(), Pi.min()) < lo_p or max(Pr.max(), Pi.max()) > hi_p:
        raise AssertionError("P fuera de NB_P bits")

    # contadores (por componente, re e im de cada a)
    P = np.stack([Pr, Pi], axis=-1)                        # [n_out, AF, 2]
    M = np.stack([Mr, Mi], axis=-1)
    lo_m, hi_m = srange(prm["nb_m"])
    pre = np.empty_like(P)                                 # round(P / 2^s) sin saturar
    for s in np.unique(s_out).tolist():
        sel = s_out == s
        pre[sel] = round_shift(P[sel], int(s), conv)
    tie = np.zeros(P.shape, dtype=bool)
    tie_even = np.zeros(P.shape, dtype=bool)
    for s in np.unique(s_out).tolist():
        if s >= 1:
            sel = s_out == s
            low = P[sel] & ((1 << s) - 1)
            t = low == (1 << (s - 1))
            tie[sel] = t
            tie_even[sel] = t & (((P[sel] >> s) & 1) == 0)
    stats = dict(n_pos=npos, n_err=int(close[:, 1].sum()), n_drop=sum(f[0] == "drop" for f in faults.values()),
                 n_dup=sum(f[0] == "dup" for f in faults.values()),
                 s_min_out=int(s_out.min()), s_max_out=int(s_out.max()), n_s_values=int(np.unique(s_out).size),
                 n_shl=int((s_out <= 0).sum()), n_sat_pos=int((pre > hi_m).sum()), n_sat_neg=int((pre < lo_m).sum()),
                 n_m_max=int((pre == hi_m).sum()), n_m_min=int((pre == lo_m).sum()),
                 n_ties=int(tie.sum()), n_ties_even=int(tie_even.sum()),
                 p_bits=int(max(np.abs(P).max(), 1)).bit_length() + 1)

    # que cada parte haya cubierto lo que promete
    if "corner" in kinds:
        top = L << (prm["nb_a"] + prm["nb_u"] - 1)
        if not ((Pr == 0) & (Pi == top)).any():
            raise RuntimeError("corner: no aparece la esquina de H31 (P_re = 0, P_im = +L 2^(NB_A+NB_U-1))")
    if "dsweep" in kinds:
        if not ({prm["s_min"], 0, prm["s_max"]} <= set(np.unique(s_out).tolist())):
            raise RuntimeError("dsweep: faltan s = -G, 0 o el maximo")
    if "sat" in kinds and min(stats["n_sat_pos"], stats["n_sat_neg"], stats["n_m_max"], stats["n_m_min"]) == 0:
        raise RuntimeError(f"sat: falta un lado del borde: {stats}")
    if "ties" in kinds and (stats["n_ties_even"] == 0 or stats["n_ties"] == stats["n_ties_even"]):
        raise RuntimeError(f"ties: faltan empates pares o impares: {stats}")
    if faults and stats["n_err"] == 0:
        raise RuntimeError("faults: ninguna posicion con err")
    if not faults and (stats["n_err"] != 0 or n_out != npos):
        raise AssertionError("sin fallas: tiene que salir una posicion por posicion, sin err")

    ref = dict(stream=stream, P=P, M=M, close=close, stats=stats, info=info)
    if prm["e2e"]:
        cfg = OutCfg(D=int(info["d_phantom"]), G=prm["g"], N_IT=info["n_it"], W_M=prm["nb_m"],
                     W_OUT=info["nb_out"], ROUND_CONV_M=conv, ROUND_CONV_OUT=bool(info["round_conv_out"]))
        out, (Mr2, Mi2), xn = output_stage(Pr, Pi, cfg)
        if not (np.array_equal(Mr2, Mr) and np.array_equal(Mi2, Mi)):
            raise AssertionError("phantom: M de p_to_m != M de output_stage")
        ref["mag"], ref["xn"] = np.asarray(out, dtype=I64), np.asarray(xn, dtype=I64)
        stats.update(mag_max=int(out.max()), n_mag_zero=int((out == 0).sum()))
    return ref


def write_case(out_dir: Path, spec: dict, prm: dict, ref: dict) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("mag_ref.bin", "xn_ref.bin"):                # de una corrida anterior con E2E
        if (out_dir / f).exists() and not prm["e2e"]:
            (out_dir / f).unlink()
    s, st = ref["stats"], ref["stream"]
    lines = [f"{k} {prm[k]}" for k in ("l", "af", "nb_a", "nb_u", "g", "nb_m", "round_conv", "nb_d", "nb_p", "e2e")]
    lines += [f"n {st.shape[0]}", f"n_out {ref['P'].shape[0]}", f"seed {int(spec.get('seed', 1))}",
              f"parts {','.join(str(p) for p in spec['parts'])}",
              f"faults {','.join(spec.get('faults', [])) or '-'}",
              f"d {spec.get('d', 13)}", f"d_junk {int(bool(spec.get('d_junk', False)))}",
              f"shuffle {int(bool(spec.get('shuffle', 'phantom' not in str(spec['parts']))))}"]
    lines += [f"{k} {v}" for k, v in s.items()]
    lines += [f"{k} {v}" for k, v in ref["info"].items()]
    (out_dir / "case.txt").write_text("\n".join(lines) + "\n")
    lim32 = (-(1 << 31), (1 << 31) - 1)
    for name, arr in (("stream", st), ("m", ref["M"])):
        if arr.min() < lim32[0] or arr.max() > lim32[1]:
            raise AssertionError(f"{name}: no entra en int32")
    st.astype("<i4").tofile(out_dir / "stream.bin")
    ref["P"].astype("<i8").tofile(out_dir / "p_ref.bin")
    ref["M"].astype("<i4").tofile(out_dir / "m_ref.bin")
    ref["close"].astype("<i4").tofile(out_dir / "close_ref.bin")
    if prm["e2e"]:
        ref["mag"].astype("<i4").tofile(out_dir / "mag_ref.bin")
        ref["xn"].astype("<i4").tofile(out_dir / "xn_ref.bin")
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
    total = 0
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
        total += size
        n += 1
        s = ref["stats"]
        print(f"  {name:12s} -> {case['VEC_DIR']:26s} L={prm['l']} AF={prm['af']} NB_A={prm['nb_a']} "
              f"NB_U={prm['nb_u']} G={prm['g']} NB_M={prm['nb_m']} RC={prm['round_conv']} NB_D={prm['nb_d']} "
              f"NB_P={prm['nb_p']}  muestras={ref['stream'].shape[0]} salidas={ref['P'].shape[0]} "
              f"err={s['n_err']} s=[{s['s_min_out']}, {s['s_max_out']}] sat=+{s['n_sat_pos']}/-{s['n_sat_neg']} "
              f"empates={s['n_ties']} (pares {s['n_ties_even']}) P={s['p_bits']} b "
              f"{size / 1024:.0f} KiB ({time.time() - t0:.1f} s)", flush=True)
    if only and n != len(only):
        print(f"[ERROR] casos pedidos {sorted(only)}, generados {n}", file=sys.stderr)
        return 1
    print(f"{n} casos, {total / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

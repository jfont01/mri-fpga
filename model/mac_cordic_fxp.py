#!/usr/bin/env python3
"""
mac_cordic_fxp.py -- modelo bit-exacto (enteros) de la parte del PL que sigue a
la fft2d: MAC de desplegado, redondeo de salida y magnitud por CORDIC (puntos 3
y 5 del MAC). Es la referencia contra la que se va a validar el modelo C++ y,
despues, el RTL.

Entradas, en el ORDEN DEL STREAM de la fft2d (o = t1*N2 + t2):
    A[l, o]        salida de la fft2d, enteros Q1.15 (re, im)
    U[o, l, a]     U''_q del buffer del PS (unfold_ref.unpack_hw), Q1.15

1) MAC, exacto en enteros (no depende del orden de las bobinas):
       P[o, a] = sum_l U[o, l, a] * A[l, o]          |P| < 2^(NB_U + 18)
2) Redondeo a la entrada del CORDIC (corrimiento D - G de sesion, half-up como
   cast.v, saturacion a W_M bits con signo):
       M[o, a] = sat(floor((P + 2^(D-G-1)) / 2^(D-G)))
   G son bits de guarda: M tiene G bits fraccionales mas que la salida.
3) CORDIC en modo vectoring sobre (|Re M|, |Im M|), primer cuadrante, n
   iteraciones, corrimientos aritmeticos (floor):
       si y >= 0:  x += y >> i,  y -= x_viejo >> i
       si no:      x -= y >> i,  y += x_viejo >> i
   x_n = K_n * |M| (K_n = prod sqrt(1 + 2^-2i)). La ganancia se pliega en U''
   (el PS multiplica U' por 1/K_n), asi que x_n ya es |m_hat| en la escala de
   la salida.
4) Salida: out = sat_u(floor((x_n + 2^(G-1)) / 2^G)), sin signo de W_OUT bits.

Relacion con la imagen fisica (bitacora, seccion 4.9):
    m_hat = 2^(S-19+E) * P * 2^-30   ->   out ~ |m_hat| * 2^F_OUT
    con D = 30 - (S - 19 + E) - F_OUT = 49 - S - E - F_OUT
y el pixel de (o, a) es x = br(t1), y = br(t2) + a * N2.

Uso:
    python3 model/mac_cordic_fxp.py      # self-test
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

I64 = np.int64


@dataclass(frozen=True)
class OutCfg:
    D: int            # corrimiento total P -> salida (de sesion)
    G: int = 3        # bits de guarda del CORDIC
    N_IT: int = 12    # iteraciones del CORDIC
    W_M: int = 20     # ancho con signo de M (entrada del CORDIC)
    W_OUT: int = 16   # ancho sin signo de la salida


def cordic_gain(n_it: int) -> float:
    k = 1.0
    for i in range(n_it):
        k *= math.sqrt(1.0 + 2.0 ** (-2 * i))
    return k


def round_shift(x, s: int):
    """floor((x + 2^(s-1)) / 2^s) en enteros (half-up, como cast.v). s <= 0: corrimiento a la izquierda."""
    x = np.asarray(x, dtype=I64)
    if s <= 0:
        return x << (-s)
    return (x + (I64(1) << (s - 1))) >> s


def sat(x, nb: int, signed: bool = True):
    lo, hi = (-(1 << (nb - 1)), (1 << (nb - 1)) - 1) if signed else (0, (1 << nb) - 1)
    return np.clip(x, lo, hi)


def mac(Ar, Ai, Ur, Ui):
    """
    A[l, o] (re, im) y U[o, l, a] (re, im) enteros -> P[o, a] (re, im) exactos.
    Producto complejo con 4 multiplicaciones: re = ur*ar - ui*ai, im = ur*ai + ui*ar.
    """
    Ar = np.asarray(Ar, dtype=I64).T[:, :, None]          # [o, l, 1]
    Ai = np.asarray(Ai, dtype=I64).T[:, :, None]
    Ur = np.asarray(Ur, dtype=I64)
    Ui = np.asarray(Ui, dtype=I64)
    Pr = np.zeros((Ur.shape[0], Ur.shape[2]), dtype=I64)
    Pi = np.zeros_like(Pr)
    for l in range(Ur.shape[1]):                          # orden irrelevante (exacto)
        Pr += Ur[:, l, :] * Ar[:, l, :] - Ui[:, l, :] * Ai[:, l, :]
        Pi += Ur[:, l, :] * Ai[:, l, :] + Ui[:, l, :] * Ar[:, l, :]
    return Pr, Pi


def cordic_mag(mr, mi, n_it: int):
    """Magnitud por CORDIC vectoring, enteros. Devuelve x_n = K_n |m| (aprox)."""
    x = np.abs(np.asarray(mr, dtype=I64))
    y = np.abs(np.asarray(mi, dtype=I64))
    for i in range(n_it):
        xs, ys = x >> i, y >> i
        pos = y >= 0
        x, y = np.where(pos, x + ys, x - ys), np.where(pos, y - xs, y + xs)
    return x


def output_stage(Pr, Pi, cfg: OutCfg):
    """P -> M (redondeo + saturacion) -> CORDIC -> salida sin signo."""
    s = cfg.D - cfg.G
    Mr = sat(round_shift(Pr, s), cfg.W_M)
    Mi = sat(round_shift(Pi, s), cfg.W_M)
    xn = cordic_mag(Mr, Mi, cfg.N_IT)
    out = sat(round_shift(xn, cfg.G), cfg.W_OUT, signed=False)
    return out, (Mr, Mi), xn


def _selftest() -> int:
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FALLA'}] {name} {extra}")

    print("mac_cordic_fxp: self-test")
    rng = np.random.default_rng(3)
    # 1) MAC exacto contra producto complejo en enteros de Python (sin limites)
    L, T, af = 8, 64, 2
    Ar = rng.integers(-2**15, 2**15, (L, T)); Ai = rng.integers(-2**15, 2**15, (L, T))
    Ur = rng.integers(-2**15, 2**15, (T, L, af)); Ui = rng.integers(-2**15, 2**15, (T, L, af))
    Pr, Pi = mac(Ar, Ai, Ur, Ui)
    o, a = 17, 1
    ref = sum(complex(int(Ur[o, l, a]), int(Ui[o, l, a])) * complex(int(Ar[l, o]), int(Ai[l, o])) for l in range(L))
    check("MAC exacto", Pr[o, a] == int(ref.real) and Pi[o, a] == int(ref.imag))
    # cota: |P| < 2^(16 + 18)  (NB_U = 16) -> 36 bits con signo
    worst = L * 2 * (2**15) * (2**15)
    check("cota NB_U + 20 = 36 bits", worst <= 2**34 and np.abs(Pr).max() < 2**35)
    # peor caso real: A = U = -2^15 en re e im
    Pw, _ = mac(np.full((L, 1), -2**15), np.full((L, 1), -2**15),
                np.full((1, L, 1), -2**15), np.full((1, L, 1), 2**15 - 1))
    check("peor caso dentro de 36 bits", -2**35 <= Pw.min() and Pw.max() < 2**35, f"({int(Pw.max()).bit_length() + 1} bits)")

    # 2) CORDIC: error contra la magnitud exacta, para varias iteraciones
    mr = rng.integers(-2**19, 2**19, 200000); mi = rng.integers(-2**19, 2**19, 200000)
    exact = np.hypot(mr.astype(float), mi.astype(float))
    for n_it in (8, 12, 16):
        k = cordic_gain(n_it)
        xn = cordic_mag(mr, mi, n_it)
        err = xn / k - exact                             # en LSB de la entrada
        bound = exact * (1 - math.cos(math.atan(2.0 ** (-(n_it - 1))))) + 2 * n_it
        check(f"CORDIC n={n_it}: |error| <= cota (residuo angular + truncados)",
              np.all(np.abs(err) <= bound + 1), f"(max {np.abs(err).max():.2f} LSB, cota {bound.max():.1f})")
    # simetria: magnitud no depende de los signos
    check("CORDIC: |(-a, b)| = |(a, -b)| = |(a, b)|",
          np.array_equal(cordic_mag(-mr, mi, 12), cordic_mag(mr, -mi, 12)))
    print("== OK" if ok else "== FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())

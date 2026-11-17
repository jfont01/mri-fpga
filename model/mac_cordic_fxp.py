#!/usr/bin/env python3
"""
mac_cordic_fxp.py -- modelo bit-exacto (enteros) de la parte del PL que sigue a
la fft2d: MAC de desplegado, redondeo de salida y magnitud por CORDIC (puntos 3
y 5 del MAC). Es la referencia contra la que se valida el modelo C++ (paso D2:
modulo unfold = pasos 1 y 2, modulo cordic_mag = pasos 3 y 4) y, despues, el RTL.

Entradas, en el ORDEN DEL STREAM de la fft2d (o = t1*N2 + t2):
    A[l, o]        salida de la fft2d, enteros Q1.15 (re, im)
    U[o, l, a]     U''_q del buffer del PS (unfold_ref.unpack_hw), Q1.15

1) [unfold] MAC, exacto en enteros (no depende del orden de las bobinas):
       P[o, a] = sum_l U[o, l, a] * A[l, o]
   Con NB_A = NB_U = 16 y L = 8 (bitacora H31, seccion 4.13):
       -(2^34 - 2^18) <= P_re <= 2^34 - 2^18
       -(2^34 - 2^19) <= P_im <= +2^34
   P_im llega a +2^34 porque ur*ai + ui*ar = +2^31 con las cuatro en -2^15, asi
   que P necesita NB_P = NB_A + NB_U + 1 + ceil(log2 L) = 36 bits con signo.
2) [unfold] Redondeo a la entrada del CORDIC: corrimiento s = D - G (de sesion)
   y saturacion a W_M bits con signo:
       M[o, a] = sat_{W_M}(round(P / 2^s))
   round, con s > 0: half-up, floor((P + 2^(s-1)) / 2^s), como cast.v; con
   ROUND_CONV_M, los empates van al par (cast.v con ROUND_CONV = 1). Con s <= 0,
   corrimiento a la izquierda exacto (despues, la saturacion).
   G son bits de guarda: M tiene G bits fraccionales mas que la salida.
   W_M = W_OUT + 1 + G (= 17 + G = 22 por default) es suficiente y es el minimo:
   la salida satura con |M| >= 0.607 * 2^(W_OUT + G) (0.607 = 1/K_n), asi que
   saturar cada componente de M en 2^(W_OUT + G) solo recorta valores cuya
   salida igual satura; con un bit menos se recortarian valores que no saturan.
3) [cordic_mag] CORDIC en modo vectoring sobre (|Re M|, |Im M|), primer
   cuadrante, n = N_IT iteraciones, corrimientos aritmeticos (floor), los dos
   lados con los valores de antes:
       si y >= 0:  x += y >> i,  y -= x >> i
       si no:      x -= y >> i,  y += x >> i
   x_n ~ K_n * |M|, con K_n = prod_{i < n} sqrt(1 + 2^-2i).
4) [cordic_mag] Salida: out = sat_u(round(x_n / 2^G)), sin signo de W_OUT bits
   (half-up; con ROUND_CONV_OUT, empates al par).

Escala (bitacora 4.9 y 4.11). El PS pliega la ganancia del CORDIC en U''
(U'' = U' * 2^-E / K_n, con U' = U / g_l), asi que lo que da el MAC es m_hat / K_n,
no m_hat:
    m_hat / K_n = 2^(S-19+E) * P * 2^-30
El CORDIC vuelve a multiplicar por K_n, y la salida queda
    out ~ K_n * |P| * 2^-D = |m_hat| * 2^F_OUT,   con D = 49 - S - E - F_OUT.
El pixel de (o, a) es x = br(t1), y = br(t2) + a * N2.

Uso:
    python3 model/mac_cordic_fxp.py      # self-test
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

I64 = np.int64


@dataclass(frozen=True)
class OutCfg:
    D: int                        # corrimiento total P -> salida (de sesion)
    G: int = 5                    # bits de guarda del CORDIC
    N_IT: int = 12                # iteraciones del CORDIC
    W_M: int | None = None        # ancho con signo de M; None -> W_OUT + 1 + G (= 22)
    W_OUT: int = 16               # ancho sin signo de la salida
    ROUND_CONV_M: bool = False    # P -> M con empates al par (unfold, ROUND_CONV)
    ROUND_CONV_OUT: bool = False  # x_n -> salida con empates al par (cordic_mag, ROUND_CONV)

    def __post_init__(self):
        if self.W_M is None:
            object.__setattr__(self, "W_M", self.W_OUT + 1 + self.G)
        if self.G < 0 or self.N_IT < 1 or self.W_OUT < 1 or self.W_M < 2:
            raise ValueError(f"OutCfg invalida: {self}")


def cordic_gain(n_it: int) -> float:
    k = 1.0
    for i in range(n_it):
        k *= math.sqrt(1.0 + 2.0 ** (-2 * i))
    return k


def round_shift(x, s: int, conv: bool = False):
    """
    x / 2^s redondeado, en enteros.
    s > 0: half-up, floor((x + 2^(s-1)) / 2^s), como cast.v. Con conv, los empates
           van al par, como cast.v con ROUND_CONV = 1: m + (r > h) + (r == h) lsb(m),
           con m = floor(x / 2^s), r = x - m 2^s y h = 2^(s-1).
    s <= 0: corrimiento a la izquierda, exacto.
    En int64: si el resultado no entra (numpy envolveria sin avisar), OverflowError.
    """
    x = np.asarray(x, dtype=I64)
    if s <= 0:
        if s < 0 and x.size:
            lim = 1 << (63 + s)                          # x << -s entra en int64
            if int(x.min()) < -lim or int(x.max()) >= lim:
                raise OverflowError(f"round_shift: x << {-s} no entra en int64")
        return x << (-s)
    if s > 62:
        raise ValueError(f"round_shift: s = {s} > 62")
    h = I64(1) << (s - 1)
    if conv:
        m = x >> s                                       # >> de numpy en int64 = floor
        r = x - (m << s)
        return m + ((r > h) | ((r == h) & ((m & 1) == 1)))
    if x.size and int(x.max()) > np.iinfo(I64).max - int(h):
        raise OverflowError("round_shift: x + 2^(s-1) no entra en int64")
    return (x + h) >> s


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


def p_to_m(Pr, Pi, s: int, w_m: int, conv: bool = False):
    """Paso 2 (modulo unfold): M = sat_{w_m}(round(P / 2^s)), re e im por separado."""
    return sat(round_shift(Pr, s, conv), w_m), sat(round_shift(Pi, s, conv), w_m)


def m_to_out(Mr, Mi, n_it: int, g: int, w_out: int, conv: bool = False):
    """Pasos 3 y 4 (modulo cordic_mag): x_n = CORDIC(|Re M|, |Im M|), out = sat_u(round(x_n / 2^g))."""
    xn = cordic_mag(Mr, Mi, n_it)
    return sat(round_shift(xn, g, conv), w_out, signed=False), xn


def output_stage(Pr, Pi, cfg: OutCfg):
    """P -> M (redondeo + saturacion) -> CORDIC -> salida sin signo."""
    Mr, Mi = p_to_m(Pr, Pi, cfg.D - cfg.G, cfg.W_M, cfg.ROUND_CONV_M)
    out, xn = m_to_out(Mr, Mi, cfg.N_IT, cfg.G, cfg.W_OUT, cfg.ROUND_CONV_OUT)
    return out, (Mr, Mi), xn


def _selftest() -> int:
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FALLA'}] {name} {extra}")

    print("mac_cordic_fxp: self-test")
    rng = np.random.default_rng(3)
    cfg0 = OutCfg(D=0)
    check("defaults: G = 5, N_IT = 12, W_M = 17 + G = 22, W_OUT = 16, half-up",
          (cfg0.G, cfg0.N_IT, cfg0.W_M, cfg0.W_OUT, cfg0.ROUND_CONV_M, cfg0.ROUND_CONV_OUT)
          == (5, 12, 22, 16, False, False))
    check("W_M derivado de G y W_OUT; explicito se respeta",
          OutCfg(D=0, G=3).W_M == 20 and OutCfg(D=0, G=5, W_M=30).W_M == 30)

    # 1) MAC exacto contra producto complejo en enteros de Python (sin limites)
    L, T, af = 8, 64, 2
    NB_A = NB_U = 16
    Ar = rng.integers(-2**15, 2**15, (L, T)); Ai = rng.integers(-2**15, 2**15, (L, T))
    Ur = rng.integers(-2**15, 2**15, (T, L, af)); Ui = rng.integers(-2**15, 2**15, (T, L, af))
    Pr, Pi = mac(Ar, Ai, Ur, Ui)
    o, a = 17, 1
    ref = sum(complex(int(Ur[o, l, a]), int(Ui[o, l, a])) * complex(int(Ar[l, o]), int(Ai[l, o])) for l in range(L))
    check("MAC exacto", Pr[o, a] == int(ref.real) and Pi[o, a] == int(ref.imag))
    # Rango de un producto complejo: es multilineal en (ur, ui, ar, ai), asi que los extremos
    # estan en los vertices de la caja [-2^15, 2^15 - 1]^4: alcanza con las 16 combinaciones.
    ext = (-2**15, 2**15 - 1)
    re_v = [ur * ar - ui * ai for ur in ext for ui in ext for ar in ext for ai in ext]
    im_v = [ur * ai + ui * ar for ur in ext for ui in ext for ar in ext for ai in ext]
    check("producto: re en [-(2^31 - 2^15), 2^31 - 2^15], im en [-(2^31 - 2^16), +2^31]",
          (min(re_v), max(re_v), min(im_v), max(im_v)) == (-(2**31 - 2**15), 2**31 - 2**15, -(2**31 - 2**16), 2**31))
    # Esquina de H31: A = U'' = -2^15 - j 2^15 en las 8 bobinas -> P_im = 8 * 2^31 = 2^34
    c = np.full((L, 1), -2**15)
    Pr_w, Pi_w = mac(c, c, c.T[:, :, None], c.T[:, :, None])
    nb_p = NB_A + NB_U + 1 + math.ceil(math.log2(L))
    bits = int(Pi_w[0, 0]).bit_length() + 1
    check("esquina H31: P_re = 0, P_im = +2^34 (36 bits con signo, no entra en 35)",
          Pr_w[0, 0] == 0 and Pi_w[0, 0] == 2**34 and bits == 36 == nb_p,
          f"(NB_P = NB_A + NB_U + 1 + ceil(log2 L) = {nb_p})")

    # 2) Redondeo: half-up y convergente contra Fraction (exacto; round() de Fraction va al par)
    def ref_round(v, s, conv):
        q = Fraction(int(v), 2**s)
        return round(q) if conv else math.floor(q + Fraction(1, 2))
    good = True
    small = np.arange(-4096, 4097)
    big = np.concatenate([rng.integers(-2**35, 2**35, 3000), [2**34, -(2**34 - 2**19), 2**35 - 1, -2**35]])
    for conv in (False, True):
        for s in range(1, 7):
            good &= all(ref_round(v, s, conv) == r for v, r in zip(small, round_shift(small, s, conv)))
        for s in (1, 8, 13, 20, 35, 58):
            vals = np.concatenate([big, (big >> s << s) + (1 << (s - 1))])     # con empates
            good &= all(ref_round(v, s, conv) == r for v, r in zip(vals, round_shift(vals, s, conv)))
    check("round_shift (half-up y convergente) == referencia exacta, s = 1 ... 58", good)
    ties = np.array([3, 5, -3, -5])                       # +-1.5 y +-2.5 con s = 1
    check("empates: half-up 1.5, 2.5, -1.5, -2.5 -> 2, 3, -1, -2; convergente -> 2, 2, -2, -2",
          list(round_shift(ties, 1)) == [2, 3, -1, -2] and list(round_shift(ties, 1, True)) == [2, 2, -2, -2])
    check("s <= 0: corrimiento a la izquierda exacto (s = -5 con la esquina de 36 bits)",
          round_shift(np.array([2**34, -2**35]), -5)[0] == 2**39 and round_shift(np.array([7]), 0)[0] == 7)
    try:
        round_shift(np.array([2**40]), -30)
        check("s <= 0: desborde de int64 detectado", False)
    except OverflowError:
        check("s <= 0: desborde de int64 detectado", True)

    # 3) CORDIC: error contra la magnitud exacta, para varias iteraciones (rango completo de M)
    hm = 2 ** (cfg0.W_M - 1)
    mr = rng.integers(-hm, hm, 200000); mi = rng.integers(-hm, hm, 200000)
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

    # 4) Saturacion de M: con W_M = 17 + G solo recorta lo que igual satura la salida; con uno menos, no
    D, n_sat = 13, 400000
    r = 2.0 * hm * np.sqrt(rng.random(n_sat))                              # |M| hasta 2 x el rango
    ph = rng.random(n_sat) * 2 * np.pi
    sc = 2.0 ** (D - cfg0.G)                                               # P = M * 2^(D - G)
    Pr_s = np.rint(r * np.cos(ph) * sc).astype(I64); Pi_s = np.rint(r * np.sin(ph) * sc).astype(I64)
    out_22, (Mr22, _), _ = output_stage(Pr_s, Pi_s, OutCfg(D=D))
    out_wide, _, _ = output_stage(Pr_s, Pi_s, OutCfg(D=D, W_M=40))
    out_21, _, _ = output_stage(Pr_s, Pi_s, OutCfg(D=D, W_M=21))
    clipped = int((np.abs(Mr22) >= hm - 1).sum())
    check("W_M = 22 recorta M pero la salida es identica a sin recortar (W_M = 40)",
          clipped > 0 and np.array_equal(out_22, out_wide), f"({clipped} recortes de Re M)")
    check("W_M = 21 cambia la salida (22 es el minimo)", not np.array_equal(out_21, out_wide),
          f"({int((out_21 != out_wide).sum())} salidas distintas)")
    check("saturacion: M en [-2^21, 2^21 - 1], salida en [0, 65535] y llega a 65535",
          Mr22.min() == -hm and Mr22.max() == hm - 1 and out_22.min() >= 0 and out_22.max() == 2**16 - 1)

    # 5) Escala (docstring): con 1/K_n plegado en U'', 2^(S-19+E) P 2^-30 = m_hat / K_n y out ~ |m_hat| 2^F_OUT
    S_, E_, F_OUT, n_it = 6, 12, 15, 12
    Kn = cordic_gain(n_it)
    T5 = 200000
    A_int = rng.integers(-2**15, 2**15, (2, L, T5))                        # A Q1.15 (re, im)
    Up = (rng.uniform(-1, 1, (T5, L)) + 1j * rng.uniform(-1, 1, (T5, L))) * 2.0 ** 11   # U' = U / g_l
    q = Up / Kn * 2.0 ** (-E_) * 2.0 ** 15                                 # U'' en LSB de Q1.15
    assert max(np.abs(q.real).max(), np.abs(q.imag).max()) < 2**15
    qr, qi = np.rint(q.real).astype(I64), np.rint(q.imag).astype(I64)
    Pr5, Pi5 = mac(A_int[0], A_int[1], qr[:, :, None], qi[:, :, None])
    A_c = (A_int[0] + 1j * A_int[1]).T * 2.0 ** -15                        # [o, l]
    m_hat = 2.0 ** (S_ - 19) * np.sum(Up * A_c, axis=1)
    lhs = 2.0 ** (S_ - 19 + E_) * (Pr5[:, 0] + 1j * Pi5[:, 0]) * 2.0 ** -30
    rel_k = np.abs(lhs - m_hat / Kn).max() / np.abs(m_hat / Kn).max()
    rel_1 = np.abs(lhs - m_hat).max() / np.abs(m_hat).max()
    D5 = 49 - S_ - E_ - F_OUT
    out5, _, _ = output_stage(Pr5[:, 0], Pi5[:, 0], OutCfg(D=D5, N_IT=n_it))
    keep = out5 < 2**16 - 1
    err5 = out5[keep] - np.abs(m_hat[keep]) * 2.0 ** F_OUT                   # incluye cuantizar U''
    err_hw = out5[keep] - Kn * np.hypot(Pr5[keep, 0], Pi5[keep, 0]) * 2.0 ** -D5   # solo la cadena
    check("escala: 2^(S-19+E) P 2^-30 es m_hat / K_n, no m_hat",
          rel_k < 1e-3 and abs(rel_1 - (1 - 1 / Kn)) < 1e-2,
          f"(contra m_hat/K_n {rel_k:.1e}; contra m_hat {rel_1:.3f} = 1 - 1/K_n)")
    check("escala: out ~ |m_hat| 2^F_OUT con D = 49 - S - E - F_OUT (media)",
          abs(err5.mean()) < 0.1 and keep.sum() > 0.99 * T5,
          f"(D = {D5}; media {err5.mean():+.3f} LSB, rms {np.sqrt(np.mean(err5 ** 2)):.3f}, salida hasta {out5.max()})")
    check("cadena P -> salida contra K_n |P| 2^-D: |error| < 1 LSB",
          np.abs(err_hw).max() < 1,
          f"(media {err_hw.mean():+.3f}, rms {np.sqrt(np.mean(err_hw ** 2)):.3f}, max {np.abs(err_hw).max():.2f} LSB)")

    # 6) ROUND_CONV_M / ROUND_CONV_OUT: cada uno va a su paso, y el convergente baja la media (bitacora 4.13)
    Pk, Qk = Pr5[keep, 0], Pi5[keep, 0]
    ideal = Kn * np.hypot(Pk, Qk) * 2.0 ** -D5
    means = {}
    for cm in (False, True):
        for co in (False, True):
            o6, _, _ = output_stage(Pk, Qk, OutCfg(D=D5, N_IT=n_it, ROUND_CONV_M=cm, ROUND_CONV_OUT=co))
            o6m, _ = m_to_out(*p_to_m(Pk, Qk, D5 - 5, 22, cm), n_it, 5, 16, co)
            ok_route = np.array_equal(o6, o6m)
            means[cm, co] = (o6 - ideal).mean()
            check(f"output_stage(ROUND_CONV_M = {int(cm)}, ROUND_CONV_OUT = {int(co)}) == p_to_m + m_to_out",
                  ok_route, f"(media {means[cm, co]:+.4f} LSB)")
    dm = means[False, False] - means[True, True]
    check("convergente/convergente baja la media ~0.016 LSB respecto de half-up/half-up",
          0.005 < dm < 0.03, f"({dm:.4f} LSB)")
    print("== OK" if ok else "== FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())

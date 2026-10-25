#!/usr/bin/env python3
"""
unfold_ref.py -- referencia del calculo de U en el PS (MAC de desplegado,
punto 4) y del buffer que el PS le entrega al PL.

Por posicion plegada (x, p), con S[l, a] = s_l[x, p + a*N/Af]  (L x Af):

    A = S^H S + lam*I          Af x Af, hermitiana definida positiva
    A = L D L^H                L triangular inferior con unos en la diagonal,
                               D real y positiva: sin raices cuadradas
    U = A^-1 S^H               Af x L: por cada bobina l, A u = S^H[:, l] con
                               L z = b,  w = D^-1 z,  L^H u = w

Con lam = 0 es la matriz de desplegado de Pruessmann con Psi = I (datos
preblanqueados). lam > 0 es Tikhonov (Paul & Mathew, cap. 2): hace falta cuando
los mapas se anulan (fuera del objeto, ESPIRiT) y S^H S deja de ser invertible.

Despues, para el PL:

    U'[a, l]  = U[a, l] * c_l,  c_l = (1/g_l) * (1/K)       g_l: ganancia del
                                                            k-space de la bobina;
                                                            K: ganancia del CORDIC
                                                            (si se pliega aca)
    E         = exponente de frexp del maximo |Re|, |Im| de U' (comun a la sesion)
    U''_q     = sat16(rint(U' * 2^(15 - E)))                Q1.15, redondeo al par
    palabra   = (im << 16) | (re & 0xFFFF)                  como los puertos de la FFT

y el buffer va en el ORDEN DEL STREAM de la fft2d: o = t1*N2 + t2 es la
posicion (x = br(t1), p = br(t2)).

Regla de este archivo: todas las cuentas van con re/im separados, en el mismo
orden en que las va a hacer el C del PS (sumas secuenciales sobre l, 1/d y 1/g
calculados una vez, escalas por potencias de 2 con ldexp, rint al par). Asi el
C nativo se puede exigir bit a bit contra esta referencia.

Uso:
    python3 model/unfold_ref.py          # self-test (fantoma 256 x 256 y casos chicos)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ----------------------------------------------------------- utilidades
def bitrev_perm(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])


def cordic_gain(n_it: int) -> float:
    """K_n = prod_{i<n} sqrt(1 + 2^-2i): ganancia del CORDIC con n iteraciones."""
    k = 1.0
    for i in range(n_it):
        k *= math.sqrt(1.0 + 2.0 ** (-2 * i))
    return k


def smaps_to_S(smaps: np.ndarray, af: int):
    """smaps[l, x, y] (convencion de los scripts) -> Sr, Si [x, p, l, a]."""
    L, n, _ = smaps.shape
    S = smaps.reshape(L, n, af, n // af).transpose(1, 3, 0, 2)
    return np.ascontiguousarray(S.real), np.ascontiguousarray(S.imag)


# ------------------------------------------------------------ S^H S + lam I
def gram(Sr, Si, lam: float = 0.0):
    """
    A[i][j] = sum_l conj(S[l,i]) S[l,j] (+ lam en la diagonal), solo i >= j.
    Devuelve Ar, Ai [x, p, Af, Af] (la parte superior queda en cero).
    Suma secuencial en l (como el bucle del C), no np.sum (que es por pares).
    """
    *pos, L, af = Sr.shape
    Ar = np.zeros((*pos, af, af))
    Ai = np.zeros((*pos, af, af))
    for i in range(af):
        for j in range(i + 1):
            accr = np.zeros(pos)
            acci = np.zeros(pos)
            for l in range(L):
                sri, sii = Sr[..., l, i], Si[..., l, i]
                srj, sij = Sr[..., l, j], Si[..., l, j]
                # conj(s_i) * s_j = (sri - j sii)(srj + j sij)
                accr = accr + (sri * srj + sii * sij)
                acci = acci + (sri * sij - sii * srj)
            if i == j:
                accr = accr + lam
                acci = np.zeros(pos)            # exactamente real
            Ar[..., i, j], Ai[..., i, j] = accr, acci
    return Ar, Ai


# ----------------------------------------------------------------- L D L^H
def ldlh(Ar, Ai):
    """
    A = L D L^H por columnas (j = 0 .. Af-1):
        d_j  = A_jj - sum_{k<j} |L_jk|^2 d_k
        L_ij = (A_ij - sum_{k<j} L_ik d_k conj(L_jk)) / d_j      (i > j)
    Devuelve Lr, Li [.., Af, Af], d y dinv [.., Af]. dinv = 1/d (una division por
    pivote). Si algun d_j <= 0 (A no definida positiva) queda en dinv un inf/nan:
    lo reporta ldlh_ok().
    """
    *pos, af, _ = Ar.shape
    Lr = np.zeros_like(Ar)
    Li = np.zeros_like(Ai)
    d = np.zeros((*pos, af))
    dinv = np.zeros((*pos, af))
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(af):
            acc = Ar[..., j, j]
            for k in range(j):
                lr, li = Lr[..., j, k], Li[..., j, k]
                acc = acc - (lr * lr + li * li) * d[..., k]
            d[..., j] = acc
            dinv[..., j] = 1.0 / acc
            Lr[..., j, j] = 1.0
            for i in range(j + 1, af):
                accr, acci = Ar[..., i, j], Ai[..., i, j]
                for k in range(j):
                    # L_ik * d_k * conj(L_jk)
                    ar, ai = Lr[..., i, k] * d[..., k], Li[..., i, k] * d[..., k]
                    br, bi = Lr[..., j, k], -Li[..., j, k]
                    accr = accr - (ar * br - ai * bi)
                    acci = acci - (ar * bi + ai * br)
                Lr[..., i, j] = accr * dinv[..., j]
                Li[..., i, j] = acci * dinv[..., j]
    return Lr, Li, d, dinv


def ldlh_ok(d) -> np.ndarray:
    """Posiciones donde la factorizacion es valida (todos los pivotes > 0 y finitos)."""
    return np.all(np.isfinite(d) & (d > 0), axis=-1)


def solve_U(Lr, Li, dinv, Sr, Si):
    """
    U[a, l] para todas las bobinas: b = S^H[:, l] = conj(S[l, :]),
        L z = b (adelante), w = dinv * z, L^H u = w (atras).
    Devuelve Ur, Ui [x, p, Af, L].
    """
    *pos, L, af = Sr.shape
    Ur = np.zeros((*pos, af, L))
    Ui = np.zeros((*pos, af, L))
    for l in range(L):
        zr = [None] * af
        zi = [None] * af
        for i in range(af):                      # L z = b
            accr, acci = Sr[..., l, i], -Si[..., l, i]
            for k in range(i):
                lr, li = Lr[..., i, k], Li[..., i, k]
                accr = accr - (lr * zr[k] - li * zi[k])
                acci = acci - (lr * zi[k] + li * zr[k])
            zr[i], zi[i] = accr, acci
        wr = [zr[i] * dinv[..., i] for i in range(af)]
        wi = [zi[i] * dinv[..., i] for i in range(af)]
        ur = [None] * af
        ui = [None] * af
        for i in range(af - 1, -1, -1):          # L^H u = w,  (L^H)_ik = conj(L_ki)
            accr, acci = wr[i], wi[i]
            for k in range(i + 1, af):
                lr, li = Lr[..., k, i], -Li[..., k, i]
                accr = accr - (lr * ur[k] - li * ui[k])
                acci = acci - (lr * ui[k] + li * ur[k])
            ur[i], ui[i] = accr, acci
        for i in range(af):
            Ur[..., i, l], Ui[..., i, l] = ur[i], ui[i]
    return Ur, Ui


def unfolding_ldlh(smaps: np.ndarray, af: int, lam: float = 0.0):
    """U[x, p, a, l] (complejo) y la mascara de posiciones validas."""
    Sr, Si = smaps_to_S(smaps, af)
    Ar, Ai = gram(Sr, Si, lam)
    Lr, Li, d, dinv = ldlh(Ar, Ai)
    Ur, Ui = solve_U(Lr, Li, dinv, Sr, Si)
    return Ur + 1j * Ui, ldlh_ok(d)


# ------------------------------------------------- escala, cuantizacion, buffer
def scale_quantize(Ur, Ui, g, k_cordic: float | None = None, nb: int = 16):
    """
    U' = U * c_l con c_l = (1/g_l) * (1/K) (o solo 1/g_l), E por frexp del
    maximo de |Re|, |Im|, y U''_q = sat(rint(U' * 2^(nb-1-E))).
    Devuelve (Uq_re, Uq_im) enteros, E y c_l.
    """
    inv_k = 1.0 / k_cordic if k_cordic is not None else 1.0
    c = np.array([(1.0 / gl) * inv_k for gl in g])      # uno por bobina
    Upr = Ur * c                                          # broadcast sobre l (ultimo eje)
    Upi = Ui * c
    maxc = max(np.abs(Upr).max(), np.abs(Upi).max())
    _, E = math.frexp(maxc)                               # maxc = m * 2^E, m en [0.5, 1)
    f = nb - 1
    lo, hi = -(1 << f), (1 << f) - 1
    qr = np.clip(np.rint(np.ldexp(Upr, f - E)), lo, hi).astype(np.int64)
    qi = np.clip(np.rint(np.ldexp(Upi, f - E)), lo, hi).astype(np.int64)
    return qr, qi, E, c


def pack_hw(qr, qi, layout: str = "pos_coil_a") -> np.ndarray:
    """
    U''_q[x, p, a, l] -> palabras de 32 bits en el orden del stream de la fft2d.
      pos_coil_a : indice ((o * L + l) * Af + a)   (todo lo de una posicion junto)
      coil_pos_a : indice ((l * T + o) * Af + a)   (una bobina entera por vez)
    o = t1 * N2 + t2  <->  (x = br(t1), p = br(t2)).
    """
    n1, n2, af, L = qr.shape
    br1, br2 = bitrev_perm(n1), bitrev_perm(n2)
    hr = qr[br1][:, br2]                                  # [t1, t2, a, l]
    hi = qi[br1][:, br2]
    words = ((hi.astype(np.int64) & 0xFFFF) << 16) | (hr.astype(np.int64) & 0xFFFF)
    words = words.reshape(n1 * n2, af, L)                 # [o, a, l]
    if layout == "pos_coil_a":
        out = words.transpose(0, 2, 1)                    # [o, l, a]
    elif layout == "coil_pos_a":
        out = words.transpose(2, 0, 1)                    # [l, o, a]
    else:
        raise ValueError(layout)
    return np.ascontiguousarray(out).reshape(-1).astype(np.uint32)


def unpack_hw(buf: np.ndarray, n1: int, n2: int, af: int, L: int, layout: str = "pos_coil_a"):
    """Inversa de pack_hw, en orden del stream: (re, im) [o, l, a] como enteros con signo."""
    w = buf.astype(np.int64)
    if layout == "pos_coil_a":
        w = w.reshape(n1 * n2, L, af)
    else:
        w = w.reshape(L, n1 * n2, af).transpose(1, 0, 2)
    re = ((w & 0xFFFF) ^ 0x8000) - 0x8000
    im = (((w >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
    return re, im


# ---------------------------------------------------------------- self-test
def _selftest() -> int:
    import fft2d_ref as R
    from fft2d_mask_study import build_phantom

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FALLA'}] {name} {extra}")

    print("unfold_ref: self-test")
    rng = np.random.default_rng(7)
    # mapas aleatorios complejos, Af = 2 y 4, contra np.linalg.solve
    for af, L in ((2, 8), (4, 8), (3, 6)):
        n = 12 * af
        sm = rng.standard_normal((L, n, n)) + 1j * rng.standard_normal((L, n, n))
        for lam in (0.0, 0.3):
            Sr, Si = smaps_to_S(sm, af)
            Ar, Ai = gram(Sr, Si, lam)
            Lr, Li, d, dinv = ldlh(Ar, Ai)
            S = Sr + 1j * Si                                       # [x, p, l, a]
            Sh = np.conj(np.swapaxes(S, -1, -2))
            A = Sh @ S + lam * np.eye(af)
            Lc = Lr + 1j * Li
            rec = Lc @ (d[..., :, None] * np.conj(np.swapaxes(Lc, -1, -2)))
            e_ldl = np.abs(rec - A).max() / np.abs(A).max()
            Ur, Ui = solve_U(Lr, Li, dinv, Sr, Si)
            Ref = np.linalg.solve(A, Sh)
            e_u = np.abs((Ur + 1j * Ui) - Ref).max() / np.abs(Ref).max()
            check(f"Af={af} L={L} lam={lam}: L D L^H = A y U = solve",
                  e_ldl < 1e-13 and e_u < 1e-12, f"({e_ldl:.1e}, {e_u:.1e})")
            if lam == 0.0:
                US = (Ur + 1j * Ui) @ S
                e_i = np.abs(US - np.eye(af)).max()
                check(f"Af={af} L={L}: U S = I", e_i < 1e-12, f"({e_i:.1e})")

    # fantoma 256 x 256 (rapido): contra fft2d_ref.unfolding_matrix
    n, L, af = 256, 8, 2
    _, smaps, _, _ = build_phantom(HERE / "gen", n, L, af)
    U, valid = unfolding_ldlh(smaps, af)
    e = np.abs(U - R.unfolding_matrix(smaps, af)).max() / np.abs(U).max()
    check("fantoma 256: LDL^H = fft2d_ref.unfolding_matrix", e < 1e-13 and valid.all(), f"({e:.1e})")

    # cuantizacion y empaquetado
    g = np.linspace(2e-4, 3e-4, L)
    qr, qi, E, _ = scale_quantize(U.real, U.imag, g, cordic_gain(12))
    for layout in ("pos_coil_a", "coil_pos_a"):
        buf = pack_hw(qr, qi, layout)
        re, im = unpack_hw(buf, n, n // af, af, L, layout)
        br1, br2 = bitrev_perm(n), bitrev_perm(n // af)
        exp_r = qr[br1][:, br2].reshape(-1, af, L).transpose(0, 2, 1)
        exp_i = qi[br1][:, br2].reshape(-1, af, L).transpose(0, 2, 1)
        check(f"pack/unpack {layout}", np.array_equal(re, exp_r) and np.array_equal(im, exp_i)
              and buf.dtype == np.uint32 and buf.size == n * (n // af) * af * L)
    maxq = max(np.abs(qr).max(), np.abs(qi).max())
    check("E por frexp: max |U''_q| en [2^14, 2^15)", (1 << 14) <= maxq <= (1 << 15) - 1, f"(E = {E}, max {maxq})")
    print("== OK" if ok else "== FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())

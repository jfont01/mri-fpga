#!/usr/bin/env python3
"""
s2_direcciones.py -- generador de direcciones del corner-turn in situ, con 2 muestras por
palabra, y que la salida sea la transpuesta (con la permutacion elegida) de la entrada.

Notacion (un carril, una muestra por ciclo):
  * t = r*N1 + slot: indice de ESCRITURA (orden en que sale la pasada 1), nb = n1b + n2b bits.
  * s: indice de LECTURA (orden en que entra a la pasada 2), columna afuera, fila adentro.
  * Q(s): que indice t hay que leer en la posicion s.
       opcion A (orden del nucleo, H33):   s = c*N2 + k  ->  t = k*N1 + c          (rotacion)
       opcion B (x natural en la lectura): s = x*N2 + k  ->  t = k*N1 + br(x ^ f)   (f = N1/2
                                            si se absorbe el fftshift en x; si no, 0)
  * In situ: W_{j+1}(s) = R_j(s) = W_j(Q(s)), asi que W_j = W_0 o Q^j (W_0 = identidad).
  * Memoria: direccion fisica de nb bits; palabra = fis >> 1 (nb - 1 bits), mitad = fis & 1.
    Escritura de media palabra (BWE), lectura de la palabra entera y mux por la mitad.

Generadores (lo que haria el RTL):
  A: rotacion: dir = rotl_nb(cnt, k_j), k_{j+1} = (k_j + n1b) mod nb.     Periodo nb/gcd(n1b, nb).
  B: permutacion de bits + XOR: bit i de dir = bit sel_j[i] de cnt XOR c_j[i];
     sel_{j+1} = sel_j o P, c_{j+1} = c_j ^ pi_j(d)   (Q(s) = P(s) ^ d).

Chequeos:
  1. Tamano completo (1024 x 512), solo direcciones: 40 bobinas (A) y 2 periodos (B);
     el generador "de hardware" coincide con W_0 o Q^j compuesto explicitamente, y el
     lector recibe exactamente el t que corresponde (memoria de medias palabras).
  2. Sin BWE (escribir la palabra entera) se pisa la otra mitad sin leer: cuantas veces.
  3. Tamano chico (64 x 32), datos reales, 3 tramas x 8 bobinas encadenadas, simulacion
     ciclo a ciclo con el lector adelantado m ciclos sobre el escritor: la pasada 1 y la 2
     con el nucleo bit a bit (fft1d_r22sdf_fxp) y la salida comparada contra el modelo C++
     de la fft2d (memoria ideal ping-pong), compilado chico, bit a bit.

Uso:  python3 s2_direcciones.py --repo <clon>     (sale con 0 si todo da)
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def check(tag: str, cond: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if cond else 'MAL'}] {tag} {extra}")
    if not cond:
        FAILS.append(tag)


def bitrev(v: np.ndarray, nbits: int) -> np.ndarray:
    out = np.zeros_like(v)
    for i in range(nbits):
        out |= ((v >> i) & 1) << (nbits - 1 - i)
    return out


def rotl(v: np.ndarray, k: int, nb: int) -> np.ndarray:
    k %= nb
    mask = (1 << nb) - 1
    return ((v << k) | (v >> (nb - k))) & mask if k else v.copy()


class Geometry:
    def __init__(self, n1: int, n2: int):
        self.n1, self.n2 = n1, n2
        self.n1b, self.n2b = n1.bit_length() - 1, n2.bit_length() - 1
        self.nb = self.n1b + self.n2b
        self.T = n1 * n2
        self.s = np.arange(self.T, dtype=np.int64)

    # Q(s) para cada opcion
    def q_a(self) -> np.ndarray:
        c, k = self.s // self.n2, self.s % self.n2
        return k * self.n1 + c

    def q_b(self, fshift: bool) -> np.ndarray:
        x, k = self.s // self.n2, self.s % self.n2
        f = self.n1 // 2 if fshift else 0
        return k * self.n1 + bitrev(x ^ f, self.n1b)

    def as_bitmap(self, q: np.ndarray):
        """Q(s) = P(s) ^ d con P permutacion de bits: devuelve (src, d) con
        bit i de Q(s) = bit src[i] de s XOR d_i. Falla si Q no es de esa forma."""
        d = int(q[0])
        src = []
        for i in range(self.nb):
            hit = [b for b in range(self.nb) if ((int(q[1 << b]) ^ d) >> i) & 1]
            assert len(hit) == 1, "Q no es una permutacion de bits + XOR"
            src.append(hit[0])
        recon = np.full(self.T, d, dtype=np.int64)
        for i, b in enumerate(src):
            recon ^= ((self.s >> b) & 1) << i
        assert np.array_equal(recon, q), "Q no es afin"
        return src, d


# ---------------------------------------------------------------- generadores
class GenRot:
    """Opcion A: rotacion de nb bits, k += n1b por bobina."""
    def __init__(self, g: Geometry, k0: int = 0):
        self.g, self.k = g, k0

    def addr(self) -> np.ndarray:
        return rotl(self.g.s, self.k, self.g.nb)

    def next(self) -> None:
        self.k = (self.k + self.g.n1b) % self.g.nb


class GenPerm:
    """Opcion B: sel[i] (19 x 5 bits) y mascara c (19 bits)."""
    def __init__(self, g: Geometry, src, d):
        self.g, self.P, self.d = g, list(src), d
        self.sel = list(range(g.nb))   # W_0 = identidad
        self.c = 0

    def apply(self, v: np.ndarray) -> np.ndarray:
        out = np.full(v.shape, self.c, dtype=np.int64)
        for i, b in enumerate(self.sel):
            out ^= ((v >> b) & 1) << i
        return out

    def addr(self) -> np.ndarray:
        return self.apply(self.g.s)

    def next(self) -> None:
        # W_{j+1}(s) = W_j(P(s) ^ d) = pi_j(P(s)) ^ pi_j(d) ^ c_j
        pid = 0
        for i, b in enumerate(self.sel):
            pid |= ((self.d >> b) & 1) << i
        self.c ^= pid
        self.sel = [self.P[b] for b in self.sel]


def period_of(gen_factory, n_max: int) -> int:
    g0 = gen_factory()
    a0 = g0.addr()
    g = gen_factory()
    for j in range(1, n_max + 1):
        g.next()
        if np.array_equal(g.addr(), a0):
            return j
    return -1


# ------------------------------------------------- 1. tamano completo, direcciones
def full_size_check(g: Geometry, q: np.ndarray, make_gen, n_coils: int, tag: str) -> None:
    wgen, rgen = make_gen(), make_gen()
    rgen.next()                          # el lector de la bobina j usa W_{j+1}
    # composicion explicita de referencia
    w_ref = g.s.copy()                   # W_0
    mem = np.full((g.T // 2, 2), -1, dtype=np.int64)     # medias palabras con el t escrito
    ok_gen = ok_data = True
    for j in range(n_coils):
        wa = wgen.addr()
        ok_gen &= bool(np.array_equal(wa, w_ref))
        # escritura de la bobina j (medias palabras): guarda (j, t)
        mem[wa >> 1, wa & 1] = j * g.T + g.s
        # lectura de la bobina j
        ra = rgen.addr()
        got = mem[ra >> 1, ra & 1]
        ok_data &= bool(np.array_equal(got, j * g.T + q))
        w_ref = w_ref[q]                 # W_{j+1} = W_j o Q
        wgen.next()
        rgen.next()
    check(f"{tag}: generador == W_0 o Q^j en {n_coils} bobinas", ok_gen)
    check(f"{tag}: el lector recibe exactamente t = Q(s) en {n_coils} bobinas", ok_data)


def bwe_need(g: Geometry, make_gen, lead: int, n_coils: int) -> list[int]:
    """Sin BWE: por bobina, cuantas escrituras de la bobina j+1 pisan la otra mitad de una
    palabra que la bobina j todavia no leyo (lector adelantado 'lead' ciclos)."""
    rgen = make_gen()
    out = []
    for _ in range(n_coils):
        rgen.next()
        ra = rgen.addr()                   # direccion leida en la posicion s (= escrita por j+1 en s)
        inv = np.empty(g.T, dtype=np.int64)
        inv[ra] = g.s                      # posicion de lectura de cada direccion
        partner_read = inv[ra ^ 1]
        out.append(int(np.count_nonzero(partner_read > g.s + lead)))
    return out


# ------------------------------------------------ 3. tamano chico, datos reales
def small_size_data(repo: Path, opt: str, fshift: bool, lead: int, expect_ok: bool = True) -> None:
    sys.path.insert(0, str(repo / "model"))
    from fft2d_mask_study import build_phantom, q15, CORE        # noqa: E402
    from fft1d_r22sdf_fxp import Cfg, fft1d_r22sdf_fxp, load_twiddles   # noqa: E402

    n, L, af = 64, 8, 2
    n1, n2 = n, n // af
    g = Geometry(n1, n2)
    m1, m2 = 25, 4                          # mascaras del caso n64x32_phantom (doc/05 5.7)
    tw = repo / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"
    tw1, tw2 = load_twiddles(tw / f"n{n1}_tw18", n1, 18), load_twiddles(tw / f"n{n2}_tw18", n2, 18)
    gen_dir = repo / "model" / "gen"
    _, _, _, k_hw = build_phantom(gen_dir, n, L, af)              # [l, k', r]
    # 3 tramas distintas: rotacion de fase y de bobinas por trama
    frames = []
    for f in range(3):
        kf = np.roll(k_hw, f, axis=0) * np.exp(1j * 0.7 * f)
        peak = np.maximum(np.abs(kf.real), np.abs(kf.imag)).max(axis=(1, 2))
        frames.append(kf * (0.9 / peak)[:, None, None])
    kin = np.concatenate(frames, axis=0)                          # [24, n2, n1]
    qr, qi = q15(kin)
    n_coils = kin.shape[0]

    # pasada 1 bit a bit (orden de salida = slot): h[l, k', slot]
    c1 = Cfg(N=n1, SCALE_MASK=m1, **CORE)
    hr, hi = fft1d_r22sdf_fxp(qr, qi, c1, *tw1)
    wr_stream = (hr.reshape(n_coils, -1), hi.reshape(n_coils, -1))     # indice t = k'*N1 + slot

    if opt == "A":
        q = g.q_a()
        make = lambda: GenRot(g)                                       # noqa: E731
    else:
        q = g.q_b(fshift)
        src, d = g.as_bitmap(q)
        make = lambda: GenPerm(g, src, d)                              # noqa: E731

    # simulacion ciclo a ciclo: ciclo c: el escritor escribe la bobina j_w en la posicion
    # c - j_w*T (si hay), el lector lee la bobina j_r en c - lead - (j_r + 1)*T + ... ;
    # agenda continua: escritor de la bobina j en [j*T, (j+1)*T), lector de la bobina j en
    # [(j+1)*T - lead, (j+2)*T - lead) (adelantado 'lead' ciclos al escritor de j + 1).
    mem = np.zeros((g.T // 2, 2, 2), dtype=np.int64)     # [palabra, mitad, re/im]
    wgen, rgen = make(), make()
    rgen.next()
    wa_cur, ra_cur = wgen.addr(), rgen.addr()
    rd_re = np.zeros((n_coils, g.T), dtype=np.int64)
    rd_im = np.zeros_like(rd_re)
    total = (n_coils + 1) * g.T
    for c in range(total):
        # puerto A: lectura (antes que la escritura del mismo ciclo, UG573)
        jr, sr = divmod(c + lead - g.T, g.T)
        if 0 <= jr < n_coils and c + lead >= g.T:
            if sr == 0 and jr > 0:
                rgen.next(); ra_cur = rgen.addr()
            a = ra_cur[sr]
            word = mem[a >> 1]                            # palabra entera (2 muestras)
            rd_re[jr, sr], rd_im[jr, sr] = word[a & 1]    # mux por la mitad
        # puerto B: escritura de media palabra
        jw, sw = divmod(c, g.T)
        if jw < n_coils:
            if sw == 0 and jw > 0:
                wgen.next(); wa_cur = wgen.addr()
            a = wa_cur[sw]
            mem[a >> 1, a & 1] = (wr_stream[0][jw, sw], wr_stream[1][jw, sw])   # BWE
    # el lector de la ultima bobina corre solo al final (vaciado): ya incluido

    # pasada 2 bit a bit sobre lo leido: columnas de N2
    c2 = Cfg(N=n2, SCALE_MASK=m2, **CORE)
    ar, ai = fft1d_r22sdf_fxp(rd_re.reshape(n_coils, n1, n2), rd_im.reshape(n_coils, n1, n2), c2, *tw2)

    # referencia: modelo C++ de la fft2d (memoria ideal), compilado chico
    work = Path(__file__).resolve().parent / "build"
    work.mkdir(exist_ok=True)
    exe = work / "fft2d_dump_64x32"
    if not exe.exists():
        rt = repo / "rtlsim"
        cmd = ["g++", "-std=c++17", "-O2", "-w", "-isystem", str(rt / "include"), f"-I{rt}",
               f"-I{repo / 'modules/fft2d/cpp'}", "-DFFT2D_N1=64", "-DFFT2D_N2=32",
               "-DFFT2D_TW_DIR1=../fft1d_r22sdf/rtl/twiddles/n64_tw18",
               "-DFFT2D_TW_DIR2=../fft1d_r22sdf/rtl/twiddles/n32_tw18",
               str(repo / "modules/fft2d/testbench/fft2d_dump.cpp"), "-o", str(exe)]
        subprocess.run(cmd, check=True)
    kbin = work / "k24.bin"
    obin = work / "o24.bin"
    np.stack([qr, qi], axis=-1).astype("<i4").tofile(kbin)
    r = subprocess.run([str(exe), "--in", str(kbin), "--out", str(obin), "--coils", str(n_coils),
                        "--mask1", str(m1), "--mask2", str(m2)], cwd=repo / "modules/fft2d",
                       capture_output=True, text=True, check=True)
    ref = np.fromfile(obin, dtype="<i4").reshape(n_coils, n1, n2, 2)    # [l, t1, t2] orden hw
    if opt == "B":
        f = n1 // 2 if fshift else 0
        x = np.arange(n1)
        ref = ref[:, bitrev(x ^ f, g.n1b)]           # columna x natural (con fftshift: x ^ N1/2)
    same = np.array_equal(ar, ref[..., 0]) and np.array_equal(ai, ref[..., 1])
    tag = f"opcion {opt}{' + fftshift x' if fshift else ''}, 64x32, 24 bobinas, adelanto {lead}"
    n_bad = np.count_nonzero(ar != ref[..., 0]) + np.count_nonzero(ai != ref[..., 1])
    if expect_ok:
        check(f"{tag}: salida == modelo C++ fft2d bit a bit", same, f"({n_bad} distintas)")
    else:
        check(f"{tag}: (fuera de la ventana) la salida DIFIERE del modelo", not same, f"({n_bad} distintas)")


def lead_max(g: Geometry, q: np.ndarray) -> int:
    """Con escritor de la bobina j en jT + t y lector en (j+1)T - lead + s: RAW exige
    lead <= T - 1 - max(t(s) - s)."""
    return int(g.T - 1 - (q - g.s).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "repo")
    a = ap.parse_args()

    g = Geometry(1024, 512)
    print(f"1) Tamano completo: {g.n2} x {g.n1}, direccion de {g.nb} bits, palabra de {g.nb - 1}")
    qa = g.q_a()
    pa = period_of(lambda: GenRot(g), 64)
    check("A: periodo de la rotacion = 19", pa == 19, f"({pa})")
    full_size_check(g, qa, lambda: GenRot(g), 40, "A (rotacion)")
    for fs in (False, True):
        qb = g.q_b(fs)
        src, d = g.as_bitmap(qb)
        pb = period_of(lambda: GenPerm(g, src, d), 4000)
        print(f"   B{' + fftshift' if fs else ''}: Q es permutacion de bits + XOR (d = {d:#x}); "
              f"periodo {pb}; src = {src}")
        full_size_check(g, qb, lambda: GenPerm(g, src, d), min(2 * pb, 80) if pb > 0 else 40,
                        f"B{' + fftshift' if fs else ''}")

    print("\n2) Por que hace falta el byte write enable (opcion A, un periodo de 19 bobinas)")
    for lead in (1, 1023):
        na = bwe_need(g, lambda: GenRot(g), lead, 19)
        print(f"   adelanto {lead}: sin BWE, medias palabras sin leer que se pisarian por bobina: {na}")
        check(f"sin BWE hay pisadas en alguna bobina (adelanto {lead})", max(na) > 0)

    print("\n3) Tamano chico con datos reales contra el modelo C++ de la fft2d")
    gs = Geometry(64, 32)
    for opt, fs in (("A", False), ("B", False), ("B", True)):
        q = gs.q_a() if opt == "A" else gs.q_b(fs)
        lm = lead_max(gs, q)
        print(f"   opcion {opt}{' + fftshift' if fs else ''}: adelanto maximo {lm} "
              f"(N1 + N2 - 2 = {gs.n1 + gs.n2 - 2})")
        for lead in (0, 1, gs.n1 - 1, lm):
            small_size_data(a.repo, opt, fs, lead)
        small_size_data(a.repo, opt, fs, lm + 1, expect_ok=False)

    print("\n== OK" if not FAILS else f"\n== FALLA: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

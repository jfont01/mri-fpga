#!/usr/bin/env python3
"""
s4_reorden.py -- de la fft2d a unfold y de cordic_mag a la imagen: layouts en la DDR,
coil reorder por bloques, orden de U'' y escritura de la imagen con direcciones.

Solo movimiento de datos (enteros como etiquetas), con el orden de la opcion A (el stream
de la fft2d de hoy, o = t1*N2 + t2 <-> x = br(t1), p = br(t2)).

  1. A_l a la DDR en dos layouts:
       coil-major  [l][o]          (la fft2d escribe 2 MiB seguidos por bobina)
       por bloques [b][l][B]       (la fft2d escribe trozos de 4B bytes; unfold lee 32B bytes)
  2. coil reorder: por bloque b de B posiciones, 8 trozos -> BRAM ping-pong -> stream
     (o, l) con l adentro (lo que espera unfold, P2/decision 13) e i_last con l = L-1.
  3. U'' en pos_coil_a (unfold_ref.pack_hw): 2 palabras por ciclo, en el mismo orden; el MAC
     con el stream da lo mismo que el producto directo.
  4. Escritura de la imagen: por columna t1, 1024 pixeles a una BRAM con direccion
     y = br9(t2) + 512*a (con fftshift: 512*(1-a)) y la columna entera (2 KiB) a la DDR en
     x = br10(t1) (con fftshift: ^512), imagen x-major (m^T).

Uso: python3 s4_reorden.py --repo <clon>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def check(tag: str, cond: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if cond else 'MAL'}] {tag} {extra}")
    if not cond:
        FAILS.append(tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "repo")
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo / "model"))
    import unfold_ref as UR                                             # noqa: E402

    n1, n2, L, af = 1024, 512, 8, 2
    T = n1 * n2
    br1, br2 = UR.bitrev_perm(n1), UR.bitrev_perm(n2)
    rng = np.random.default_rng(5)
    # A_l en el orden del stream (etiqueta = valor al azar de 16+16 bits, como enteros)
    A = rng.integers(-2 ** 15, 2 ** 15, size=(L, T, 2), dtype=np.int64)

    # ------------------------------------------------------------- 1 y 2
    for B in (512, 1024):
        nb = T // B
        # coil-major: trozo (l, b) = A[l, b*B:(b+1)*B]; la DMA lo trae a la BRAM [buf][l][i]
        ddr_cm = A.reshape(L, nb, B, 2)                                 # [l][b][i]
        # por bloques: la fft2d de la bobina l escribe el trozo b en ((b*L + l)*B)
        ddr_bk = np.zeros((nb, L, B, 2), dtype=np.int64)
        for l in range(L):
            ddr_bk[:, l] = A[l].reshape(nb, B, 2)
        for name, ddr in (("coil-major", ddr_cm.transpose(1, 0, 2, 3)), ("por bloques", ddr_bk)):
            stream = []
            last = []
            bram = np.zeros((2, L, B, 2), dtype=np.int64)
            for b in range(nb):
                buf = b & 1
                bram[buf] = ddr[b]                                      # 8 trozos de 4B bytes (o uno de 32B)
                blk = bram[buf].transpose(1, 0, 2)                      # lectura: i afuera, l adentro
                stream.append(blk.reshape(-1, 2))
                last.append(np.tile(np.arange(L) == L - 1, B))
            stream = np.concatenate(stream)
            last = np.concatenate(last)
            exp = A.transpose(1, 0, 2).reshape(-1, 2)                   # [o][l]
            check(f"B = {B}, {name}: stream (o, l) == lo que espera unfold", np.array_equal(stream, exp))
            check(f"B = {B}, {name}: i_last cada {L}", bool(last.sum() == T and last[L - 1]))
        print(f"     B = {B}: trozos de {4 * B} B (coil-major: {L} por bloque, separados 2 MiB) o "
              f"bloques de {4 * B * L} B seguidos; BRAM ping-pong 2 x {L} x {B} x 32 b = "
              f"{2 * L * B * 4 // 1024} KiB = {2 * L * B * 4 // 4096} RAMB36")

    # ------------------------------------------------------------------ 3. U''
    print("\n3) U'' en pos_coil_a: el MAC con el stream da el producto directo")
    n1s, n2s, Ls = 64, 32, L
    Ts = n1s * n2s
    qr = rng.integers(-2 ** 15, 2 ** 15, size=(n1s, n2s, af, Ls))
    qi = rng.integers(-2 ** 15, 2 ** 15, size=(n1s, n2s, af, Ls))
    buf = UR.pack_hw(qr, qi, "pos_coil_a")
    ur, ui = UR.unpack_hw(buf, n1s, n2s, af, Ls, "pos_coil_a")          # [o, l, a]
    Ar = rng.integers(-2 ** 15, 2 ** 15, size=(Ls, Ts))
    Ai = rng.integers(-2 ** 15, 2 ** 15, size=(Ls, Ts))
    # stream: ciclo n = o*L + l lleva A[l, o] y U''[(o*L + l)*Af + a], a = 0, 1
    P_re = np.zeros((Ts, af), dtype=np.int64)
    P_im = np.zeros((Ts, af), dtype=np.int64)
    w = buf.astype(np.int64)
    for n in range(Ts * Ls):
        o, l = divmod(n, Ls)
        for av in range(af):
            word = w[n * af + av]
            u_r = ((word & 0xFFFF) ^ 0x8000) - 0x8000
            u_i = (((word >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
            P_re[o, av] += u_r * Ar[l, o] - u_i * Ai[l, o]
            P_im[o, av] += u_r * Ai[l, o] + u_i * Ar[l, o]
    # directo, en orden natural: U''[x, p, a, l] * A_nat[l, x, p]
    hr, hi = qr[UR.bitrev_perm(n1s)][:, UR.bitrev_perm(n2s)], qi[UR.bitrev_perm(n1s)][:, UR.bitrev_perm(n2s)]
    Ur, Ui = hr.reshape(Ts, af, Ls), hi.reshape(Ts, af, Ls)
    D_re = np.einsum("oal,lo->oa", Ur, Ar) - np.einsum("oal,lo->oa", Ui, Ai)
    D_im = np.einsum("oal,lo->oa", Ur, Ai) + np.einsum("oal,lo->oa", Ui, Ar)
    check("P del stream == U''[o] . A[o] (64 x 32, L = 8)", np.array_equal(P_re, D_re) and np.array_equal(P_im, D_im))
    print(f"     U'': {af} palabras de 32 b por ciclo = {af * 4 * 200e6 / 1e9:.1f} GB/s, en secuencia")

    # --------------------------------------------------------------- 4. imagen
    print("\n4) Escritura de la imagen (u16) con direcciones, imagen x-major (m^T)")
    N = n1
    for fsh in (False, True):
        img = np.full((N, N), -1, dtype=np.int64)                      # [x][y] en la DDR
        col_bram = np.zeros((2, N), dtype=np.int64)
        for t1 in range(n1):
            cb = t1 & 1
            # stream de cordic: posicion (t1, t2), a = 0, 1; etiqueta = id del pixel verdadero
            t2 = np.arange(n2)
            x_true = br1[t1] ^ (N // 2 if fsh else 0)
            for av in range(af):
                if fsh:
                    y_true = (br2[t2] + n2 * av + N // 2) % N             # coordenadas verdaderas
                    y_addr = br2[t2] + n2 * (1 - av)
                else:
                    y_true = br2[t2] + n2 * av
                    y_addr = y_true
                col_bram[cb, y_addr] = x_true * N + y_true
            x_addr = br1[t1] ^ (N // 2 if fsh else 0)
            img[x_addr] = col_bram[cb]                                   # 2 KiB seguidos
        ids = np.arange(N)[:, None] * N + np.arange(N)[None, :]
        check(f"imagen x-major con {'fftshift' if fsh else 'sin fftshift'}: cada pixel en su lugar",
              np.array_equal(img, ids))
    print(f"     columna: {N} x 16 b = {N * 2} B por rafaga; BRAM ping-pong 2 x 2 KiB = 1 RAMB36")

    print("\n== OK" if not FAILS else f"\n== FALLA: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

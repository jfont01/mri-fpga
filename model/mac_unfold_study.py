#!/usr/bin/env python3
"""
mac_unfold_study.py -- MAC de desplegado, puntos 1 a 3 (teoria, con NumPy).

Usa el fantoma del paso 4 y la salida BIT-EXACTA de la fft2d (cadena de
Python, mascaras 801 / 384, normalizacion por bobina), o sea exactamente lo
que entrega el modelo C++ validado.

  (1) Escalas. Comprueba que
          m_hat[x, p + a N/Af] = 2^(S1+S2-19) * sum_l (U[x,p,a,l] / g_l) * A_l[x,p]
      da lo mismo que el camino del estudio de mascaras (phys + unfold), y
      que con U'' = U' * 2^-E (U' = U / g_l, E comun por sesion)
          m_hat = 2^(S1+S2-19+E) * sum_l U''_l * A_l.
  (2) Formato de U''. Rango de U' por posicion (alcanza un E comun?) y
      barrido de bits NB_U: SNR de la imagen y SNR del error de U solo.
  (3) Acumulacion. Bits usados contra la cota NB_U + 20; la suma exacta no
      depende del orden de las bobinas; costo de guardar la suma parcial
      con W bits (por si se acumula bobina por bobina en memoria).

Uso (desde la raiz del repo):
    python3 model/mac_unfold_study.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft2d_ref as R                                                   # noqa: E402
from fft2d_mask_study import FS, Chain, build_phantom, popcount, snr_db  # noqa: E402


def quant(v, nb):
    """Q1.(nb-1) con redondeo al mas cercano y saturacion -> enteros."""
    f = nb - 1
    return np.clip(np.round(v * 2.0 ** f), -2.0 ** f, 2.0 ** f - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("--tw-dir", type=Path, default=HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles")
    ap.add_argument("-N", type=int, default=1024)
    ap.add_argument("-L", type=int, default=8)
    ap.add_argument("--af", type=int, default=2)
    ap.add_argument("--masks", default="801,384")
    ap.add_argument("--nb-u", type=int, default=16, help="NB_U para el punto 3")
    a = ap.parse_args()
    m1, m2 = (int(v) for v in a.masks.split(","))

    t0 = time.perf_counter()
    m, smaps, y_true, k_hw = build_phantom(a.gen_dir, a.N, a.L, a.af)
    ch = Chain(k_hw, smaps, m, a.af, a.tw_dir, "bobina")
    hr, hi = ch.pass1(m1)
    ar, ai = ch.pass2(hr, hi, m2)
    A = (ar + 1j * ai)[:, ch.ib1][:, :, ch.ib2]          # enteros Q1.15, [l, x, p]
    s1, s2 = popcount(m1, ch.L1), popcount(m2, ch.L2)
    S, LL, n, L = s1 + s2, ch.L1 + ch.L2, a.N, a.L
    U, g = ch.U, ch.g                                    # U[x, p, a, l]
    print(f"fantoma {n}x{n}, L={L}, Af={a.af}; mascaras {m1}/{m2} (S1={s1}, S2={s2}); "
          f"salida bit-exacta de la fft2d ({time.perf_counter() - t0:.1f} s)")

    # ------------------------------------------------------------------ (1)
    print("\n(1) ESCALAS")
    mh_study = R.unfold(ch.phys(A / FS, s1, s2), U, a.af)
    Up = U / g[None, None, None, :]
    mh_fac = 2.0 ** (S - LL) * np.einsum("xpal,lxp->xap", Up, A / FS).reshape(n, -1)
    print(f"  m_hat = 2^(S-{LL}) sum_l (U/g_l) A_l  vs  phys+unfold: "
          f"error relativo {np.abs(mh_fac - mh_study).max() / np.abs(mh_study).max():.1e}")
    comp = np.maximum(np.abs(Up.real), np.abs(Up.imag))
    E = int(np.ceil(np.log2(comp.max())))
    print(f"  g_l: {g.min():.3g} .. {g.max():.3g} (relacion {g.max() / g.min():.2f});  "
          f"max componente de U' = {comp.max():.4g} -> E = {E}")
    print(f"  m_hat = 2^(S-{LL}+E) sum_l U''_l A_l = 2^{S - LL + E} * (suma con U'' y A en [-1, 1))")
    print(f"  SNR imagen (U en float): {snr_db(m, mh_study):.3f} dB")

    # ------------------------------------------------------------------ (2)
    print("\n(2) FORMATO DE U''")
    pp = comp.max(axis=(2, 3))
    print("  max por posicion plegada, en dB respecto del max global: "
          + ", ".join(f"p{q} {20 * np.log10(np.percentile(pp, q) / comp.max()):.1f}" for q in (1, 50, 99)))
    Upp = Up * 2.0 ** (-E)
    mh_float = 2.0 ** (S - LL + E) * np.einsum("xpal,lxp->xap", Upp, A / FS).reshape(n, -1)
    print("  NB_U | SNR imagen | perdida | SNR solo de U | bits usados / cota NB_U+20")
    for nb in (8, 10, 12, 14, 15, 16, 18, 20):
        f = nb - 1
        Uq = quant(Upp.real, nb) + 1j * quant(Upp.imag, nb)
        P = np.einsum("xpal,lxp->xap", Uq, A)            # enteros exactos (< 2^53)
        used = int(np.ceil(np.log2(max(np.abs(P.real).max(), np.abs(P.imag).max()) + 1))) + 1
        mh = 2.0 ** (S - LL + E - 15 - f) * P.reshape(n, -1)
        s = snr_db(m, mh)
        print(f"  {nb:4d} | {s:9.3f}  | {snr_db(m, mh_float) - s:6.3f}  | {snr_db(mh_float, mh):8.2f} dB  | {used:2d} / {nb + 20}")

    # ------------------------------------------------------------------ (3)
    nb = a.nb_u
    f = nb - 1
    print(f"\n(3) ACUMULACION (NB_U = {nb})")
    Ur = quant(Upp.real, nb).astype(np.int64)
    Ui = quant(Upp.imag, nb).astype(np.int64)
    Arl = np.transpose(A.real.astype(np.int64), (1, 2, 0))[:, :, None, :]
    Ail = np.transpose(A.imag.astype(np.int64), (1, 2, 0))[:, :, None, :]
    Pr, Pi = Ur * Arl - Ui * Ail, Ur * Ail + Ui * Arl  # producto por bobina [x, p, a, l]
    ref_r, ref_i = Pr.sum(-1), Pi.sum(-1)
    scale = 2.0 ** (S - LL + E - 15 - f)

    def image(sr, si):
        return (scale * (sr + 1j * si)).transpose(0, 2, 1).reshape(n, -1)

    for name, idx in (("inverso", range(L - 1, -1, -1)), ("permutado", np.random.default_rng(0).permutation(L))):
        sr, si = np.zeros_like(ref_r), np.zeros_like(ref_i)
        for l in idx:
            sr += Pr[..., l]
            si += Pi[..., l]
        print(f"  suma exacta en orden {name}: identica a la natural: "
              f"{np.array_equal(sr, ref_r) and np.array_equal(si, ref_i)}")
    bound = nb + 20
    print(f"  suma parcial guardada con W bits (punto binario fijo para la cota de {bound}):")
    for W in (16, 20, 24, 28, 32):
        drop = bound - W
        q = lambda x: np.floor((x + (1 << (drop - 1))) / (1 << drop)).astype(np.int64) * (1 << drop)
        out = {}
        for name, idx in (("natural", range(L)), ("inverso", range(L - 1, -1, -1))):
            sr, si = np.zeros_like(ref_r), np.zeros_like(ref_i)
            for l in idx:
                sr, si = q(sr + Pr[..., l]), q(si + Pi[..., l])
            out[name] = (sr, si)
        same = all(np.array_equal(x, y) for x, y in zip(out["natural"], out["inverso"]))
        s = snr_db(m, image(*out["natural"]))
        print(f"    W={W:2d}: SNR {s:7.3f} dB (perdida {snr_db(m, image(ref_r, ref_i)) - s:6.3f}); "
              f"natural == inverso: {same}")
    words = a.af * L * (n * n // a.af)
    print(f"\n  U'' en memoria con {nb} bits (re + im en 32 bits): {words} complejos = "
          f"{words * 4 / 2**20:.1f} MiB; leido una vez por trama a 20 fps: {words * 4 * 20 / 1e6:.0f} MB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

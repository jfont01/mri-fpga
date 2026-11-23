#!/usr/bin/env python3
"""
s3_fftshift.py -- donde va el fftshift (y el (-1)^(x+y)) de la IFFT con datos centrados.

El hardware calcula la IDFT pura (sin shifts) y entrega bit-reversed en cada eje
(fft2d_ref.fft2d_hw_order). Se comparan tres convenciones del k-space de entrada:

  1. scripts del repo:          K = fft2(c)                         (sin shifts)
  2. gen_kspace --fftshift:     K = fftshift(fft2(c))               (check D de fft2d_ref)
  3. fastMRI (fft2c):           K = fftshift(fft2(ifftshift(c)))    (imagen tambien centrada)

c_l = S_l * m son las imagenes de bobina, con el objeto centrado en la imagen.
Con 3, el hardware sin shifts entrega (-1)^(x+p) * fold(ifftshift(c)): los CUADRANTES
cambiados, no solo signos. Se corrige gratis con direcciones:
  * mapas en coordenadas del hardware: S_hw = ifftshift(S) (lo hace el PS, una vez por sesion);
  * al escribir la imagen: x = br10(j1) XOR 512 (x + 512 mod 1024) e y = br9(j2) + 512*(1 - a)
    (y + 512 mod 1024 = cambiar a = 0 por a = 1).
Tambien: submuestreo en las lineas impares (delta = 1): las dos replicas se suman con
(-1)^a, que va a U (S_eff[l, a] = (-1)^a S[l, a]).

Uso: python3 s3_fftshift.py --repo <clon> [-N 1024]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def check(tag: str, err: float, tol: float = 1e-9) -> None:
    good = err <= tol
    print(f"  [{'OK ' if good else 'MAL'}] {tag}: {err:.2e}")
    if not good:
        FAILS.append(tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "repo")
    ap.add_argument("-N", type=int, default=1024)
    ap.add_argument("-L", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo / "model"))
    import fft2d_ref as R                                              # noqa: E402

    N, L, af = a.N, a.L, 2
    n2 = N // af
    gd = a.repo / "model" / "gen"
    gp, gs, gc = (R._load(gd, x) for x in ("gen_phantom", "gen_smaps", "gen_coils"))
    m = gp.gen_shepp_logan_2d(N=N, A=1.0).astype(np.complex128)
    S = gs.gen_sensitivity_maps_2d(N, L, radius_factor=1.0, sigma_factor=0.8, phase_scale=1.0,
                                   normalize=True)
    c = gc.compute_coil_images(m, S)                                    # [l, x, y] (y = PE)
    br1, br2 = R.bitrev_perm(N), R.bitrev_perm(n2)
    x = np.arange(N)[:, None]
    p = np.arange(n2)[None, :]
    sgn = (-1.0) ** (x + p)
    mm = np.abs(m).max()

    def hw_fold(K, delta=0):
        """k-space [l, r, e] completo -> lineas delta, delta+2, ... -> imagen plegada que
        calcula el hardware, en orden natural [l, x, p] (IDFT pura, escala de ifft2)."""
        k_alias = K[:, :, delta::af]
        return np.fft.ifft2(k_alias, axes=(-2, -1))

    U_true = R.unfolding_matrix(S, af)
    print(f"N = {N}, L = {L}, Af = {af}")

    print("1) K = fft2(c): el hardware da la imagen plegada en coordenadas verdaderas")
    y1 = hw_fold(np.fft.fft2(c, axes=(-2, -1)))
    check("y_hw == fold(c)", np.abs(y1 - R.fold(c, af)).max() / np.abs(y1).max())

    print("2) K = fftshift(fft2(c)) (gen_kspace --fftshift, check D): solo signos")
    y2 = hw_fold(np.fft.fftshift(np.fft.fft2(c, axes=(-2, -1)), axes=(-2, -1)))
    check("y_hw == (-1)^(x+p) fold(c)", np.abs(y2 - sgn * R.fold(c, af)).max() / np.abs(y2).max())
    check("|unfold(y_hw, U(S))| == |m|", np.abs(np.abs(R.unfold(y2, U_true, af)) - np.abs(m)).max() / mm)

    print("3) K = fftshift(fft2(ifftshift(c))) (fastMRI fft2c): cuadrantes cambiados")
    K3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(c, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    y3 = hw_fold(K3)
    cs = np.fft.ifftshift(c, axes=(-2, -1))
    check("y_hw == (-1)^(x+p) fold(ifftshift(c))", np.abs(y3 - sgn * R.fold(cs, af)).max() / np.abs(y3).max())
    bad = np.abs(np.abs(R.unfold(y3, U_true, af)) - np.abs(m)).max() / mm
    print(f"     con los mapas en coordenadas verdaderas, |unfold| - |m| = {bad:.2f} (relativo): MAL, "
          f"como se espera")
    if bad < 0.1:
        FAILS.append("se esperaba que los mapas sin shift fallaran")
    S_hw = np.fft.ifftshift(S, axes=(-2, -1))
    U_hw = R.unfolding_matrix(S_hw, af)
    mh = R.unfold(y3, U_hw, af)                                         # coordenadas del hardware
    check("|unfold(y_hw, U(ifftshift(S)))| == |ifftshift(m)|",
          np.abs(np.abs(mh) - np.abs(np.fft.ifftshift(m))).max() / mm)
    check("fftshift de eso == |m|", np.abs(np.abs(np.fft.fftshift(mh)) - np.abs(m)).max() / mm)

    print("   3b) todo en orden del hardware (stream de la fft2d, opcion A) y escritura con direcciones")
    a_hw = y3[:, br1][:, :, br2]                                        # [l, j1, j2]
    U_hw_s = U_hw[br1][:, br2]                                          # [j1, j2, a, l]  (orden del buffer)
    mo = np.einsum("xpal,lxp->xpa", U_hw_s, a_hw)                       # [j1, j2, a]
    out = np.zeros((N, N))
    j1 = np.arange(N)[:, None]
    j2 = np.arange(n2)[None, :]
    xa = br1[j1] ^ (N // 2)                                             # x verdadera
    for av in range(af):
        ya = br2[j2] + n2 * (1 - av)                                    # y verdadera (a cambiado)
        out[np.broadcast_to(xa, (N, n2)), np.broadcast_to(ya, (N, n2))] = np.abs(mo[:, :, av])
    check("imagen escrita con x = br10(j1)^512, y = br9(j2) + 512(1-a) == |m|", np.abs(out - np.abs(m)).max() / mm)

    print("   3c) opcion B con el fftshift de x absorbido en el corner-turn (columna x ^ 512)")
    xb = np.arange(N)
    a_b = y3[:, xb ^ (N // 2)][:, :, br2]                               # stream: x natural (verdadera), p br
    U_b = U_hw[xb ^ (N // 2)][:, br2]
    mo_b = np.einsum("xpal,lxp->xpa", U_b, a_b)
    out_b = np.zeros((N, N))
    for av in range(af):
        ya = br2[j2] + n2 * (1 - av)
        out_b[np.broadcast_to(xb[:, None], (N, n2)), np.broadcast_to(ya, (N, n2))] = np.abs(mo_b[:, :, av])
    check("imagen con x = contador natural, y = br9(j2) + 512(1-a) == |m|", np.abs(out_b - np.abs(m)).max() / mm)

    print("4) Submuestreo en las lineas impares (delta = 1): (-1)^a entre replicas, va a U")
    K1 = np.fft.fft2(c, axes=(-2, -1))
    yo = hw_fold(K1, delta=1)
    ph = np.exp(-2j * np.pi * np.arange(N) / N)[None, None, :]         # e^{-j 2 pi y / N} (delta = 1)
    check("y_hw == fold(c * e^{-j 2 pi y/N})",
          np.abs(yo - R.fold(c * ph, af)).max() / np.abs(yo).max())
    S_eff = S * ph                                                      # el factor comun en p no cambia |m|
    check("|unfold(y_hw, U(S * e^{-j2pi y/N}))| == |m|",
          np.abs(np.abs(R.unfold(yo, R.unfolding_matrix(S_eff, af), af)) - np.abs(m)).max() / mm)
    # e^{-j 2 pi (p + 512 a)/1024} = e^{-j pi p/512} (-1)^a: el (-1)^a es lo que importa
    S_a = S.copy()
    S_a[:, :, n2:] *= -1.0
    check("equivalente: S_eff[l, a] = (-1)^a S[l, a] (el resto es una fase comun)",
          np.abs(np.abs(R.unfold(yo, R.unfolding_matrix(S_a, af), af)) - np.abs(m)).max() / mm)

    print("\n== OK" if not FAILS else f"\n== FALLA: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
mac_output_study.py -- MAC de desplegado, puntos 4 y 5 (fantoma 1024 x 1024,
L = 8, Af = 2, mascaras 801 / 384, salida bit-exacta de la fft2d).

  A. U por L D L^H (unfold_ref.py, la referencia del PS) contra np.linalg.solve,
     U S = I, cuantizacion con 1/K plegado, tamano del buffer y cuentas por
     posicion.
  B. Mapas que se anulan fuera del objeto (como ESPIRiT): sin regularizar la
     factorizacion falla; barrido de Tikhonov lam = mu * escala.
  C. Interfaz PS -> PL: el MAC hecho con el buffer empaquetado y el stream de
     la fft2d en orden hardware da EXACTAMENTE lo mismo que en orden natural,
     para los dos layouts del buffer.
  D. Salida: redondeo de P, CORDIC (iteraciones n, bits de guarda G) y salida
     u16; SNR de la imagen de punta a punta. Figura opcional (--fig).

Uso (desde la raiz del repo):
    python3 model/mac_output_study.py
    python3 model/mac_output_study.py --fig build/mac_output.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft2d_ref as R                                                    # noqa: E402
from fft2d_mask_study import FS, Chain, build_phantom, popcount, snr_db   # noqa: E402
from mac_cordic_fxp import OutCfg, mac, output_stage                     # noqa: E402
from unfold_ref import (bitrev_perm, cordic_gain, gram, ldlh, ldlh_ok,    # noqa: E402
                        pack_hw, scale_quantize, smaps_to_S, solve_U,
                        unfolding_ldlh, unpack_hw)


def dilate(mask: np.ndarray, it: int) -> np.ndarray:
    """Dilatacion binaria con vecindad de 4 (como scipy.ndimage.binary_dilation), sin scipy."""
    out = mask.copy()
    for _ in range(it):
        prev = out.copy()
        out[1:, :] |= prev[:-1, :]
        out[:-1, :] |= prev[1:, :]
        out[:, 1:] |= prev[:, :-1]
        out[:, :-1] |= prev[:, 1:]
    return out


def flops_per_position(af: int, L: int) -> dict:
    """Cuentas reales (mul, add/sub, div) de gram + ldlh + solve_U + escala, por posicion."""
    off = af * (af - 1) // 2
    gram_ops = L * (af * (2 + 1) + off * (4 + 2)) + L * (af + 2 * off)   # productos + acumulacion
    ldl_ops = sum(2 * j + 1 + (af - 1 - j) * (8 * j + 2 + 2) for j in range(af))
    fwd = sum(8 * i for i in range(af)); bwd = sum(8 * (af - 1 - i) for i in range(af))
    solve_ops = L * (fwd + 2 * af + bwd)
    scale_ops = 2 * af * L
    return {"flops": gram_ops + ldl_ops + solve_ops + scale_ops, "div": af, "rint": 2 * af * L}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("--tw-dir", type=Path, default=HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles")
    ap.add_argument("--fig", type=Path, default=None, help="guardar la figura de la imagen final")
    a = ap.parse_args()

    n, L, af, m1, m2 = 1024, 8, 2, 801, 384
    n1, n2 = n, n // af
    T = n1 * n2
    t0 = time.perf_counter()
    m, smaps, y_true, k_hw = build_phantom(a.gen_dir, n, L, af)
    ch = Chain(k_hw, smaps, m, af, a.tw_dir, "bobina")
    hr, hi = ch.pass1(m1)
    ar, ai = ch.pass2(hr, hi, m2)                      # [l, t1, t2]: el stream de la fft2d
    S = popcount(m1, ch.L1) + popcount(m2, ch.L2)
    A_r, A_i = ar.reshape(L, T), ai.reshape(L, T)      # [l, o]
    br1, br2 = bitrev_perm(n1), bitrev_perm(n2)
    print(f"fantoma {n}x{n}, L={L}, Af={af}, mascaras {m1}/{m2} (S={S}); "
          f"salida bit-exacta de la fft2d ({time.perf_counter() - t0:.1f} s)")

    def to_image(v):                                   # v[o, a] -> imagen [x, y], y = a*N2 + p
        img = np.empty((n1, n2, af), dtype=v.dtype)
        img[br1[:, None], br2[None, :], :] = v.reshape(n1, n2, af)
        return img.transpose(0, 2, 1).reshape(n1, af * n2)

    # ------------------------------------------------------------------ A
    print("\nA. U POR L D L^H (referencia del PS)")
    Sr, Si = smaps_to_S(smaps, af)
    Lr, Li, d, dinv = ldlh(*gram(Sr, Si, 0.0))
    Ur, Ui = solve_U(Lr, Li, dinv, Sr, Si)
    U = Ur + 1j * Ui
    Uref = R.unfolding_matrix(smaps, af)
    S_ = Sr + 1j * Si
    print(f"  validas: {ldlh_ok(d).mean() * 100:.1f}%;  |U - solve| / max|U| = {np.abs(U - Uref).max() / np.abs(U).max():.1e};  "
          f"max |U S - I| = {np.abs(U @ S_ - np.eye(af)).max():.1e}")
    print(f"  pivotes d: min {d.min():.3f}, max {d.max():.3f} (mapas normalizados: d ~ 1)")
    fl = flops_per_position(af, L)
    print(f"  cuentas por posicion: {fl['flops']} flops + {fl['div']} divisiones + {fl['rint']} rint;  "
          f"por sesion: {fl['flops'] * T / 1e6:.0f} Mflop")
    n_it_ref = 12
    K = cordic_gain(n_it_ref)
    qr, qi, E, _ = scale_quantize(U.real, U.imag, ch.g, K, 16)
    qr2, qi2, E2, _ = scale_quantize(Uref.real, Uref.imag, ch.g, K, 16)
    ndiff = int((qr != qr2).sum() + (qi != qi2).sum())
    print(f"  U''_q (16 bits, 1/K_{n_it_ref} plegado): E = {E};  distintas de cuantizar el U de solve: "
          f"{ndiff} de {2 * qr.size} ({'+-1 LSB' if ndiff and max(np.abs(qr - qr2).max(), np.abs(qi - qi2).max()) == 1 else 'ninguna'})")
    buf = pack_hw(qr, qi, "pos_coil_a")
    print(f"  buffer: {buf.size} palabras de 32 bits = {buf.nbytes / 2**20:.1f} MiB")

    # ------------------------------------------------------------------ B
    print("\nB. MAPAS QUE SE ANULAN FUERA DEL OBJETO (tipo ESPIRiT)")
    mask = dilate(np.abs(m) > 0, 8)
    sm = smaps * mask[None]
    nin = mask.reshape(n, af, n2).transpose(0, 2, 1).sum(-1)
    print("  posiciones con 0 / 1 / 2 pixeles plegados adentro: "
          + " / ".join(f"{100 * (nin == k).mean():.1f}%" for k in range(3)))
    Sr_m, Si_m = smaps_to_S(sm, af)
    Ar_m, Ai_m = gram(Sr_m, Si_m, 0.0)
    _, _, d_m, _ = ldlh(Ar_m, Ai_m)
    scale = float(np.median(Ar_m[..., 0, 0][nin == 2]))
    print(f"  sin regularizar: factorizacion valida en {ldlh_ok(d_m).mean() * 100:.1f}% (d = 0 en el resto); "
          f"escala = mediana de diag(S^H S) = {scale:.3f}")
    A_nat = (ar + 1j * ai)[:, ch.ib1][:, :, ch.ib2] / FS
    mh_full = 2.0 ** (S - 19) * np.einsum("xpal,lxp->xap", ch.U / ch.g, A_nat).reshape(n, -1)
    print(f"  referencia (mapas completos): SNR adentro {snr_db(m[mask], mh_full[mask]):.2f} dB")
    for mu in (1e-6, 1e-4, 1e-3, 1e-2):
        Um, valid = unfolding_ldlh(sm, af, mu * scale)
        Up = Um / ch.g
        maxc = max(np.abs(Up.real).max(), np.abs(Up.imag).max())
        mh = 2.0 ** (S - 19) * np.einsum("xpal,lxp->xap", Up, A_nat).reshape(n, -1)
        print(f"  mu = {mu:5.0e}: validas {valid.mean() * 100:5.1f}%, E = {np.frexp(maxc)[1]}, "
              f"SNR adentro {snr_db(m[mask], mh[mask]):6.2f} dB, |m_hat| afuera max {np.abs(mh[~mask]).max():.1e}")

    # ------------------------------------------------------------------ C
    print("\nC. INTERFAZ PS -> PL EN ORDEN HARDWARE")
    A_int = (ar + 1j * ai)[:, ch.ib1][:, :, ch.ib2]                 # [l, x, p] natural
    Pn_r = np.zeros((n1, n2, af), dtype=np.int64)
    Pn_i = np.zeros_like(Pn_r)
    for l in range(L):                                               # natural: [x, p, a]
        ur, ui = qr[..., l].astype(np.int64), qi[..., l].astype(np.int64)
        xr, xi = A_int[l].real.astype(np.int64)[..., None], A_int[l].imag.astype(np.int64)[..., None]
        Pn_r += ur * xr - ui * xi
        Pn_i += ur * xi + ui * xr
    for layout in ("pos_coil_a", "coil_pos_a"):
        Ur_s, Ui_s = unpack_hw(pack_hw(qr, qi, layout), n1, n2, af, L, layout)
        Pr, Pi = mac(A_r, A_i, Ur_s, Ui_s)                           # [o, a], orden del stream
        same = np.array_equal(to_image(Pr), Pn_r.transpose(0, 2, 1).reshape(n1, -1)) and \
            np.array_equal(to_image(Pi), Pn_i.transpose(0, 2, 1).reshape(n1, -1))
        print(f"  layout {layout}: P en orden hardware == P en orden natural: {same}")
    print(f"  bits de P usados: {int(max(np.abs(Pr).max(), np.abs(Pi).max())).bit_length() + 1} "
          f"(cota exacta 16 + 16 + 3 = 35)")

    # ------------------------------------------------------------------ D
    F_OUT = 15
    D = 49 - S - E - F_OUT
    print(f"\nD. SALIDA: P -> M (G bits de guarda) -> CORDIC -> u16 con F_OUT = {F_OUT}  (D = 49 - S - E - F_OUT = {D})")
    ideal = None
    rows = []
    for n_it in (8, 10, 12, 16):
        Kn = cordic_gain(n_it)
        qr_n, qi_n, E_n, _ = scale_quantize(U.real, U.imag, ch.g, Kn, 16)
        Ur_s, Ui_s = unpack_hw(pack_hw(qr_n, qi_n), n1, n2, af, L)
        Pr_n, Pi_n = mac(A_r, A_i, Ur_s, Ui_s)
        D_n = 49 - S - E_n - F_OUT
        ideal_n = np.hypot(Pr_n.astype(float), Pi_n.astype(float)) * Kn * 2.0 ** (-D_n)
        for G in (0, 3, 4, 5):
            cfg = OutCfg(D=D_n, G=G, N_IT=n_it, W_M=17 + G, W_OUT=16)
            out, _, _ = output_stage(Pr_n, Pi_n, cfg)
            err = out - ideal_n
            img = to_image(out.astype(float) * 2.0 ** -F_OUT)
            rows.append((n_it, G, np.abs(err).max(), np.sqrt(np.mean(err ** 2)), err.mean(), snr_db(m, img), out.max()))
            if n_it == n_it_ref and G == 5:
                final_img, ideal = img, to_image(ideal_n * 2.0 ** -F_OUT)
    print("   n   G | error vs |P| exacta (LSB de salida): max   rms    media | SNR imagen | max salida")
    for r in rows:
        print(f"  {r[0]:2d}  {r[1]}  |                                  {r[2]:5.2f}  {r[3]:5.3f}  {r[4]:+6.3f} | {r[5]:8.2f}   | {r[6]}")
    print(f"  referencias: |m_hat| exacta {snr_db(m, ideal):.2f} dB;  m_hat complejo (punto 3) "
          f"{snr_db(m, mh_full):.2f} dB;  redondeo ideal a u16: rms 0,289 LSB")
    print(f"  elegido: n = {n_it_ref}, G = 5 -> SNR {snr_db(m, final_img):.2f} dB")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
        ax[0].imshow(np.abs(m).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[0].set_title("fantoma original |m|")
        ax[1].imshow(final_img.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[1].set_title(f"salida u16 del PL (MAC + CORDIC n={n_it_ref}, G=5)\nSNR {snr_db(m, final_img):.2f} dB")
        im = ax[2].imshow((20 * np.log10(np.abs(final_img - np.abs(m)) + 1e-12)).clip(-120, None).T,
                          cmap="magma", origin="lower", vmin=-120, vmax=-20)
        ax[2].set_title("| salida - |m| |  [dB re 1]")
        fig.colorbar(im, ax=ax[2], shrink=0.8)
        for x in ax:
            x.set_xlabel("x (readout)"); x.set_ylabel("y (phase-encode)")
        a.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=100)
        print(f"  figura: {a.fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

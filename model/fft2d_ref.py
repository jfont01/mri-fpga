#!/usr/bin/env python3
"""
fft2d_ref.py -- referencia de la FFT 2D de la tesis (paso 1 del plan FFT 2D).

Define, en un solo lugar, QUE tiene que calcular el hardware por bobina y en
que orden lo entrega, y lo ata a los scripts de estimulos (gen_*.py) que ya
reproducen el plegado de Pruessmann.

CONVENCIONES
------------
Scripts gen_*:  K[l, r, e]  de forma (L, N, N), con K = fft2(coil_imgs) SIN
                fftshift. gen_aliased_kspace (axis="y", default) se queda con
                e = 0, Af, 2Af, ... del ULTIMO eje:  K_alias[l, r, k']  (L, N, N/Af).

                  r : readout (eje 1, "Nx" en los scripts), completo
                  e : phase-encode (eje 2, "Ny" en los scripts), submuestreado
                  (Ojo: en gen_phantom/gen_smaps las imagenes son [Y, X] por
                  np.mgrid, asi que el eje "Ny" de los scripts es la columna X.)

Hardware:       por bobina, las N/Af lineas adquiridas, cada una con las N
                muestras de readout CONTIGUAS (asi llega una adquisicion):

                  K_hw[l, k', r] = K_alias[l, r, k']        forma (L, N/Af, N)

                Pasada 1: IDFT de N     sobre r,  fila por fila     (N = 1024)
                Pasada 2: IDFT de N/Af  sobre k', columna por columna (512)
                Salida:   imagen plegada A[l, p, x] = a_alias[l, x, p]

                Sin fftshift/ifftshift en ningun lado: el hardware calcula la
                IDFT pura. Con salida en magnitud, centrar o no el k-space solo
                cambia signos (-1)^(x+p) que se cancelan en |m| (ver check D).

Escalado:       el nucleo calcula 2^-S * sum(...) por pasada (S = bits en 1 de
                SCALE_MASK), sin el 1/N de numpy. Con mascaras completas
                (S1 = log2 N, S2 = log2 N/Af) la salida es exactamente
                np.fft.ifft2. En general:  A = 2^(log2(N*N/Af) - S1 - S2) * ifft2.

IFFT:           con el mismo nucleo de FFT:  N*IFFT(X) = swap(FFT(swap(X))),
                swap(a + jb) = b + ja (cableado, sin negar nada).

Orden:          cada pasada entrega su eje en BIT-REVERSED. La pasada 2 opera
                por columnas, asi que el orden de columnas no le importa:

                  A_hw[l, j2, j1] = A[l, bitrev9(j2), bitrev10(j1)]

                El unfolding es punto a punto por posicion plegada (x, p): con U
                guardado en ese mismo orden, solo la escritura de la imagen
                final necesita direcciones en orden natural (ver check C).

Plegado:        a_alias[l, x, p] = sum_{a=0}^{Af-1} c_l[x, p + a*N/Af]
                (Pruessmann, con la normalizacion de ifft2 sobre (N, N/Af):
                no aparece 1/Af).

Ubicacion en el repo: model/fft2d_ref.py, con los scripts de estimulos en
model/gen/ (es el --gen-dir por defecto). Se importa desde el estudio de
mascaras (paso 4) y desde el generador de vectores de la fft2d (paso 5).

Uso (self-test, N = 1024, L = 8, Shepp-Logan):
    python3 model/fft2d_ref.py [--gen-dir model/gen] [-N 1024] [-L 8] [--af 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

C = NDArray[np.complex128]


# ----------------------------------------------------------------- utilidades
def bitrev_perm(n: int) -> NDArray[np.intp]:
    """br[j] = bitrev(j) en log2(n) bits."""
    b = n.bit_length() - 1
    if n != 1 << b:
        raise ValueError(f"n={n} no es potencia de 2")
    j = np.arange(n)
    r = np.zeros(n, dtype=np.intp)
    for i in range(b):
        r |= ((j >> i) & 1) << (b - 1 - i)
    return r


def swap(z: C) -> C:
    """swap(a + jb) = b + ja."""
    return z.imag + 1j * z.real


# ------------------------------------------------------- referencia (natural)
def to_hw_layout(k_alias: C) -> C:
    """K_alias[l, r, k'] (scripts) -> K_hw[l, k', r] (lineas de readout contiguas)."""
    return np.ascontiguousarray(np.swapaxes(k_alias, -1, -2))


def idft_pass(x: C, axis: int, s: int) -> C:
    """Una pasada del nucleo con INVERSE, en orden natural: 2^-S * sum X[k] e^{+j2pi kn/N}."""
    n = x.shape[axis]
    return np.fft.ifft(x, axis=axis) * n * 2.0 ** (-s)


def fft2d_golden(k_hw: C, s1: int, s2: int) -> C:
    """Imagen plegada A[..., p, x] en orden natural, con el escalado de las mascaras."""
    return idft_pass(idft_pass(k_hw, -1, s1), -2, s2)


# ------------------------------------------- modelo de flujo (orden hardware)
def core_pass_hw(x: C, axis: int, s: int) -> C:
    """
    Lo que hace un nucleo R2^2SDF con INVERSE sobre cada vector del eje 'axis':
    swap -> FFT (2^-S) -> swap, y entrega el eje en bit-reversed.
    """
    n = x.shape[axis]
    y = swap(np.fft.fft(swap(x), axis=axis)) * 2.0 ** (-s)
    return np.take(y, bitrev_perm(n), axis=axis)


def fft2d_hw_order(k_hw: C, s1: int, s2: int) -> C:
    """
    Pasada 1 por filas (readout), corner-turn implicito, pasada 2 por columnas.
    Devuelve A_hw[..., j2, j1] = A[..., bitrev(j2), bitrev(j1)].
    """
    h_hw = core_pass_hw(k_hw, -1, s1)   # [k', j1]: cada fila en bit-reversed de x
    return core_pass_hw(h_hw, -2, s2)   # [j2, j1]: cada columna en bit-reversed de p


def hw_to_natural(a_hw: C) -> C:
    """Deshace el bit-reversed de los dos ejes."""
    n2, n1 = a_hw.shape[-2:]
    i2, i1 = np.argsort(bitrev_perm(n2)), np.argsort(bitrev_perm(n1))
    return a_hw[..., i2, :][..., :, i1]


# ------------------------------------------------------------ SENSE (float)
def fold(coils: C, af: int) -> C:
    """Plegado de Pruessmann sobre el ultimo eje: sum_a c[..., p + a*N/Af]."""
    n = coils.shape[-1]
    return coils.reshape(*coils.shape[:-1], af, n // af).sum(axis=-2)


def unfolding_matrix(smaps: C, af: int) -> C:
    """
    U = (S^H S)^-1 S^H por posicion plegada (x, p), con S[l, a] = s_l[x, p + a*N/Af].
    smaps: (L, N, N) en la convencion de los scripts. Devuelve U[x, p, a, l].
    Falla (LinAlgError) si S^H S es singular en alguna posicion: pasa, por
    ejemplo, con L = Af = 4 y estos mapas simetricos. Con L = 8, Af = 2 o 4, no.
    """
    L, n, _ = smaps.shape
    S = smaps.reshape(L, n, af, n // af).transpose(1, 3, 0, 2)   # [x, p, l, a]
    Sh = np.conj(np.swapaxes(S, -1, -2))                          # [x, p, a, l]
    return np.linalg.solve(Sh @ S, Sh)


def unfold(y_alias: C, U: C, af: int) -> C:
    """y_alias[l, x, p], U[x, p, a, l] -> m_hat[x, p + a*N/Af]."""
    m = np.einsum("xpal,lxp->xap", U, y_alias)          # [x, a, p]
    return m.reshape(m.shape[0], -1)                    # [x, a*N/Af + p]


# ------------------------------------------------------------------ self-test
def _load(gen_dir: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, gen_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gen-dir", type=Path, default=Path(__file__).resolve().parent / "gen",
                    help="carpeta con gen_*.py (default: <carpeta de este script>/gen)")
    ap.add_argument("-N", type=int, default=1024)
    ap.add_argument("-L", type=int, default=8)
    ap.add_argument("--af", type=int, default=2)
    args = ap.parse_args()
    N, L, Af = args.N, args.L, args.af
    n2 = N // Af
    ok = True

    def check(tag: str, err: float, tol: float) -> None:
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"  [{'OK ' if good else 'MAL'}] {tag}: {err:.3e} (tol {tol:.0e})")

    gp = _load(args.gen_dir, "gen_phantom")
    gs = _load(args.gen_dir, "gen_smaps")
    gc = _load(args.gen_dir, "gen_coils")
    gk = _load(args.gen_dir, "gen_kspace")
    ga = _load(args.gen_dir, "gen_aliased_kspace")
    gl = _load(args.gen_dir, "gen_coil_aliased")

    # Defaults de linea de comandos de gen_smaps (radius 1.0, sigma 0.8, phase 1.0)
    m = gp.gen_shepp_logan_2d(N=N, A=1.0).astype(np.complex128)
    smaps = gs.gen_sensitivity_maps_2d(N, L, radius_factor=1.0, sigma_factor=0.8,
                                       phase_scale=1.0, normalize=True)
    coils = gc.compute_coil_images(m, smaps)
    K = gk.compute_kspace_from_coils(coils, use_fftshift=False)
    K_alias = ga.undersample_kspace_ny(K, Af)                 # (L, N, N/Af)
    y_alias = gl.reconstruct_images_ifft2(K_alias)            # (L, N, N/Af)
    print(f"N={N}  L={L}  Af={Af}   K_alias {K_alias.shape} -> K_hw {(L, n2, N)}")

    scale = np.abs(y_alias).max()
    print("A) plegado: ifft2(K_alias) == sum_a c[x, p + a*N/Af] (Pruessmann)")
    check("max|ifft2 - fold| / max|y|", np.abs(y_alias - fold(coils, Af)).max() / scale, 1e-12)

    print("B) flujo del hardware (swap-FFT, bit-reversed por pasada, corner-turn)")
    k_hw = to_hw_layout(K_alias)
    s1, s2 = N.bit_length() - 1, n2.bit_length() - 1
    A = fft2d_golden(k_hw, s1, s2)
    check("golden(S completas) == ifft2 transpuesta",
          np.abs(A - np.swapaxes(y_alias, -1, -2)).max() / scale, 1e-12)
    A_hw = fft2d_hw_order(k_hw, s1, s2)
    check("A_hw == A[bitrev(j2), bitrev(j1)]",
          np.abs(A_hw - A[:, bitrev_perm(n2)][:, :, bitrev_perm(N)]).max() / scale, 1e-12)
    check("hw_to_natural(A_hw) == A", np.abs(hw_to_natural(A_hw) - A).max() / scale, 1e-12)
    # Mascaras parciales: solo cambia una potencia de 2
    for (t1, t2) in ((4, 3), (8, 1), (0, 0)):
        g = 2.0 ** (s1 + s2 - t1 - t2)
        check(f"S1={t1}, S2={t2}: A = 2^{s1 + s2 - t1 - t2} * ifft2",
              np.abs(fft2d_golden(k_hw, t1, t2) - g * A).max() / (g * scale), 1e-12)

    print("C) unfolding SENSE en float, en orden natural y en orden hardware")
    U = unfolding_matrix(smaps, Af)                           # [x, p, a, l]
    m_hat = unfold(y_alias, U, Af)
    check("max|m_hat - m| / max|m|", np.abs(m_hat - m).max() / np.abs(m).max(), 1e-9)
    # Orden hardware: y_hw[l, j2, j1], U_hw con el mismo orden; solo se reordena al escribir.
    br2, br1 = bitrev_perm(n2), bitrev_perm(N)
    U_hw = U[br1][:, br2]                                     # [j1, j2, a, l]
    y_hw = np.swapaxes(A_hw, -1, -2)                          # [l, j1, j2]
    m_hw = np.einsum("xpal,lxp->xap", U_hw, y_hw)              # [j1, a, j2]
    m_out = np.empty((N, N), dtype=np.complex128)
    for a in range(Af):                                       # escritura con direcciones
        m_out[np.ix_(br1, a * n2 + br2)] = m_hw[:, a, :]
    check("unfolding en orden hw + escritura reordenada == m",
          np.abs(m_out - m).max() / np.abs(m).max(), 1e-9)

    print("D) k-space centrado (fftshift, como fastMRI o gen_kspace --fftshift)")
    Kc = gk.compute_kspace_from_coils(coils, use_fftshift=True)
    yc = gl.reconstruct_images_ifft2(ga.undersample_kspace_ny(Kc, Af))
    x = np.arange(N)[:, None]
    p = np.arange(n2)[None, :]
    sgn = (-1.0) ** (x + p)
    check("ifft2 del centrado == (-1)^(x+p) * sin centrar",
          np.abs(yc - sgn * y_alias).max() / scale, 1e-12)
    check("|unfold(centrado)| == |m|",
          np.abs(np.abs(unfold(yc, U, Af)) - np.abs(m)).max() / np.abs(m).max(), 1e-9)

    print("E) rango dinamico, pico/rms por componente (para el paso 4)")

    def par_db(arr: C) -> float:
        comp = np.concatenate([arr.real.ravel(), arr.imag.ravel()])
        return 20 * np.log10(np.abs(comp).max() / np.sqrt(np.mean(comp ** 2)))

    for name, arr in (("k-space K_hw", k_hw), ("tras pasada 1", idft_pass(k_hw, -1, 0)),
                      ("imagen plegada", fft2d_golden(k_hw, 0, 0))):
        per_coil = [par_db(arr[l]) for l in range(L)]
        print(f"  {name:15s} {L} bobinas juntas {par_db(arr):5.1f} dB;"
              f"  por bobina {min(per_coil):5.1f} a {max(per_coil):5.1f} dB")

    print("TODO OK" if ok else "HAY FALLAS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

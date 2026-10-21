#!/usr/bin/env python3
"""
fft2d_mask_study.py -- paso 4 del plan FFT 2D: eleccion de SCALE_MASK para las
dos pasadas (filas de 1024, columnas de 512) y de la normalizacion de entrada.

Cadena simulada bit a bit (fft1d_r22sdf_fxp, validado contra el RTL), con la
configuracion elegida para el nucleo (datapath 20, twiddle 18, puertos 16):

    k-space float -> normalizacion -> Q1.15 -> pasada 1 (IFFT 1024, mascara m1)
                  -> Q1.15 (intermedio en DDR) -> pasada 2 (IFFT 512, mascara m2)
                  -> Q1.15 -> desplegado SENSE en float (U = (S^H S)^-1 S^H) -> m_hat

Metricas (SNR en dB, energia de la referencia sobre energia del error):
  SNR_plegada : imagenes plegadas contra el plegado exacto de las bobinas.
  SNR_imagen  : imagen desplegada m_hat contra el fantoma m (complejo). Es la
                que importa: incluye como pesa cada bobina en el desplegado.
  techos      : (a) solo cuantizar la entrada; (b) entrada + intermedio y salida
                en 16 bits con FFT ideal. Dicen cuanto se puede ganar con mascaras.
  margen      : 20 log10(1 / pico) del peor punto interno de cada pasada.

Normalizacion de entrada:
  comun    : una escala para las 8 bobinas (pico de la mas fuerte -> 0,9).
  bobina   : una escala por bobina (cada pico -> 0,9). La ganancia de cada bobina
             se compensa despues en U, que precalcula el PS: no cuesta hardware.

Mascaras:
  lazy(theta) : se escala una etapa solo cuando, sin escalarla, el pico (calculado
                en float sin escalar, incluyendo la salida de cada cmul) superaria
                theta. theta = 1: sin margen; theta = 0,5: 6 dB de margen interno.
  --brute     : barrido exhaustivo de todas las ubicaciones para S1 y S2 dados.

Uso:
    python3 model/fft2d_mask_study.py [--norm comun|bobina|ambas] [--eval 801,384 ...]
            [--brute] [--tw-dir modules/fft1d_r22sdf/rtl/twiddles] [-N 1024] [-L 8]

  Sin --brute tarda ~1 minuto. --brute barre todas las ubicaciones con S1 y S2
  alrededor de lo mejor (~20 minutos): confirma que la ubicacion casi no importa.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fft2d_ref as R                                    # noqa: E402
from fft1d_r22sdf_fxp import Cfg, Stats, fft1d_r22sdf_fxp, load_twiddles   # noqa: E402

CORE = dict(NB=20, NBF=19, NB_TW=18, NB_IN=16, NB_OUT=16, INVERSE=1)
FS = 2 ** 15                       # escala de Q1.15


def comp_peak(a):
    return float(np.maximum(np.abs(a.real), np.abs(a.imag)).max())


def snr_db(ref, x):
    return float(10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs(x - ref) ** 2)))


def popcount(v, n):
    return bin(v & ((1 << n) - 1)).count("1")


def q15(x):
    """float (escala 1 = fondo de escala) -> enteros Q1.15 con redondeo y saturacion."""
    return (np.clip(np.round(x.real * FS), -FS, FS - 1).astype(np.int64),
            np.clip(np.round(x.imag * FS), -FS, FS - 1).astype(np.int64))


# ------------------------------------------------- picos por etapa en float
def unscaled_stage_peaks(x, n):
    """
    Pico de componente (max |re|, |im|) despues de cada etapa radix-2, sin
    escalar, para la IFFT del nucleo (swap -> FFT R2^2 -> swap; el swap no
    cambia los picos). En las etapas impares de cada unidad se toma el maximo
    entre la salida de la mariposa y la del cmul (una rotacion puede subir el
    pico de componente hasta sqrt(2)).
    """
    L2 = n.bit_length() - 1
    y = R.swap(np.asarray(x, dtype=complex)).reshape(-1, n)
    W = np.exp(-2j * np.pi * np.arange(n) / n)
    peaks = []
    log_m = L2
    while log_m >= 2:
        m = 1 << log_m
        b = y.reshape(-1, n // m, m)
        a, c = b[..., :m // 2], b[..., m // 2:]
        b = np.concatenate([a + c, a - c], -1)
        b[..., 3 * m // 4:] *= -1j
        peaks.append(comp_peak(b))
        parts = []
        for h in (0, m // 2):
            a, c = b[..., h:h + m // 4], b[..., h + m // 4:h + m // 2]
            parts += [a + c, a - c]
        b = np.concatenate(parts, -1)
        p = comp_peak(b)
        if log_m != 2:
            i = np.arange(m)
            sel = 2 * ((i >> (log_m - 2)) & 1) + ((i >> (log_m - 1)) & 1)
            num = ((i % (m // 4)) << (L2 - log_m)) & ((1 << (L2 - 2)) - 1)
            b = b * W[(num * sel) % n]
            p = max(p, comp_peak(b))
        peaks.append(p)
        y = b.reshape(-1, n)
        log_m -= 2
    if log_m == 1:
        a, c = y[..., 0::2], y[..., 1::2]
        peaks.append(max(comp_peak(a + c), comp_peak(a - c)))
    return np.array(peaks)


def lazy_mask(peaks, theta):
    """Escala solo donde hace falta para que ningun pico supere theta. None si no se puede."""
    g, mask = 1.0, 0
    for j, p in enumerate(peaks):
        if p * g >= theta:
            g /= 2
            mask |= 1 << j
        if p * g >= theta:
            return None
    return mask


# ---------------------------------------------------------------- la cadena
class Chain:
    def __init__(self, k_hw, smaps, m_true, af, tw_dir: Path, norm: str):
        self.L, self.n2, self.n1 = k_hw.shape
        self.af, self.m = af, m_true
        n1, n2 = self.n1, self.n2
        # normalizacion
        if norm == "comun":
            self.g = np.full(self.L, 0.9 / comp_peak(k_hw))
        else:
            self.g = np.array([0.9 / comp_peak(k_hw[l]) for l in range(self.L)])
        self.xin = k_hw * self.g[:, None, None]
        self.qr, self.qi = q15(self.xin)
        self.tw1 = load_twiddles(tw_dir / f"n{n1}_tw18", n1, 18)
        self.tw2 = load_twiddles(tw_dir / f"n{n2}_tw18", n2, 18)
        self.U = R.unfolding_matrix(smaps, af)
        self.ib1 = np.argsort(R.bitrev_perm(n1))
        self.ib2 = np.argsort(R.bitrev_perm(n2))
        self.y_true = None
        self.L1, self.L2 = n1.bit_length() - 1, n2.bit_length() - 1

    def set_truth(self, y_alias_true):
        self.y_true = y_alias_true                       # [l, x, p], fisico

    def phys(self, a_nat, s1, s2):
        """A en Q1.15 (orden natural [l, x, p]) -> unidades fisicas de y_alias."""
        return a_nat * 2.0 ** (s1 + s2 - self.L1 - self.L2) / self.g[:, None, None]

    def pass1(self, m1, stats=None):
        c = Cfg(N=self.n1, SCALE_MASK=m1, **CORE)
        return fft1d_r22sdf_fxp(self.qr, self.qi, c, *self.tw1, stats)        # [l, k', j1]

    def pass2(self, hr, hi, m2, stats=None):
        c = Cfg(N=self.n2, SCALE_MASK=m2, **CORE)
        return fft1d_r22sdf_fxp(np.swapaxes(hr, 1, 2), np.swapaxes(hi, 1, 2), c, *self.tw2, stats)  # [l, j1, j2]

    def evaluate(self, m1, m2, h=None):
        st1, st2 = Stats(), Stats()
        hr, hi = h if h is not None else self.pass1(m1, st1)
        ar, ai = self.pass2(hr, hi, m2, st2)
        a = (ar + 1j * ai) / FS
        a_nat = a[:, self.ib1][:, :, self.ib2]                               # [l, x, p]
        s1, s2 = popcount(m1, self.L1), popcount(m2, self.L2)
        y = self.phys(a_nat, s1, s2)
        mh = R.unfold(y, self.U, self.af)
        return dict(m1=m1, m2=m2, S1=s1, S2=s2,
                    sat1=None if h is not None else st1.total_sat, sat2=st2.total_sat,
                    peak1=None if h is not None else max(st1.peak.values()),
                    peak2=max(st2.peak.values()),
                    snr_alias=snr_db(self.y_true, y), snr_img=snr_db(self.m, mh))

    def ceilings(self, s1, s2):
        """(a) solo entrada cuantizada; (b) + intermedio y salida en Q1.15, FFT ideal."""
        xq = (self.qr + 1j * self.qi) / FS
        a = R.fft2d_golden(xq, s1, s2)                                       # [l, p, x]
        y_a = self.phys(np.swapaxes(a, 1, 2), s1, s2)
        h = R.idft_pass(xq, -1, s1)
        hq = (lambda t: (t[0] + 1j * t[1]) / FS)(q15(h))
        a_b = R.idft_pass(hq, -2, s2)
        a_b = (lambda t: (t[0] + 1j * t[1]) / FS)(q15(a_b))
        y_b = self.phys(np.swapaxes(a_b, 1, 2), s1, s2)
        return {k: dict(snr_alias=snr_db(self.y_true, y), snr_img=snr_db(self.m, R.unfold(y, self.U, self.af)))
                for k, y in (("a_entrada", y_a), ("b_entrada_y_16b", y_b))}


# ----------------------------------------------------------------- estudio
def build_phantom(gen_dir: Path, n, L, af):
    gp, gs, gc, gk, ga = (R._load(gen_dir, x) for x in
                          ("gen_phantom", "gen_smaps", "gen_coils", "gen_kspace", "gen_aliased_kspace"))
    m = gp.gen_shepp_logan_2d(N=n, A=1.0).astype(np.complex128)
    smaps = gs.gen_sensitivity_maps_2d(n, L, radius_factor=1.0, sigma_factor=0.8,
                                       phase_scale=1.0, normalize=True)
    coils = gc.compute_coil_images(m, smaps)
    k_alias = ga.undersample_kspace_ny(gk.compute_kspace_from_coils(coils), af)
    return m, smaps, R.fold(coils, af), R.to_hw_layout(k_alias)


def fmt_mask(v, n):
    bits = [str(i) for i in range(n) if (v >> i) & 1]
    return f"{v:4d} (etapas {','.join(bits) if bits else '-'})"


def study(chain: Chain, brute: bool, log):
    L1, L2 = chain.L1, chain.L2
    res = dict(lazy=[], brute1=[], brute2=[])

    # ---- pasada 1: picos en float sin escalar y mascaras lazy
    p1 = unscaled_stage_peaks((chain.qr + 1j * chain.qi) / FS, chain.n1)
    log(f"  pasada 1, pico sin escalar por etapa (dB re fondo de escala): "
        f"{' '.join(f'{20*np.log10(p):+.1f}' for p in p1)}")
    cand1 = {th: lazy_mask(p1, th) for th in (1.0, 0.5)}
    for th, m1 in cand1.items():
        log(f"  lazy(theta={th}) pasada 1: {fmt_mask(m1, L1)}")

    best = None
    for th in (1.0, 0.5):
        m1 = cand1[th]
        st1 = Stats()
        h = chain.pass1(m1, st1)
        # pasada 2: picos en float sobre la salida real de la pasada 1
        hc = (h[0] + 1j * h[1]) / FS
        p2 = unscaled_stage_peaks(np.swapaxes(hc, 1, 2), chain.n2)
        m2 = lazy_mask(p2, th)
        log(f"  pasada 2 tras lazy({th}), pico sin escalar por etapa (dB): "
            f"{' '.join(f'{20*np.log10(p):+.1f}' for p in p2)}  -> {fmt_mask(m2, L2)}")
        r = chain.evaluate(m1, m2, h)
        r.update(sat1=st1.total_sat, peak1=max(st1.peak.values()), theta=th)
        res["lazy"].append(r)
        if best is None or r["snr_img"] > best["snr_img"]:
            best = r

    if brute:
        t0 = time.time()
        s1c = best["S1"]
        for s1 in (s1c - 1, s1c, s1c + 1):
            for bits in itertools.combinations(range(L1), s1):
                m1 = sum(1 << b for b in bits)
                st = Stats()
                h = chain.pass1(m1, st)
                xq = (chain.qr + 1j * chain.qi) / FS
                href = R.idft_pass(xq, -1, s1)
                hn = ((h[0] + 1j * h[1]) / FS)[:, :, chain.ib1]
                res["brute1"].append(dict(m1=m1, S1=s1, sat1=st.total_sat, peak1=max(st.peak.values()),
                                          snr_h=snr_db(href, hn)))
        log(f"  barrido pasada 1: {len(res['brute1'])} mascaras en {time.time() - t0:.0f} s")
        ok1 = [r for r in res["brute1"] if r["sat1"] == 0]
        b1 = max(ok1, key=lambda r: r["snr_h"])
        h = chain.pass1(b1["m1"])
        t0 = time.time()
        s2c = best["S2"]
        for s2 in (s2c - 1, s2c, s2c + 1):
            for bits in itertools.combinations(range(L2), s2):
                m2 = sum(1 << b for b in bits)
                r = chain.evaluate(b1["m1"], m2, h)
                r.update(sat1=b1["sat1"], peak1=b1["peak1"])
                res["brute2"].append(r)
        log(f"  barrido pasada 2 (con m1 = {b1['m1']}): {len(res['brute2'])} mascaras en {time.time() - t0:.0f} s")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Estudio de mascaras de la FFT 2D (paso 4)")
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("--tw-dir", type=Path, default=HERE.parent / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles")
    ap.add_argument("-N", type=int, default=1024)
    ap.add_argument("-L", type=int, default=8)
    ap.add_argument("--af", type=int, default=2)
    ap.add_argument("--norm", choices=("comun", "bobina", "ambas"), default="ambas")
    ap.add_argument("--brute", action="store_true", help="barrido exhaustivo alrededor de lo mejor")
    ap.add_argument("--eval", nargs="*", default=["801,384"], metavar="M1,M2",
                    help="pares de mascaras a evaluar ademas de las lazy (default: la propuesta 801,384)")
    ap.add_argument("--json", type=Path, default=None, help="guardar todos los resultados")
    a = ap.parse_args()

    log = print
    m, smaps, y_true, k_hw = build_phantom(a.gen_dir, a.N, a.L, a.af)
    log(f"fantoma Shepp-Logan N={a.N}, L={a.L}, Af={a.af}; k-space por bobina {k_hw.shape[1:]} "
        f"(lineas x readout); nucleo 20/18/16")
    out = {}
    for norm in (("comun", "bobina") if a.norm == "ambas" else (a.norm,)):
        log(f"\n=== normalizacion {norm} ===")
        ch = Chain(k_hw, smaps, m, a.af, a.tw_dir, norm)
        ch.set_truth(y_true)
        res = study(ch, a.brute, log)
        res["ceilings"] = {}
        log("  resultados (SNR en dB; margen = 20 log10(1/pico)):")
        for r in res["lazy"]:
            c = ch.ceilings(r["S1"], r["S2"])
            res["ceilings"][f"{r['S1']},{r['S2']}"] = c
            log(f"    lazy({r['theta']}): m1 = {fmt_mask(r['m1'], ch.L1)}  m2 = {fmt_mask(r['m2'], ch.L2)}")
            log(f"      saturaciones {r['sat1']} / {r['sat2']};  margen {-20*np.log10(r['peak1']):.1f} / "
                f"{-20*np.log10(r['peak2']):.1f} dB")
            log(f"      SNR plegada {r['snr_alias']:.2f}  imagen {r['snr_img']:.2f}"
                f"   | techo (a) entrada {c['a_entrada']['snr_img']:.2f}"
                f"  (b) entrada + 16 b {c['b_entrada_y_16b']['snr_img']:.2f}")
        for txt in a.eval:
            m1, m2 = (int(v) for v in txt.split(","))
            r = ch.evaluate(m1, m2)
            log(f"    par m1 = {fmt_mask(m1, ch.L1)}  m2 = {fmt_mask(m2, ch.L2)}: saturaciones {r['sat1']} / {r['sat2']};"
                f"  margen {-20*np.log10(r['peak1']):.1f} / {-20*np.log10(r['peak2']):.1f} dB;  SNR imagen {r['snr_img']:.2f} dB")
            res.setdefault("eval", []).append(r)
        for ref in ((1023, 511), (784, 511)):
            r = ch.evaluate(*ref)
            log(f"    referencia m1={ref[0]}, m2={ref[1]}: saturaciones {r['sat1']} / {r['sat2']}, "
                f"SNR imagen {r['snr_img']:.2f} dB")
            res.setdefault("ref", []).append(r)
        if a.brute:
            ok = sorted([r for r in res["brute2"] if r["sat2"] == 0], key=lambda r: -r["snr_img"])[:5]
            log("    barrido: mejores 5 sin saturacion")
            for r in ok:
                log(f"      m1={fmt_mask(r['m1'], ch.L1)} m2={fmt_mask(r['m2'], ch.L2)}  "
                    f"SNR imagen {r['snr_img']:.2f}  margen {-20*np.log10(r['peak1']):.1f}/{-20*np.log10(r['peak2']):.1f} dB")
        out[norm] = res
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=float))
        log(f"\nresultados en {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

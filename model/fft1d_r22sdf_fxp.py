#!/usr/bin/env python3
"""
fft1d_r22sdf_fxp.py -- modelo bit-exacto y VECTORIZADO del nucleo fft1d_r22sdf.

Calcula, con aritmetica entera, exactamente lo mismo que el RTL (y que el
modelo C++), pero sobre muchas tramas a la vez: un array (..., N) procesa
todas las filas de una pasada de la FFT 2D en una llamada. Sirve para el
estudio de mascaras (paso 4), donde el modelo C++ (~0,3 s por trama de 1024)
seria demasiado lento.

No modela tiempos (latencia, valid, CMUL_PIPE): solo los valores, que no
dependen de CMUL_PIPE. La validacion es contra los vectores del vm:

    python3 model/fft1d_r22sdf_fxp.py --validate modules/fft1d_r22sdf

lee el JSON de regresion del modulo y compara, caso por caso, la salida del
modelo con build/<caso>/simulation/vectors/expected/out_ports (que a su vez es
igual bit a bit al RTL). Hay que haber corrido run_regression_sim antes.

ARITMETICA (copia de los .v)
----------------------------
  entrada      Q(NB_IN, NBF_IN) -> Q(NB, NBF): desplazamiento, exacto.
  INVERSE      swap real/imaginaria a la entrada y a la salida.
  mariposa     s = d +- x en NB+1 bits; con SCALE se baja 1 bit con redondeo
               half-up; en los dos casos satura a NB bits (cast.v).
  -j           (re, im) -> (im, -re), con -(-2^(NB-1)) saturado (btfly_bf2ii).
  cmul         producto exacto, un solo cast: baja NBF_TW bits con half-up y
               satura. Con direccion de twiddle 0 se puentea (W^0 = 1).
  salida       Q(NB, NBF) -> Q(NB_OUT, NBF_OUT) con half-up y saturacion.
  half-up      floor((x + 2^(k-1)) / 2^k), como cast.v (suma BIAS y >>>).
  ROUND_CONV   con 1, las mariposas que escalan y la salida redondean al par en
               los empates (cast.v con ROUND_CONV = 1): m + (r > h) + (r == h) lsb(m),
               con m = floor(x / 2^k), r = x - m 2^k y h = 2^(k-1). El cmul no cambia.

Orden: la entrada en orden natural; la salida en el orden del stream
(bit-reversed), igual que o_cplx_sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

I64 = np.int64


# ----------------------------------------------------------------- formatos
@dataclass(frozen=True)
class Cfg:
    N: int = 64
    NB: int = 16
    NBF: int = 15
    NB_TW: int | None = None
    NBF_TW: int | None = None
    NB_IN: int | None = None
    NBF_IN: int | None = None
    NB_OUT: int | None = None
    NBF_OUT: int | None = None
    SCALE_MASK: int = 0xFFFFFFFF
    INVERSE: int = 0
    ROUND_CONV: int = 0

    def fmt(self, nb, nbf):
        """Misma regla de defaults que el RTL: NBF_x = NB_x - (NB - NBF)."""
        nb = self.NB if nb is None else nb
        nbf = nb - (self.NB - self.NBF) if nbf is None else nbf
        return nb, nbf

    @property
    def tw(self):
        return self.fmt(self.NB_TW, self.NBF_TW)

    @property
    def inp(self):
        return self.fmt(self.NB_IN, self.NBF_IN)

    @property
    def out(self):
        return self.fmt(self.NB_OUT, self.NBF_OUT)

    @property
    def log2n(self):
        return self.N.bit_length() - 1

    def scales(self, stage: int) -> bool:
        return ((self.SCALE_MASK >> stage) & 1) == 1

    @classmethod
    def from_case(cls, case: dict) -> "Cfg":
        keys = [f.name for f in cls.__dataclass_fields__.values()]
        return cls(**{k: int(case[k]) for k in keys if k in case})


@dataclass
class Stats:
    """Por etapa radix-2 (en orden de tiempo) y para cada cmul y el cast de salida."""
    sat: dict = field(default_factory=dict)    # nombre -> cantidad de componentes saturadas
    peak: dict = field(default_factory=dict)   # nombre -> max |componente| / 2^(NB-1)

    def add(self, name, n_sat, peak):
        self.sat[name] = self.sat.get(name, 0) + int(n_sat)
        self.peak[name] = max(self.peak.get(name, 0.0), float(peak))

    @property
    def total_sat(self):
        return sum(self.sat.values())


# --------------------------------------------------------------- aritmetica
def _cast(x, drop: int, nb: int, stats: Stats | None = None, name: str = "", conv: bool = False):
    """cast.v con ROUND_MODE = 1: half-up (o, con conv, empates al par) al bajar 'drop'
    bits y saturacion a nb bits."""
    if drop < 0:
        raise ValueError("cast no puede agregar bits fraccionales")
    if drop > 0 and conv:
        q = x >> drop
        r = x - (q << drop)
        h = I64(1) << (drop - 1)
        x = q + ((r > h) | ((r == h) & ((q & 1) == 1)))
    elif drop > 0:
        x = (x + (I64(1) << (drop - 1))) >> drop     # >> de numpy en int64 = floor
    lo, hi = -(1 << (nb - 1)), (1 << (nb - 1)) - 1
    if stats is not None:
        stats.add(name, np.count_nonzero((x < lo) | (x > hi)),
                  np.abs(np.clip(x, lo, hi)).max(initial=0) / 2 ** (nb - 1))
    return np.clip(x, lo, hi)


def _bfly(ar, ai, cr, ci, scale: bool, nb: int, stats, name, conv: bool = False):
    """(a + c, a - c) con el cast de la mariposa (a = linea de retardo, c = stream)."""
    d = 1 if scale else 0
    return (_cast(ar + cr, d, nb, stats, name, conv), _cast(ai + ci, d, nb, stats, name, conv),
            _cast(ar - cr, d, nb, stats, name, conv), _cast(ai - ci, d, nb, stats, name, conv))


def load_twiddles(tw_dir: Path, n: int, nb_tw: int):
    """tw_re.mem / tw_im.mem (hex de NB_TW bits, complemento a 2) -> enteros con signo."""
    out = []
    for comp in ("re", "im"):
        vals = [int(t, 16) for t in (tw_dir / f"tw_{comp}.mem").read_text().split()][:n]
        if len(vals) != n:
            raise ValueError(f"{tw_dir}/tw_{comp}.mem: {len(vals)} valores, se esperaban {n}")
        v = np.array(vals, dtype=I64)
        out.append(np.where(v >= (1 << (nb_tw - 1)), v - (1 << nb_tw), v))
    return out[0], out[1]


def _tw_addr(log2n: int, log_m: int):
    """Direccion de twiddle por posicion local i del bloque de M (copia de la unidad)."""
    m = 1 << log_m
    i = np.arange(m)
    sel = 2 * ((i >> (log_m - 2)) & 1) + ((i >> (log_m - 1)) & 1)
    num = ((i % (m // 4)) << (log2n - log_m)) & ((1 << (log2n - 2)) - 1)
    return (num * sel) % (1 << log2n)


# ------------------------------------------------------------------- nucleo
def fft1d_r22sdf_fxp(x_re, x_im, cfg: Cfg, tw_re, tw_im, stats: Stats | None = None):
    """
    x_re, x_im : enteros (..., N) en Q(NB_IN, NBF_IN), orden natural.
    Devuelve (y_re, y_im) enteros (..., N) en Q(NB_OUT, NBF_OUT), orden del
    stream (bit-reversed), exactamente lo que entrega el RTL.
    """
    N, L = cfg.N, cfg.log2n
    if N != 1 << L or L < 2:
        raise ValueError("N tiene que ser potencia de 2 y >= 4")
    nb, nbf = cfg.NB, cfg.NBF
    nb_tw, nbf_tw = cfg.tw
    nb_in, nbf_in = cfg.inp
    nb_out, nbf_out = cfg.out
    conv = cfg.ROUND_CONV == 1
    sh_in = nbf - nbf_in
    if sh_in < 0 or (nb - nbf) < (nb_in - nbf_in) or nbf < nbf_out:
        raise ValueError("formatos incompatibles (mismas guardas que el RTL)")

    xr = np.asarray(x_re, dtype=I64)
    xi = np.asarray(x_im, dtype=I64)
    if cfg.INVERSE:
        xr, xi = xi, xr
    yr, yi = xr << sh_in, xi << sh_in
    shape = yr.shape
    mn, mx = -(1 << (nb - 1)), (1 << (nb - 1)) - 1

    stage = 0
    log_m = L
    while log_m >= 2:
        m = 1 << log_m
        r = yr.reshape(*shape[:-1], N // m, m)
        i_ = yi.reshape(*shape[:-1], N // m, m)
        # BF2II (primera): span M/2, -j en el ultimo cuarto
        h = m // 2
        sr, si, dr, di = _bfly(r[..., :h], i_[..., :h], r[..., h:], i_[..., h:],
                               cfg.scales(stage), nb, stats, f"etapa{stage}", conv)
        r = np.concatenate([sr, dr], axis=-1)
        i_ = np.concatenate([si, di], axis=-1)
        q = 3 * m // 4
        re_q, im_q = r[..., q:].copy(), i_[..., q:].copy()
        r[..., q:] = im_q
        i_[..., q:] = np.where(re_q == mn, mx, -re_q)
        stage += 1
        # BF2I (segunda): span M/4 dentro de cada mitad
        f = m // 4
        parts_r, parts_i = [], []
        for h0 in (0, m // 2):
            sr, si, dr, di = _bfly(r[..., h0:h0 + f], i_[..., h0:h0 + f],
                                   r[..., h0 + f:h0 + 2 * f], i_[..., h0 + f:h0 + 2 * f],
                                   cfg.scales(stage), nb, stats, f"etapa{stage}", conv)
            parts_r += [sr, dr]
            parts_i += [si, di]
        r = np.concatenate(parts_r, axis=-1)
        i_ = np.concatenate(parts_i, axis=-1)
        stage += 1
        # multiplicador (todas las unidades menos la de M = 4)
        if log_m != 2:
            addr = _tw_addr(L, log_m)
            wr, wi = tw_re[addr], tw_im[addr]
            pr = wr * r - wi * i_
            pi = wr * i_ + wi * r
            name = f"cmul{log_m}"
            cr = _cast(pr, nbf_tw, nb, stats, name)
            ci = _cast(pi, nbf_tw, nb, stats, name)
            byp = addr == 0
            r = np.where(byp, r, cr)
            i_ = np.where(byp, i_, ci)
        yr, yi = r.reshape(shape), i_.reshape(shape)
        log_m -= 2

    if log_m == 1:   # log2(N) impar: etapa radix-2 final, span 1, sin twiddle
        sr, si, dr, di = _bfly(yr[..., 0::2], yi[..., 0::2], yr[..., 1::2], yi[..., 1::2],
                               cfg.scales(stage), nb, stats, f"etapa{stage}", conv)
        yr = np.empty_like(yr)
        yi = np.empty_like(yi)
        yr[..., 0::2], yr[..., 1::2] = sr, dr
        yi[..., 0::2], yi[..., 1::2] = si, di

    our = _cast(yr, nbf - nbf_out, nb_out, stats, "salida", conv)
    oui = _cast(yi, nbf - nbf_out, nb_out, stats, "salida", conv)
    if cfg.INVERSE:
        our, oui = oui, our
    return our, oui


# --------------------------------------------------------------- validacion
def _read_dat(p: Path):
    return np.array([int(t) for t in p.read_text().split()], dtype=I64)


def validate(module_dir: Path, json_path: Path | None = None) -> int:
    module_dir = module_dir.resolve()
    name = module_dir.name
    json_path = json_path or module_dir / "testbench" / f"{name}_tb_regression.json"
    data = json.loads(json_path.read_text())
    defaults = {k: v for k, v in data.get("defaults", {}).items() if k.isupper()}
    build = module_dir / data.get("regression_dir", "build")
    ok = run = 0
    for case in data["cases"]:
        if not case.get("enabled", True):
            continue
        params = dict(defaults)
        params.update({k: v for k, v in case.items() if k.isupper()})
        cfg = Cfg.from_case(params)
        vec = build / case["name"] / "simulation" / "vectors"
        if not (vec / "expected" / "out_ports" / "o_re.dat").exists():
            print(f"  [SKIP] {case['name']}: no hay vectores (correr run_regression_sim)")
            continue
        vi = _read_dat(vec / "stimuli" / "in_ports" / "i_valid.dat").astype(bool)
        xr = _read_dat(vec / "stimuli" / "in_ports" / "i_re.dat")[vi][:cfg.N]
        xi = _read_dat(vec / "stimuli" / "in_ports" / "i_im.dat")[vi][:cfg.N]
        vo = _read_dat(vec / "expected" / "out_ports" / "o_valid.dat").astype(bool)
        er = _read_dat(vec / "expected" / "out_ports" / "o_re.dat")[vo][:cfg.N]
        ei = _read_dat(vec / "expected" / "out_ports" / "o_im.dat")[vo][:cfg.N]
        tw_dir = module_dir / str(params.get("TW_DIR", "twiddles"))
        twr, twi = load_twiddles(tw_dir, cfg.N, cfg.tw[0])
        st = Stats()
        yr, yi = fft1d_r22sdf_fxp(xr, xi, cfg, twr, twi, st)
        good = len(er) == cfg.N and np.array_equal(yr, er) and np.array_equal(yi, ei)
        run += 1
        ok += good
        nd = int(np.count_nonzero(yr != er) + np.count_nonzero(yi != ei)) if len(er) == cfg.N else -1
        print(f"  [{'OK ' if good else 'MAL'}] {case['name']:42s} N={cfg.N:4d}"
              f"  saturaciones={st.total_sat:5d}" + ("" if good else f"  distintas={nd}"))
    print(f"== {ok}/{run} casos bit a bit contra expected")
    return 0 if ok == run and run > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Modelo bit-exacto vectorizado de fft1d_r22sdf")
    ap.add_argument("--validate", type=Path, metavar="MODULE_DIR",
                    help="carpeta del modulo (p. ej. modules/fft1d_r22sdf)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    if a.validate:
        return validate(a.validate, a.json)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
gen_twiddles.py

Genera la ROM de twiddles bit-exacta para fft1d_r2.v en formato $readmemh.

  W_N^k = exp(-j*2*pi*k/N),  k = 0 .. N/2-1
  tw_re[k] = q(cos(-2*pi*k/N)),  tw_im[k] = q(sin(-2*pi*k/N))

donde q() reproduce EXACTAMENTE la asignacion a ap_fixed<NB, NB-NBF, AP_RND, AP_SAT>
del modelo C++:
  - AP_RND : round-half-up (ties -> +inf)  ->  floor(x*2^NBF + 0.5)
  - AP_SAT : saturacion a [-2^(NB-1), 2^(NB-1)-1]

Trampa importante: cos(0)=1.0 NO es representable en Q1.15 (max = 1-2^-15),
asi que satura a 0x7FFF. En cambio -1.0 (p.ej. sin(-pi/2)) SI es exacto -> 0x8000.
El .mem lleva esos valores ya saturados; el RTL solo los lee.

OJO: el ROM depende de N. Si cambia N (o NB/NBF), hay que regenerar el .mem
antes de simular/sintetizar. No es un simple override de parametro.

Uso:
  gen_twiddles.py --N 64 [--nb 16 --nbf 15] \
                  --out-re fft1d_r2_tw_re.mem --out-im fft1d_r2_tw_im.mem
"""
import argparse
import math


def q_round_sat(x: float, nb: int, nbf: int) -> int:
    """Cuantiza x a Q(nb-nbf . nbf) con round-half-up + saturacion. Devuelve el
    entero con signo (raw)."""
    qmax = (1 << (nb - 1)) - 1
    qmin = -(1 << (nb - 1))
    q = math.floor(x * (1 << nbf) + 0.5)   # AP_RND: ties -> +inf
    if q > qmax:
        q = qmax
    elif q < qmin:
        q = qmin
    return q


def to_hex(raw: int, nb: int) -> str:
    """Complemento a dos en nb bits, en hexadecimal (ancho fijo)."""
    mask = (1 << nb) - 1
    ndig = (nb + 3) // 4
    return f"{raw & mask:0{ndig}X}"


def gen(n: int, nb: int, nbf: int, out_re: str, out_im: str) -> None:
    if n & (n - 1) != 0 or n < 2:
        raise ValueError(f"N debe ser potencia de 2 >= 2 (N={n})")

    nh = n // 2
    re_lines = []
    im_lines = []
    sat_hits = 0

    for k in range(nh):
        angle = -2.0 * math.pi * k / n
        re = q_round_sat(math.cos(angle), nb, nbf)
        im = q_round_sat(math.sin(angle), nb, nbf)
        # contar saturaciones (para reportar)
        if re == (1 << (nb - 1)) - 1 and math.cos(angle) >= 1.0:
            sat_hits += 1
        re_lines.append(to_hex(re, nb))
        im_lines.append(to_hex(im, nb))

    with open(out_re, "w") as f:
        f.write("\n".join(re_lines) + "\n")
    with open(out_im, "w") as f:
        f.write("\n".join(im_lines) + "\n")

    print(f"[gen_twiddles] N={n} NB={nb} NBF={nbf}  ({nh} entradas)")
    print(f"[gen_twiddles] {out_re}")
    print(f"[gen_twiddles] {out_im}")
    print(f"[gen_twiddles] tw_re[0]={re_lines[0]} tw_im[0]={im_lines[0]} "
          f"(cos(0)=1.0 saturado)  saturaciones={sat_hits}")


def main() -> int:
    p = argparse.ArgumentParser(prog="gen_twiddles")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--nb", type=int, default=16)
    p.add_argument("--nbf", type=int, default=15)
    p.add_argument("--out-re", default="fft1d_r2_tw_re.mem")
    p.add_argument("--out-im", default="fft1d_r2_tw_im.mem")
    a = p.parse_args()
    gen(a.N, a.nb, a.nbf, a.out_re, a.out_im)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
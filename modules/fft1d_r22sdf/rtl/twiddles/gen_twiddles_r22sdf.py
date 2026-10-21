#!/usr/bin/env python3
"""
gen_twiddles_r22sdf.py -- genera tw_re.mem / tw_im.mem para fft1d_r22sdf.

    W_N^k = exp(-j*2*pi*k/N),  k = 0 .. N-1,  en Q(NB_TW, NBF_TW)

Cada linea es un valor de NB_TW bits en hexadecimal (complemento a 2), que es lo
que leen el $readmemh del RTL y el modelo C++.

Detalles:
  * Redondeo al mas cercano y saturacion al rango de Q(NB_TW, NBF_TW).
  * W^0 = 1 no es representable en Q1.x y satura a 1 - 2^-NBF_TW. No importa:
    el RTL puentea el multiplicador cuando la direccion de twiddle es 0.
  * Con N potencia de 2 no hay empates de redondeo (los unicos valores exactos
    son 0 y +-1), asi que el modo de desempate no cambia el resultado.

Uso:
    python3 gen_twiddles_r22sdf.py --n 1024 --nb 18 --out twiddles/n1024_tw18
"""
import argparse
import os

import numpy as np


def generate(n, nb, nbf, out):
    lo, hi = -(1 << (nb - 1)), (1 << (nb - 1)) - 1
    mask = (1 << nb) - 1
    hex_w = (nb + 3) // 4
    ang = -2.0 * np.pi * np.arange(n) / n

    os.makedirs(out, exist_ok=True)
    for name, val in (("re", np.cos(ang)), ("im", np.sin(ang))):
        q = np.clip(np.round(val * 2.0 ** nbf), lo, hi).astype(np.int64)
        with open(os.path.join(out, f"tw_{name}.mem"), "w") as f:
            f.write("\n".join(f"{int(x) & mask:0{hex_w}x}" for x in q) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",   type=int, required=True, help="tamano de la FFT (N)")
    ap.add_argument("--nb",  type=int, required=True, help="bits del twiddle (NB_TW)")
    ap.add_argument("--nbf", type=int, default=None,  help="bits fraccionales (NBF_TW); por defecto NB_TW-1")
    ap.add_argument("--out", required=True,           help="directorio de salida")
    a = ap.parse_args()
    generate(a.n, a.nb, a.nbf if a.nbf is not None else a.nb - 1, a.out)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Genera la tabla de twiddles para fft1d_r22sdf.

A diferencia del R2SDF (un par de archivos por etapa), el R2^2SDF usa UNA SOLA
tabla de N entradas -- W_N^k para k = 0..N-1 -- que todas las unidades comparten
y direccionan distinto segun su resolucion M (tw_addr = tw_num * tw_sel).

Formato: hex complemento a dos, Q(NB,NBF), una entrada por linea.
W^0 = 1.0 satura al maximo representable (0x7fff en Q1.15).
"""
import numpy as np, os, sys

def to_hex_q(x, NB, NBF):
    q = int(np.round(x * (1 << NBF)))
    lo, hi = -(1 << (NB-1)), (1 << (NB-1)) - 1
    q = max(lo, min(hi, q))
    if q < 0: q += (1 << NB)
    return f"{q:0{(NB+3)//4}x}"

def gen(N, NB=16, NBF=15, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    re_v, im_v = [], []
    for k in range(N):
        w = np.exp(-2j*np.pi*k/N)
        re_v.append(to_hex_q(w.real, NB, NBF))
        im_v.append(to_hex_q(w.imag, NB, NBF))
    with open(os.path.join(outdir,"tw_re.mem"),"w") as f: f.write("\n".join(re_v)+"\n")
    with open(os.path.join(outdir,"tw_im.mem"),"w") as f: f.write("\n".join(im_v)+"\n")
    print(f"  {N} twiddles -> {outdir}/tw_re.mem, {outdir}/tw_im.mem")

if __name__ == "__main__":
    N   = int(sys.argv[1]) if len(sys.argv)>1 else 64
    NB  = int(sys.argv[2]) if len(sys.argv)>2 else 16
    NBF = int(sys.argv[3]) if len(sys.argv)>3 else 15
    out = sys.argv[4] if len(sys.argv)>4 else f"twiddles/n{N}"
    print(f"Twiddles R2^2SDF N={N} Q({NB},{NBF}):")
    gen(N, NB, NBF, out)
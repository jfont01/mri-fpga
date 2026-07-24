#!/usr/bin/env python3
"""Genera las ROM de twiddles del R2SDF: una por etapa.

La etapa k usa los exponentes p*2^k con p = 0..L_k-1, o sea que su ROM tiene
L_k entradas y se indexa DIRECTAMENTE con el puntero de la linea de retardo.
Las dos ultimas etapas (L_k <= 2) tienen solo twiddles triviales y no llevan ROM.
"""
import math, os, sys

N   = int(sys.argv[1])
NB  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NBF = int(sys.argv[3]) if len(sys.argv) > 3 else 15
OUT = sys.argv[4] if len(sys.argv) > 4 else "twiddles"

nu = int(math.log2(N))
QMAX = (1 << (NB-1)) - 1
QMIN = -(1 << (NB-1))
MASK = (1 << NB) - 1

def q(x):
    v = math.floor(x * (1 << NBF) + 0.5)
    return QMAX if v > QMAX else (QMIN if v < QMIN else v)

os.makedirs(OUT, exist_ok=True)
for k in range(nu):
    L = 1 << (nu - 1 - k)
    if L <= 2:
        continue                       # twiddles triviales, sin ROM
    re, im = [], []
    for p in range(L):
        ang = -2.0 * math.pi * (p * (1 << k)) / N
        re.append(q(math.cos(ang)))
        im.append(q(math.sin(ang)))
    for nm, vals in (("re", re), ("im", im)):
        with open(f"{OUT}/tw_s{k}_{nm}.mem", "w") as f:
            f.write("\n".join(f"{v & MASK:0{(NB+3)//4}X}" for v in vals) + "\n")
    print(f"  etapa {k}: {L:3d} entradas -> {OUT}/tw_s{k}_{{re,im}}.mem")
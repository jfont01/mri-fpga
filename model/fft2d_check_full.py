#!/usr/bin/env python3
"""
fft2d_check_full.py -- chequeo a tamano completo del modelo C++ de la fft2d
(paso 5b): el modelo ciclo a ciclo contra la cadena bit-exacta de Python, con
el fantoma del paso 4.

  1. Fantoma Shepp-Logan, mapas de sensibilidad y k-space plegado (los mismos
     generadores que fft2d_mask_study.py), normalizado por bobina y cuantizado
     a Q1.15.
  2. Compila modules/fft2d/testbench/fft2d_dump.cpp con -O2 (el Makefile usa
     -O0 y el tamano completo tarda ~17 veces mas) y lo corre desde
     modules/fft2d, con las mismas mascaras.
  3. Compara bit a bit la salida contra fft1d_r22sdf_fxp (pasada 1 + pasada 2),
     chequea los tiempos (primer o_valid = formula, sin huecos, un o_last por
     bobina) y calcula la SNR de la imagen desplegada con la salida del C++.
  4. Deja todo en modules/fft2d/build/full_check/ (kspace.bin, out.bin y
     case.json), para plotearlo con run_plot_fft2d.py --full-check.

Uso (desde la raiz del repo, con el entorno cargado):
    python3 model/fft2d_check_full.py                 # 8 bobinas, 801 / 384
    python3 model/fft2d_check_full.py -L 2 --masks 1023,511

Sale con codigo 0 si todo coincide.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fft2d_ref as R                                              # noqa: E402
from fft2d_mask_study import FS, Chain, build_phantom, popcount, snr_db   # noqa: E402


def compile_dump(repo: Path, exe: Path) -> None:
    rtlsim = Path(os.environ.get("RTLSIM_ROOT", repo / "rtlsim"))
    src = repo / "modules" / "fft2d" / "testbench" / "fft2d_dump.cpp"
    # -Wno-(maybe-)uninitialized: con -O2 los headers ap_private de Vitis HLS
    # tiran decenas de falsos positivos; con -O0 (el Makefile) no aparece ninguno.
    cmd = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
           "-Wno-uninitialized", "-Wno-maybe-uninitialized",
           "-isystem", str(rtlsim / "include"), f"-I{rtlsim}",
           f"-I{repo / 'modules' / 'fft2d' / 'cpp'}", str(src), "-o", str(exe)]
    exe.parent.mkdir(parents=True, exist_ok=True)
    print("compilando:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    ap.add_argument("--gen-dir", type=Path, default=HERE / "gen")
    ap.add_argument("-L", type=int, default=8, help="bobinas")
    ap.add_argument("--af", type=int, default=2)
    ap.add_argument("--masks", default="801,384", metavar="M1,M2")
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()

    m1, m2 = (int(v) for v in a.masks.split(","))
    repo = a.repo.resolve()
    mod = repo / "modules" / "fft2d"
    work = mod / "build" / "full_check"
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "fft2d_dump"
    tw_dir = repo / "modules" / "fft1d_r22sdf" / "rtl" / "twiddles"

    # 1. fantoma y k-space en Q1.15, normalizado por bobina
    n = 1024
    m, smaps, y_true, k_hw = build_phantom(a.gen_dir, n, a.L, a.af)
    ch = Chain(k_hw, smaps, m, a.af, tw_dir, "bobina")
    ch.set_truth(y_true)
    L, n2, n1 = k_hw.shape
    print(f"k-space: {L} bobinas x {n2} filas x {n1} (readout), Q1.15")

    kin = np.stack([ch.qr, ch.qi], axis=-1).astype("<i4")         # [l, k', r, 2]
    (work / "kspace.bin").write_bytes(kin.tobytes())

    # 2. modelo C++ ciclo a ciclo
    if not a.skip_compile:
        compile_dump(repo, exe)
    t0 = time.perf_counter()
    r = subprocess.run([str(exe), "--in", str(work / "kspace.bin"), "--out", str(work / "out.bin"),
                        "--coils", str(L), "--mask1", str(m1), "--mask2", str(m2)],
                       cwd=mod, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    info = dict(line.split(" = ") for line in r.stdout.strip().splitlines())
    info = {k: int(v) for k, v in info.items()}
    print(f"modelo C++: {dt:.1f} s; " + ", ".join(f"{k} {v}" for k, v in info.items()))

    out = np.frombuffer((work / "out.bin").read_bytes(), dtype="<i4").reshape(L, n1, n2, 2)

    # 3a. cadena de Python (bit-exacta), mismo orden: [l, j1, j2]
    t0 = time.perf_counter()
    hr, hi = ch.pass1(m1)
    ar, ai = ch.pass2(hr, hi, m2)
    print(f"cadena Python: {time.perf_counter() - t0:.1f} s")

    diff = (out[..., 0] != ar) | (out[..., 1] != ai)
    n_diff = int(diff.sum())
    print(f"bit a bit: {n_diff} muestras distintas de {diff.size}")

    # 3b. tiempos
    t = n1 * n2
    checks = {
        "primer o_valid == formula": info["first_valid"] == info["latency_formula"],
        "muestras == L * T": info["samples"] == L * t,
        "salida sin huecos": info["output_gaps"] == 0,
        "un o_last por bobina": info["o_last"] == L,
        "ultima muestra == formula + L*T - 1": info["last_valid"] == info["latency_formula"] + L * t - 1,
    }
    for k, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {k}")

    # 3c. SNR de la imagen con la salida del C++
    s1, s2 = popcount(m1, ch.L1), popcount(m2, ch.L2)
    a_hw = (out[..., 0] + 1j * out[..., 1]) / FS
    a_nat = a_hw[:, ch.ib1][:, :, ch.ib2]
    y = ch.phys(a_nat, s1, s2)
    mh = R.unfold(y, ch.U, a.af)
    print(f"S1 = {s1}, S2 = {s2}; SNR plegada {snr_db(ch.y_true, y):.2f} dB, imagen {snr_db(ch.m, mh):.2f} dB")
    print(f"tiempo de trama a 200 MHz: {(info['last_valid'] + 1) / 200e6 * 1e3:.2f} ms")

    # para run_plot_fft2d.py --full-check
    (work / "case.json").write_text(json.dumps({
        "kind": "phantom", "N": n, "L": L, "af": a.af, "norm": "bobina",
        "N1": n1, "N2": n2, "mask1": m1, "mask2": m2, "o_last": info["o_last"],
        "gains": [float(v) for v in ch.g],
    }, indent=2) + "\n")

    ok = n_diff == 0 and all(checks.values())
    print("== OK" if ok else "== FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

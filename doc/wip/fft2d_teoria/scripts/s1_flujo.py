#!/usr/bin/env python3
"""
s1_flujo.py -- flujo temporal de la FFT 2D con el corner-turn IN SITU (un carril).

Modelo de tiempos (enteros, en ciclos de 200 MHz), sin datos:

  * Pasada 1: la fila r de la bobina j entra al nucleo 1 en a[j, r] .. a[j, r] + N1 - 1.
    Sale en a[j, r] + L1 + slot (slot = 0..N1-1, orden bit-reversed), contigua.
    (L1 = 1040, L2 = 527: formulas del RTL; core_timing.cpp verifica con el modelo
    C++ que la latencia es constante por trama aunque haya huecos entre tramas.)
  * Escritura: el indice t = r*N1 + slot de la bobina j se escribe ("se emite" hacia
    la memoria) en w[j, t] = a[j, r] + L1 + slot + DW.
  * Lector: la bobina j se lee en el orden de columnas s = c*N2 + k (c = slot, k = fila),
    una por ciclo, SIN huecos, desde R_j:  lee en R_j + s el dato t(s) = k*N1 + c.
    Arranque: R_j = max(trig_j + 1, R_{j-1} + T), con trig_j = w[j, (N2-1)*N1 + K0]
    (la muestra K0 de la ULTIMA fila). Adelanto del lector sobre el escritor de la
    bobina j + 1 en regimen continuo: m = N1 - 1 - K0.
  * In situ: la bobina j + 1 escribe su indice s en la misma direccion fisica de la que
    la bobina j lee su posicion s.

Condiciones (por bobina j):
  RAW  min_s (R_j + s) - w[j, t(s)]   >= 1  (se lee algo ya escrito)
  WAR  min_s w[j+1, s] - (R_j + s)    >= 1  (se pisa algo ya leido; con 0 depende de
                                              que el puerto A lea antes que el B, UG573)
Ademas: tiempo por trama, llenado, latencia y vaciado; y el solapamiento con unfold
(doble o triple buffer de A en la DDR).

Uso: python3 s1_flujo.py   (sale con 0 si todos los chequeos dan lo esperado)
"""
from __future__ import annotations

import sys

import numpy as np

F_CLK = 200e6
N1, N2 = 1024, 512
T = N1 * N2
L1, L2 = 1040, 527           # latencias del RTL (doc/03 3.1; core_timing.cpp)
DW, RL = 1, 4                # emision de la escritura y latencia de lectura (supuestas)
LCOILS = 8
FAILS: list[str] = []


def check(tag: str, cond: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if cond else 'MAL'}] {tag} {extra}")
    if not cond:
        FAILS.append(tag)


def ms(c: float) -> str:
    return f"{c / F_CLK * 1e3:.3f} ms"


# indices de lectura: s = c*N2 + k  ->  t(s) = k*N1 + c
S = np.arange(T, dtype=np.int64)
C_OF_S, K_OF_S = S // N2, S % N2
T_OF_S = K_OF_S * N1 + C_OF_S
SLOT_OF_T, ROW_OF_T = S % N1, S // N1          # (S usado como t)


def write_times(a_rows: np.ndarray) -> np.ndarray:
    """a_rows[r] -> w[t] para una bobina."""
    return a_rows[ROW_OF_T] + L1 + SLOT_OF_T + DW


def simulate(a: np.ndarray, k0: int):
    """a[j, r]: arranque de cada fila de entrada. Devuelve dict con R, raw, war, ..."""
    n_coils = a.shape[0]
    w = [write_times(a[j]) for j in range(n_coils)]
    R = np.zeros(n_coils, dtype=np.int64)
    raw = np.zeros(n_coils, dtype=np.int64)
    war = np.full(n_coils, np.iinfo(np.int64).max, dtype=np.int64)
    for j in range(n_coils):
        trig = w[j][(N2 - 1) * N1 + k0]
        R[j] = trig + 1 if j == 0 else max(trig + 1, R[j - 1] + T)
        raw[j] = (R[j] + S - w[j][T_OF_S]).min()
        if j + 1 < n_coils:
            war[j] = (w[j + 1] - (R[j] + S)).min()
    first_out = R + RL + L2                    # primer o_valid de la pasada 2 por bobina
    return dict(R=R, raw=raw, war=war, first_out=first_out, w=w)


def rows_continuous(n_coils: int, t0: int = 0) -> np.ndarray:
    return t0 + (np.arange(n_coils * N2, dtype=np.int64) * N1).reshape(n_coils, N2)


def rows_with_gaps(n_coils: int, rng, max_gap: int, frame_pause: int = 0) -> np.ndarray:
    g = rng.integers(0, max_gap + 1, size=n_coils * N2).astype(np.int64)
    g[0] = 0
    if frame_pause:
        g[np.arange(LCOILS, n_coils, LCOILS) * N2] += frame_pause
    starts = np.cumsum(g + N1) - N1
    return starts.reshape(n_coils, N2)


def main() -> int:
    print(f"N1 = {N1}, N2 = {N2}, T = {T} ciclos = {ms(T)} por bobina; L1 = {L1}, L2 = {L2}, "
          f"DW = {DW}, RL = {RL}")

    # ---------------------------------------------------------------- 1. continuo
    print("\n1) Entrada continua, 3 tramas (24 bobinas encadenadas), K0 = 0 (adelanto m = 1023)")
    nc = 3 * LCOILS
    r = simulate(rows_continuous(nc), k0=0)
    check("RAW >= 1 en todas las bobinas", r["raw"].min() >= 1, f"(min {r['raw'].min()})")
    check("WAR >= 1 en todas las bobinas", r["war"][:-1].min() >= 1, f"(min {r['war'][:-1].min()} = adelanto)")
    dR = np.diff(r["R"])
    check("lector sin huecos entre bobinas (R_j+1 - R_j == T)", bool(np.all(dR == T)))
    lat = int(r["first_out"][0])
    lat_pp = L1 + T + 2 + L2
    print(f"     primera salida (bobina 0): ciclo {lat} = {ms(lat)}; el modelo ping-pong da "
          f"L1 + T + 2 + L2 = {lat_pp} ({lat_pp - lat} ciclos mas)")
    check("latencia in situ = L1 + DW + (N2-1)*N1 + K0 + 1 + RL + L2",
          lat == L1 + DW + (N2 - 1) * N1 + 0 + 1 + RL + L2)
    per_frame = r["first_out"][2 * LCOILS] - r["first_out"][LCOILS]
    check("periodo de trama en regimen = 8 T", per_frame == LCOILS * T, f"({per_frame} = {ms(per_frame)})")
    last_out_f0 = r["first_out"][LCOILS - 1] + T - 1
    print(f"     trama 0 (arranque): de la primera entrada a la ultima salida {last_out_f0 + 1} ciclos "
          f"= {ms(last_out_f0 + 1)} = 9 T + {last_out_f0 + 1 - 9 * T}")
    print(f"     tramas siguientes: una cada 8 T = {ms(8 * T)} ({F_CLK / (8 * T):.2f} tramas/s de FFT 2D)")

    # ------------------------------------------------- 2. sin adelanto (K0 = N1 - 1)
    print("\n2) K0 = N1 - 1: el lector arranca justo despues de la ultima escritura (m = 0)")
    r0 = simulate(rows_continuous(4), k0=N1 - 1)
    check("RAW >= 1", r0["raw"].min() >= 1)
    check("WAR == 0: lectura y escritura en la misma direccion y el mismo ciclo",
          r0["war"][:-1].min() == 0, "(solo vale si el puerto A lee antes que el B escriba)")

    # ------------------------------------------- 3. huecos entre filas y pausas
    print("\n3) Huecos al azar entre filas (0..64 ciclos) y pausa de 1 ms entre tramas, K0 = 0")
    rng = np.random.default_rng(7)
    ag = rows_with_gaps(nc, rng, 64, frame_pause=200_000)
    rg = simulate(ag, k0=0)
    check("RAW >= 1", rg["raw"].min() >= 1, f"(min {rg['raw'].min()})")
    check("WAR >= 1", rg["war"][:-1].min() >= 1, f"(min {rg['war'][:-1].min()})")
    gaps = np.diff(rg["R"]) - T
    print(f"     huecos del lector entre bobinas: min {gaps.min()}, max {gaps.max()} ciclos "
          f"(la pasada 2 los acepta: son entre tramas de 512)")
    check("el lector nunca se superpone (R_j+1 >= R_j + T)", gaps.min() >= 0)

    # ------------------------------------- 4. disparo antes de la ultima fila
    print("\n4) Disparo en la PENULTIMA fila: por que el disparo va en la ultima")
    # con entrada continua, disparar en la muestra k de la penultima fila da RAW = k - 512;
    # un hueco g antes de la ultima fila lo baja a k - 512 - g.
    k = 1000
    a_ok = rows_continuous(1)
    a_gap = a_ok.copy()
    a_gap[0, N2 - 1] += 600                 # hueco de 600 ciclos antes de la ultima fila
    for tag, rows in (("sin hueco", a_ok), ("hueco de 600 antes de la ultima fila", a_gap)):
        w0 = write_times(rows[0])
        Rb = w0[(N2 - 2) * N1 + k] + 1
        raw_b = int((Rb + S - w0[T_OF_S]).min())
        print(f"     disparo en la muestra {k} de la penultima fila, {tag}: RAW min = {raw_b}")
        if tag == "sin hueco":
            check("sin hueco RAW = k - 512", raw_b == k - 512)
        else:
            check("con el hueco el lector lee antes de que se escriba (RAW < 1)", raw_b < 1)

    # ------------------------------------------------------- 5. ventana valida
    print("\n5) Ventana del arranque del lector con entrada continua")
    w0 = write_times(rows_continuous(1)[0])
    r_min = int((w0[T_OF_S] - S).max()) + 1         # RAW >= 1
    r_max = int(w0[-1] + 1 + 0)                      # escritor j+1 arranca en w0[-1] + 1 si es continuo
    print(f"     R_j - w[j, 0] en [{r_min - w0[0]}, {r_max - 1 - w0[0]}] para RAW y WAR >= 1: "
          f"{r_max - r_min} ciclos de ventana (N1 + N2 - 2 = {N1 + N2 - 2})")
    check("ventana = N1 + N2 - 2 ciclos", r_max - r_min == N1 + N2 - 2)

    # --------------------------------------- 6. solapamiento con unfold (tramas)
    print("\n6) Solapamiento FFT (trama n + 1) con unfold (trama n); A_l en la DDR")
    # FFT continua: la bobina l de la trama n sale de first_out[8n + l] a + T - 1.
    t_unfold = LCOILS * T                 # 8 ciclos por posicion, 1 muestra de bobina por ciclo
    print(f"     unfold, un carril: {T} posiciones x {LCOILS} bobinas = {t_unfold} ciclos = {ms(t_unfold)}")
    print(f"     FFT 2D en regimen: 8 T = {ms(8 * T)}  ->  periodo = max = {ms(max(8 * T, t_unfold))}")
    # bloques: B posiciones (2 columnas) por bloque; el escritor de A escribe la columna c
    # de la bobina l en first_out + c*N2 .. ; unfold lee el bloque b cuando le toca.
    B = 2 * N2
    n_blocks = T // B
    delta = 2000                          # latencia DDR + control antes de arrancar unfold (supuesta)
    for nbuf in (2, 3):
        for sigma in (1.0, 1.10):
            # tramas: la entrada de la trama f arranca cuando (a) termina la entrada de la f-1 y
            # (b) termino el unfold de la trama f - nbuf (regla simple de sense_ctrl)
            nf = 8
            in0 = np.zeros(nf, dtype=np.int64)
            u0 = np.zeros(nf, dtype=np.int64)
            u1 = np.zeros(nf, dtype=np.int64)
            ok_blocks = True
            for f in range(nf):
                t_in = 0 if f == 0 else in0[f - 1] + LCOILS * T
                if f >= nbuf:
                    t_in = max(t_in, u1[f - nbuf])
                in0[f] = t_in
                fo = in0[f] + (L1 + DW + (N2 - 1) * N1 + 1 + RL + L2)    # primera salida de la trama
                end_fft = fo + LCOILS * T                                # ultima salida (+1)
                u0[f] = max(end_fft + delta, u1[f - 1] if f else 0)
                u1[f] = u0[f] + int(round(t_unfold * sigma))
                # chequeo por bloques contra la trama f - nbuf (mismo buffer): la bobina 0 de la
                # trama f escribe el bloque b en fo + b*B; unfold(f - nbuf) lo leyo hasta
                # u0 + (b+1)*B*8*sigma
                if f >= nbuf:
                    b = np.arange(n_blocks)
                    wr = fo + b * B
                    rd_end = u0[f - nbuf] + (b + 1) * B * LCOILS * sigma
                    ok_blocks &= bool(np.all(wr > rd_end))
            per = np.diff(u1)[2:].mean()
            print(f"     {nbuf} buffers, unfold x{sigma:.2f}: periodo {ms(per)} "
                  f"({F_CLK / per:.1f} tramas/s); ningun bloque pisado: {ok_blocks}")
            if nbuf == 3:
                check(f"3 buffers, x{sigma:.2f}: sin pisar bloques", ok_blocks)
    print("     (con 2 buffers y la regla simple, la entrada de la trama f espera el fin del unfold")
    print("      de la f - 2 y el periodo crece ~1 ms; con 3 buffers no espera)")

    # ----------------------------------------------------------------- 7. serie
    print("\n7) Sin solapar (FFT y unfold en serie, una trama por vez)")
    t_ser = (L1 + DW + (N2 - 1) * N1 + 1 + RL + L2) + LCOILS * T + delta + t_unfold
    print(f"     {ms(t_ser)} por trama ({F_CLK / t_ser:.1f} tramas/s): tambien entra en 50 ms")

    print("\n== OK" if not FAILS else f"\n== FALLA: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

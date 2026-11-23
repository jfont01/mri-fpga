#!/usr/bin/env python3
"""
s5_presupuesto.py -- numeros del presupuesto (tiempos, trafico por puerto HP, memoria en
la DDR, BRAM, URAM, DSP) y de las alternativas al corner-turn en URAM. Solo cuentas.

Fuentes de los recursos: DS987 v1.6, tabla 5 (117 120 LUT, 234 240 FF, 144 RAMB36,
64 URAM288, 1248 DSP, 3,5 Mb de RAM distribuida). DDR: H30 (2133 MT/s x 8 B = 17,07 GB/s,
muy probable) y DS987 (2400 -> 19,2). Puerto HP de 128 b a 200 MHz: 3,2 GB/s teoricos.
Nucleos FFT medidos (doc/05 5.6, post-route): 2985 LUT / 1074 FF / 16 DSP / 0 BRAM (1024,
801) y 2415 / 997 / 16 / 0 (512, 384).
"""
from __future__ import annotations

F = 200e6
N1, N2, L, AF = 1024, 512, 8, 2
T = N1 * N2
B4 = 4                                   # bytes por muestra compleja (16 + 16)
MiB = 2 ** 20
KiB = 2 ** 10
HP = 16 * F                              # 128 b x 200 MHz
DDR = 2133.33e6 * 8
RAMB36_B = 4096                          # bytes de datos de un RAMB36 con palabras de 32 b
URAM_B = 4096 * 8                        # 64 bits utiles por palabra (2 muestras)


def gbs(x: float) -> str:
    return f"{x / 1e9:.2f} GB/s"


def main() -> None:
    print("== Tiempos (un carril por bloque)")
    t_coil = T / F
    print(f"  bobina: {T} ciclos = {t_coil * 1e3:.3f} ms; trama en regimen 8T = {8 * t_coil * 1e3:.2f} ms;"
          f" arranque 9T = {9 * t_coil * 1e3:.2f} ms")
    print(f"  unfold, un carril: {L * T} ciclos = {L * T / F * 1e3:.2f} ms")
    fps = F / (8 * T)
    print(f"  solapado: {fps:.1f} tramas/s maximo; a 20 fps queda {50 - 8 * t_coil * 1e3:.1f} ms de margen por trama")

    print("\n== Trafico por puerto HP con las fases solapadas (tasas pico de cada stream)")
    k_rd = F * B4                         # k-space: 1 muestra por ciclo
    a_wr = F * B4                         # A_l de la pasada 2
    a_rd = F * B4                         # A_l hacia unfold: 1 muestra (una bobina) por ciclo
    u_rd = F * B4 * AF                    # U'': 2 palabras por ciclo
    img = F / L * AF * 2                  # 2 pixeles u16 cada 8 ciclos
    ports = {
        "HP0 (Port3) lecturas: k-space + A_l": k_rd + a_rd,
        "HP1 (Port4) lecturas: U''": u_rd,
        "HP3 (Port5) escrituras: A_l + imagen": a_wr + img,
    }
    tot = 0.0
    for k, v in ports.items():
        tot += v
        print(f"  {k:40s} {gbs(v)}  ({100 * v / HP:4.1f} % de {gbs(HP)})")
    print(f"  total {gbs(tot)} = {100 * tot / DDR:.1f} % de {gbs(DDR)} (2133) / {100 * tot / 19.2e9:.1f} % de 19,2")
    per_frame = (16 * MiB) * 3 + 32 * MiB + 2 * MiB
    print(f"  por trama: k-space 16 MiB + A_l 16 MiB escrita + 16 MiB leida + U'' 32 MiB + imagen 2 MiB "
          f"= {per_frame / MiB:.0f} MiB; a 20 fps = {per_frame * 20 / 1e9:.2f} GB/s promedio")

    print("\n== Memoria en la DDR")
    reg = [("k-space (anillo de 3 tramas)", 3 * 16 * MiB), ("A_l (3 buffers)", 3 * 16 * MiB),
           ("U'' (sesion)", 32 * MiB), ("imagen (2 buffers)", 2 * 2 * MiB)]
    for k, v in reg:
        print(f"  {k:32s} {v / MiB:6.0f} MiB")
    print(f"  {'total':32s} {sum(v for _, v in reg) / MiB:6.0f} MiB de 4096 (DDR del SoM)")

    print("\n== BRAM (todo lo que no es el corner-turn), RAMB36 de 1K x 36")
    bram = [
        ("nucleos FFT (lineas de retardo en LUTRAM, ROM en LUT; medido)", 0),
        ("FIFO de entrada del k-space (2 filas: 2048 x 32 b)", 2),
        ("FIFO de salida de la fft2d hacia S2MM (4096 x 32 b)", 4),
        ("coil reorder, B = 1024 (2 x 8 x 1024 x 32 b)", 16),
        ("FIFO de U'' (1024 x 64 b)", 2),
        ("escritura de la imagen (2 x 2 KiB)", 1),
        ("FIFO de salida de la imagen", 1),
        ("DMA / DataMover (estimado, sin verificar)", 8),
    ]
    for k, v in bram:
        print(f"  {k:62s} {v:3d}")
    tb = sum(v for _, v in bram)
    print(f"  {'total':62s} {tb:3d} de 144 ({100 * tb / 144:.0f} %)")

    print("\n== URAM y DSP")
    print(f"  corner-turn: 2^19 muestras x 32 b = {T * B4 / MiB:.0f} MiB = {T * B4 // URAM_B} URAM288 "
          f"(2 muestras por palabra de 72 b) = 100 %")
    print(f"  con 1 muestra por palabra: {T // 4096} URAM (no entra)")
    dsp = 16 + 16 + 8
    print(f"  DSP: 2 nucleos x 16 + unfold 8 + cordic 0 = {dsp} de 1248 ({100 * dsp / 1248:.1f} %)")
    print(f"  LUTRAM de los nucleos: (1023 + 511) x 40 b = {(1023 + 511) * 40 / 1e3:.0f} Kb de 3500")

    print("\n== Alternativas al corner-turn en URAM")
    # (c) por bloques en la DDR, un carril: la pasada 1 escribe tiles R x C, la 2 los lee
    extra = 2 * F * B4
    print(f"  DDR por bloques: +{gbs(extra)} en la fase FFT (+{gbs(F * B4)} escritura, +{gbs(F * B4)} lectura); "
          f"total {gbs(tot + extra)} = {100 * (tot + extra) / DDR:.0f} % de 17,07")
    for R, C in ((16, 64), (32, 128), (64, 64)):
        burst = R * C * B4
        buf = 2 * R * N1 * B4 + 2 * C * N2 * B4
        eff = burst / (burst + 3 * KiB)
        print(f"    tile {R} x {C}: rafaga {burst // KiB} KiB (eficiencia con N1/2 = 3 KiB: {100 * eff:.0f} %), "
              f"buffers 2*{R} filas + 2*{C} columnas = {buf // KiB} KiB = {buf / URAM_B:.0f} URAM o {buf / RAMB36_B:.0f} RAMB36")
    # tile_transpose: tiles cuadrados TILE x TILE con ping-pong, rafaga = una fila del tile
    for tile in (64, 128):
        burst = tile * B4
        print(f"  tile_transpose TILE = {tile}: rafaga {burst} B -> {100 * burst / (burst + 3 * KiB):.0f} % "
              f"(N1/2 = 3 KiB); buffer 2 x {tile}^2 x 4 B = {2 * tile * tile * B4 // KiB} KiB")
    # bancos por region de reloj
    print("  4 bancos de 16 URAM (uno por region de reloj): la cascada no cruza regiones; mux 4:1 de 64 b")
    print("    a la salida (~64 LUT + 64 FF) y decodificacion de 2 bits para EN/BWE; misma capacidad.")
    # ping-pong de bobinas completas
    print(f"  ping-pong de 2 bobinas: {2 * T * B4 / MiB:.0f} MiB contra {64 * 36 * KiB / MiB:.2f} MiB de URAM + "
          f"{144 * 4.5 * KiB / MiB:.2f} MiB de BRAM: no entra")


if __name__ == "__main__":
    main()

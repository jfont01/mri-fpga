# Trabajo en curso que viene de la PC con WSL2 (2026-11-21)

*Parte de la bitácora. Esta carpeta existe solo para pasar a la PC con Ubuntu el trabajo de dos agentes que se cortaron al cerrar la sesión en la PC con WSL2 (vivían en `/home/jfont/mri-fpga-wip/`, fuera del repo). **Nada de esto está revisado ni aplicado.** Cuando se integre, borrar la carpeta.*

## 1. `chicos/`: casos `small` y `g0` del JSON de `unfold` + guarda en `cast.v`

Estado: **casi terminado**. Faltaba solo terminar de verificar la guarda en el núcleo FFT.

- `unfold_json.patch`: agrega `small` y `g0` al final de `modules/unfold/testbench/unfold_tb_regression.json` y sus vectores. Verificado por el agente: gtest 20/20 (223 tests), sim 20/20, vm 20/20, 0 errores; los 18 vectores viejos salen idénticos. **Ojo:** `small` sí ejercita el recorte de t de la etapa 4 (un mutante sin recorte falla con 686 errores), pero `g0` **no** (con G = 0, D ≤ 31 < NB_X = 36): ejercita G = 0 y NB_T > NB_D.
- `cast_guard.patch`: guarda con doble `$fatal` en `cast.v` (`NB_OUT > NB_IN + 2`) y el `static_assert` igual en `cast.hpp`. El agente probó que la condición es **exacta** (razonamiento cerrado + barrido de 3096 configuraciones en XSim, 0 violaciones; ver `NOTAS.md`).
  - Regresiones con la guarda: `cast` sim 14/14, vm 14/14; `cmul` 4/4 y 4/4; `cmul_karatsuba` 4/4 y 4/4; `fft1d_r22sdf` sim 103/103 y **vm 46 de 103 pasados, 0 fallas, cuando se cortó**: falta terminarlo.
  - **Dos fallas pre-existentes** encontradas (dan igual sin la guarda, no se arreglaron): `fft1d_r2sdf` vm 0/12 porque `run_xsim.tcl` busca `twiddles/n64/tw_re.mem` y ese núcleo tiene `tw_s<k>_re.mem` (con un tcl local que acepta los dos: 12/12); `fft1d_r2` sim 0/3 porque su modelo C++ no compila con g++ 13.3 (`-Wchanges-meaning` en `state_t`).
  - **La trampa vieja** (NBF_OUT > NBF_IN) no da X como dice [08](../08_trampas.md): da **un valor equivocado sin avisar** (no corre a la izquierda), salvo que además NB_OUT > NB_IN + 2.
- `NOTAS.md`: el detalle.

**Cómo seguir:** en un clon del HEAD de `new-IT`, aplicar los dos parches (`git apply --check` primero; las `flist` de `cast` no van), correr `run_regression_vm --keep-going` completo en `fft1d_r22sdf` (103 casos) y en `unfold`, mostrarle a Julián el diff de `cast.v` y `cast.hpp` (está en `cast_guard.patch`, es corto) y aplicar si lo aprueba.

## 2. `fft2d_teoria/`: teoría del RTL de la FFT 2D + prueba de la URAM

Estado: **parte A a mitad** (scripts s1–s5 hechos, falta el borrador `diseno_fft2d.md`), **parte B sin empezar** (módulo `corner_turn` para que Julián lo sintetice). El pedido completo está en `prompts/fft2d_teoria.md`.

Resultados de los scripts (del agente; las salidas están en `scripts/*.salida.txt`; **Claude no los revisó**):

- **s1, flujo:** con el corner-turn in situ, según las notas del agente: "disparo del lector en la muestra K0 de la ULTIMA fila; RAW min 512, WAR = adelanto 1023 (K0=0); ventana de arranque N1+N2-2 = 1534"; disparar en la penúltima fila falla con huecos (ver la salida). En régimen continuo la FFT 2D tarda **8T = 20,97 ms por trama** (el llenado se paga una vez: arranque 9T + 549 ciclos); con 3 buffers de A_l en la DDR no hay esperas.
- **s2, direcciones:** además de la rotación (período 19) hay otra permutación válida, "B" (x natural), de **período 4**, y con el fftshift absorbido **período 8**: más barata que la rotación. Verificado a tamaño completo con 40 bobinas y, ciclo a ciclo en 64 × 32 con 24 bobinas, **igual bit a bit al modelo C++ `fft2d`**. Sin BWE se pisan medias palabras.
- **s3, fftshift:** con k-space de fastMRI (`fft2c` = fftshift(fft2(ifftshift(img)))) el hardware sin shifts entrega la imagen con los **cuadrantes cambiados**, no solo con signos. Arreglo propuesto: S_hw = ifftshift(S) en el PS y direcciones de escritura x = br10(j1) ^ 512, y = br9(j2) + 512·(1 − a); con líneas impares, (−1)^a en U.
- **s4, coil reorder:** el stream intercalado por bobina que espera `unfold` sale de leer A_l en trozos (B = 1024: ráfagas de 4 KiB, BRAM ping-pong de 64 KiB = 16 RAMB36); **U'' en orden `pos_coil_a`** da el producto directo; la imagen se escribe por columnas con ráfagas de 2 KiB.
- **s5, presupuesto:** DDR 132 MiB de 4096; BRAM ~34 de 144 RAMB36 (24 %) sin el corner-turn; URAM 64 (100 %); DSP 40 de 1248. Alternativas: corner-turn en DDR por bloques (+1,6 GB/s, tiles de 16 KiB al 84 % de eficiencia, 768 KiB de buffers), `tile_transpose` (ráfagas de 256–512 B: 8–14 %, no sirve), 4 bancos de 16 URAM (uno por región de reloj, mux 4:1).
- Datos de AMD que encontró: UG573 v1.14 cap. 2, "Address collision is not possible with UltraRAM", A antes que B en el mismo ciclo, BWE `PARITY_INTERLEAVED`; XPM 2024.2: con SDP + ultra, `WRITE_MODE_B` solo `read_first` o `write_first`, y `write_first` exige `READ_LATENCY_B` ≥ 3.

Los scripts esperaban el repo en `../repo`: correrlos con `--repo $FPGA_MRI_ROOT` (s2 compila `scripts/cpp/core_timing.cpp` en `scripts/build/`).

**Cómo seguir:** relanzar el agente con `prompts/fft2d_teoria.md` (ya dice que arranque desde esta carpeta), o hacerlo en el chat. Después, explicarle a Julián el diseño decisión por decisión, como con P1–P7.

## 3. `datasets/`

Los dos scripts de NumPy del informe de datasets (`doc/datasets_reales.md`) y sus notas. Correr con `python3 -I`.

## 4. `prompts/`

Los pedidos para relanzar los agentes en la PC con Ubuntu.

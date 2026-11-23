# NOTAS del agente fft2d_teoria (2026-11-20)

Tarea: parte A (teoria del RTL de la FFT 2D, borrador `diseno_fft2d.md` + scripts NumPy)
y parte B (modulo `corner_turn` de prueba de la URAM, parche).

## Lugar de trabajo
- `/home/jfont/mri-fpga-wip/fft2d_teoria/`
  - `repo/`: clon. Rama local `fft2d_teoria_base`, commit `bb083df` =
    480a141 + d2_unfold.patch + rtl_unfold.patch + rtl_cordic_mag.patch.
    (Comparado con e4974c3 del repo de Julian: solo difiere doc/, el codigo es igual.)
    venv: symlink a la del repo de Julian.
  - `scripts/`: NumPy de la parte A.
  - `ref/ug573.txt`: pdftotext de UG573 v1.14.

## Hecho
- [x] Lectura: CLAUDE.md, estado, 02, hallazgos, teoria_fft, 03, 04 (4.8, 4.10), 05 (5.7),
      07, diseno_unfold_cordic, fft2d.hpp, fft1d_r22sdf.v, fft2d_ref.py, unfold_ref.py,
      fft2d_check_full.py, drawio.
- [x] Clon preparado y commit local.

- [x] scripts/cpp/core_timing.cpp: el nucleo (modelo C++) tiene latencia constante por trama
      con huecos entre tramas (0, 5, 3000) y la ultima trama se vacia sola. OK.
- [x] s1_flujo.py (log s1_flujo.log): in situ, disparo del lector en la muestra K0 de la ULTIMA
      fila; RAW min 512, WAR = adelanto 1023 (K0=0); ventana de arranque N1+N2-2 = 1534;
      regimen 8T = 20,97 ms por trama, arranque 9T+549; disparo en la penultima fila falla con
      huecos; 3 buffers de A en DDR: sin esperas; serie 44,6 ms.
- [x] s2_direcciones.py (log): A (rotacion) periodo 19; B (x natural) es permutacion de bits con
      periodo 4, B+fftshift periodo 8 (!: mas barato que la rotacion). Tamano completo 40 bobinas,
      medias palabras; sin BWE se pisan medias palabras; 64x32 x 24 bobinas ciclo a ciclo con
      adelanto 0, 1, 63 y el maximo (94/93) == modelo C++ fft2d bit a bit; max+1 falla.
- [x] s3_fftshift.py: fastMRI fft2c -> cuadrantes cambiados; arreglo: S_hw = ifftshift(S) en el PS
      y direcciones de escritura x = br10(j1)^512, y = br9(j2) + 512(1-a); lineas impares -> (-1)^a en U.

## Pendiente
- s4 (coil reorder y escritura de imagen), s5 (presupuesto), diseno_fft2d.md, modulo corner_turn,
  parche, informe.

## Hallazgos en el camino (a confirmar con NumPy)
- check D de fft2d_ref usa `gen_kspace --fftshift` = fftshift(fft2(img)): solo centra el
  k-space. fastMRI (fft2c) es fftshift(fft2(ifftshift(img))): entonces el hardware sin shifts
  entrega ifftshift(img)*(-1)^n: cuadrantes cambiados, no solo signos.
- UG573 v1.14 cap. 2: "Address collision is not possible with UltraRAM", A antes que B en el
  mismo ciclo, BWE PARITY_INTERLEAVED (8 bits, cada uno 8 datos + 1 paridad), SELF_MASK 7c0
  para 31-64 URAM, registros extra al cruzar region de reloj (fig. 38).
- XPM 2024.2 (xpm_memory.sv): BYTE_WRITE_WIDTH 8 o 9; SDP+ultra: WRITE_MODE_B solo
  read_first o write_first; write_first exige READ_LATENCY_B >= 3.

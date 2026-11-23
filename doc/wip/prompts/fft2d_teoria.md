# Pedido para relanzar el agente de la teoría de la FFT 2D (PC con Ubuntu)

*Copia adaptada del pedido original del 2026-11-20. Antes de lanzarlo, ajustar las rutas si el repo no está en `/home/jfont/Desktop/mri-fpga`.*

---

Sos un agente trabajando solo en la tesis de Julián (repo `mri-fpga`: reconstrucción SENSE de MRI a 20 fps, imagen de 1024 × 1024, L = 8 bobinas, Af = 2, en una Kria KV260 / XCK26, PL a 200 MHz), en su PC con Ubuntu. No vas a poder preguntarle nada: tomá decisiones razonables, documentalas y dejá las que son de él como preguntas. Todo en español rioplatense (comentarios de código en ASCII). Julián es electrónico (sabe mucho de RTL y FPGA); explicá con claridad y con números.

## Reglas (de CLAUDE.md, obligatorias)
- El repo de Julián está en `/home/jfont/Desktop/mri-fpga` (rama `new-IT`). **NO lo modifiques ni hagas en él ninguna operación de git que lo cambie.** Solo leerlo y clonarlo. No toques `doc/` (proponés texto para la bitácora).
- Trabajá en un lugar durable: `/home/jfont/mri-fpga-wip/fft2d_teoria/`. Clon: `git clone /home/jfont/Desktop/mri-fpga <dir>/repo`, `git config core.fileMode false`, HEAD de `new-IT`. Venv: symlink a `scripts/python/venv` del repo de Julián.
- **Retomá desde `doc/wip/fft2d_teoria/`** (un agente anterior se cortó): leé su `NOTAS.md`, los scripts s1–s5 y sus salidas (`*.salida.txt`), y `doc/wip/README.md`. Rechequeá lo que uses (correlos con `--repo <clon>`); no rehagas lo que ya está bien.
- **Teoría antes que código.** Verificá con NumPy cada fórmula, permutación y número. Decí qué no pudiste verificar.
- Citá documento, versión y capítulo de AMD (`books/amd-docs/`: UG573 v1.14, DS925, DS987, UG901, …; buscá con `pdftotext -layout`).
- **No corras síntesis ni implementación**: las corre Julián. Sí XSim y consultas de Vivado que no sinteticen. Nada de placa.
- Guardá avance en `/home/jfont/mri-fpga-wip/fft2d_teoria/NOTAS.md`.

## Qué leer
`CLAUDE.md`; `doc/estado.md`; `doc/02_arquitectura.md` (presupuesto y corner-turn, estrategia 1, con lo del 2026-11-19 y 2026-11-20); `doc/hallazgos.md` (H19–H23, H30, H33, H34); `doc/teoria_fft.md`; `doc/03_fft_r22sdf.md`; `doc/04_precision.md` §4.8 y §4.10; `doc/05_verificacion.md` §5.7; `doc/07_integracion.md`; `doc/diseno_unfold_cordic.md` (cómo espera `unfold` los datos, `i_last`); `doc/datasets_reales.md` §3 (convención de fastMRI); `modules/fft2d/`, `modules/fft1d_r22sdf/`, `modules/tile_transpose/`, `model/fft2d_ref.py`, `model/fft2d_check_full.py`, `model/unfold_ref.py`; `doc/arquitectura_objetivo.drawio`.

## Hipótesis de trabajo (a confirmar por Julián)
- **Un carril por bloque y fases solapadas:** un núcleo FFT por pasada (1024 por filas, 512 por columnas con la etapa radix-2 final), un carril de `unfold`, un CORDIC; la FFT de la trama n + 1 corre mientras `unfold` despliega la n. Según s1, en régimen la FFT tarda 8T = 20,97 ms por trama.
- **Corner-turn in situ en URAM:** una bobina (2^19 muestras de 32 bits) en las 64 URAM288, **2 muestras por palabra de 72 bits** (hasta 18 + 18 bits por muestra), media palabra con BWE, lectura y escritura en la misma dirección en el mismo ciclo (A antes que B, UG573 v1.14 cap. 2). Las 64 URAM288 del XCK26 están **en una sola columna**, 16 por región de reloj (X2Y0–X2Y3; consulta de sitios con Vivado 2024.2). Generador de direcciones: rotación (período 19) o la permutación "B" de s2 (período 4; 8 con el fftshift).

## Parte A: borrador de diseño
Escribí `/home/jfont/mri-fpga-wip/fft2d_teoria/diseno_fft2d.md`, en el estilo de `doc/diseno_unfold_cordic.md` (decisiones numeradas F1, F2, … con opciones, recomendación y por qué), con los scripts que lo verifican. Como mínimo:
1. Flujo de una bobina y de una trama: latencias, arranque, régimen, cambio de trama sin pausa, parada; tiempo por trama.
2. Orden de las muestras: bit-reverso del núcleo, fftshift (y la convención de fastMRI de s3); qué se absorbe en el generador de direcciones.
3. Generador de direcciones con 2 muestras por palabra: bit de media palabra, combinación con lo anterior, hardware, período; rotación contra "B".
4. Salida hacia `unfold`: A_l a la DDR y vuelta intercalada por bobina (coil reorder), ráfagas, BRAM, eficiencia de DDR (H21, H22, H30, `doc/07`), y el orden que impone a U'' (`pos_coil_a` según s4).
5. Presupuesto: tráfico por puerto HP con fases solapadas, DDR, BRAM, DSP, URAM al 100 %.
6. Riesgos y alternativas si el corner-turn no entra o no cierra timing (registros de cascada, bancos por región de reloj, DDR por bloques, `tile_transpose`), con números.
7. Lo que tiene que hacer `sense_ctrl` con la FFT 2D: secuencia, reset (`unfold` y `cordic_mag` no tienen; el núcleo FFT sí, `i_rst` síncrono), errores.
8. Preguntas para Julián y plan: modelo ciclo a ciclo de la `fft2d` con memoria, RTL, vm, síntesis.
No escribas el RTL de la FFT 2D.

## Parte B: módulo de prueba `corner_turn`, para que Julián lo sintetice
Módulo nuevo `modules/corner_turn/` en tu clon, creado con `create_module` (ver `scripts/shell/module_utils.sh` y cómo se crearon `cordic_mag` y `unfold`):
- `rtl/corner_turn.v`: entrada de una muestra por ciclo (pasada 1, con valid), salida de una por ciclo (pasada 2), memoria de 2^18 palabras de 64 bits en URAM (`(* ram_style = "ultra" *)` o `xpm_memory_*` con `MEMORY_PRIMITIVE "ultra"`; justificá con UG901/UG573 y lo que encontró el agente anterior sobre XPM), media palabra con BWE, lectura y escritura en la misma dirección en el mismo ciclo, y el generador de direcciones de la parte A. Tamaño parametrizable (defaults 512 × 1024) para simular chico. Guardas con doble `$fatal`.
- Verificación funcional siguiendo el flujo del repo si es razonable (modelo C++ + gtest + sim + vm); si no, un banco de XSim autoverificado (explicá). Varias bobinas encadenadas, chico y, si XSim lo aguanta, tamaño completo con 2–3 bobinas.
- `synthesis/` e `implementation/`: wrapper con registros de entrada y salida y JSON con un caso a 200 MHz en `xck26-sfvc784-2LV-c`, con límites que fallen si no usa 64 URAM. **NO los corras.** Comprobá con `--list-cases` y que el wrapper elabora en XSim.
- En el informe: qué tiene que mirar Julián (URAM 64, la cascada, WNS y camino crítico, latencia de lectura elegida).
- Parche con `git diff --binary` (archivos nuevos incluidos) en `/home/jfont/mri-fpga-wip/fft2d_teoria/corner_turn.patch`, LF y 100644, probado sobre otro clon.

## Informe final
Resumen de la parte A con las decisiones F y la recomendación de cada una; números clave; qué se verificó en NumPy y qué no; alternativas; la parte B archivo por archivo, resultados y qué mirar en la síntesis; decisiones tomadas sin consultar; rutas; comandos para aplicar el parche y correr `run_regression_synth` / `run_regression_impl` desde `modules/corner_turn`; texto propuesto para la bitácora.

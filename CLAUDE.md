# mri-fpga: reconstrucción SENSE en tiempo real sobre Kria KV260

Tesis de grado de Ingeniería Biomédica (UNC) de Julián Font. Reconstrucción de MRI con SENSE
(Pruessmann et al. 1999) a 20 fps en una Kria KV260 (SoM K26, XCK26, Zynq UltraScale+).

**La fuente de verdad es `doc/bitacora.md`**: estado, decisiones, resultados medidos,
versiones de los archivos, trampas y pendientes. Al empezar una sesión, leé el encabezado y
la sección que corresponda a la tarea (no hace falta leerla entera):

| Tema | Sección |
|---|---|
| Parámetros, glosario de símbolos | §1 |
| Arquitectura PS / PL, presupuesto de tiempo | §2 |
| FFT 1D R2²SDF (diseño, formatos, `SCALE_MASK`, `INVERSE`, N impar) | §3 |
| Precisión: sesgo *half-up* §4.6, cota §4.7, máscaras 2D §4.8, MAC §4.9–4.11, ida y vuelta §4.12 | §4 |
| Modelo C++, gtest, síntesis e implementación §5.6, `fft2d` §5.7 | §5 |
| Infraestructura y comandos (§6.4) | §6 |
| Trampas conocidas | §8 |
| Pendientes y plan | §9 |
| Registro de cambios | §12 |

Si algo de este archivo contradice a la bitácora, vale la bitácora (y avisame).
La descripción del proyecto en claude.ai está desactualizada (dice 512 × 512).

## Parámetros vigentes

- Imagen final 1024 × 1024. L = 8 bobinas, Af = 2 (submuestreo en phase-encode).
- k-space por bobina: 512 líneas de PE × 1024 de readout → FFT 2D de 1024 × 512 por bobina.
- Reloj de la PL: 200 MHz.
- Núcleo FFT R2²SDF: datapath 20 bits, twiddles 18, puertos 16 (Q1.15), `CMUL_PIPE = 1`.
  Máscaras provisorias de la IFFT 2D: 801 (pasada de 1024) y 384 (pasada de 512).
- PS (una vez por sesión): mapas → U = (SᴴS + λI)⁻¹Sᴴ por LDLᴴ sin raíces → U″ en DDR (32 MiB). λ = 10⁻⁴ propuesto, a confirmar.
- PL (cada trama): IFFT 2D por bobina en serie → MAC con U″ → CORDIC de magnitud (propuesta: n = 12, G = 5).

## Cómo trabajamos

- Español rioplatense, informal.
- **Teoría antes que código.** Primero desglosar el problema y acordarlo; no escribir todo
  el código de una.
- **Verificar fórmulas con NumPy antes de darlas.** Compilar y simular modelo y RTL antes de
  dar algo por terminado. Decir explícitamente qué no se pudo verificar.
- Ecuaciones en formato legible: en palabras y en bloques aparte, no símbolos apretados en
  una línea.
- Con notación de papers, pedirme capturas en vez de suponer los símbolos de memoria.
- Citar documento y capítulo de AMD/Xilinx (UG579, UG573, UG1085, DS987, PG021, ...).
  Los manuales están en `books/amd-docs/` (ver abajo).
- No darme la razón por darla. Si me equivoco, decilo.
- Usar mis funciones de entorno (`run_gtest`, `run_regression_sim`, ...), no
  `python3 scripts/python/...` a mano.

## Entorno y comandos

- Linux (Ubuntu nativo), backend de simulación XSim. El entorno se carga con
  `source scripts/set_env.sh` (necesita `FPGA_MRI_ROOT`; yo uso la función `fpga_mri_set`).
  Activa el venv de `scripts/python/venv` y carga Vivado/Vitis si están.
- **Cada llamada a la terminal es una shell nueva.** Si las funciones no están, cargar el
  entorno en el mismo comando:
  `export FPGA_MRI_ROOT=<raíz del repo> && source scripts/set_env.sh && cd modules/<mod> && run_gtest`
- Desde la carpeta del módulo:
  - `run_gtest --keep-going` → `run_regression_sim` → `run_regression_vm`
    (gtest no depende de sim; sim genera los vectores que usa vm).
  - `run_plot_fft1d`, `run_plot_fft2d` (este con `--full-check`, `--only`, `--coil`).
  - `run_regression_synth`, `run_regression_impl`: tardan; **preguntame antes de correrlos**.
- Estructura de un módulo: `modules/<mod>/{rtl,cpp,testbench,synthesis,implementation,flist,build}`.
- Un solo JSON por módulo, `testbench/<mod>_tb_regression.json`, para gtest, sim y vm:
  - las claves en MAYÚSCULAS se convierten en `-D<MOD>_<KEY>`;
  - las máscaras van en decimal;
  - `N_CYCLES` ≥ latencia + N (en la `fft2d`, 0 = automático);
  - `TW_DIR` tiene que corresponder a `NB_TW`.
- Modelos y estudios en Python: `model/` (bit a bit: `fft1d_r22sdf_fxp.py`,
  `mac_cordic_fxp.py`, `unfold_ref.py`; referencia: `fft2d_ref.py`), generadores en
  `model/gen/`, estímulos de la `fft2d` con `model/gen_fft2d_stim.py --from-json ...`.
- Twiddles: `gen_twiddles_r22sdf.py` (reproduce byte a byte las ROM existentes).

## Placa: reglas de seguridad

- **`make qspi` está prohibido** (`CONFIRM_QSPI=NO`): no usarlo hasta validar el firmware
  por JTAG. Está bloqueado también en `.claude/settings.json`, pero esa regla no es una
  barrera de seguridad: no buscarle la vuelta.
- Nada que toque la placa (JTAG, `xsdb`, `xsct`, programar la PL, cargar ELF) sin que yo
  lo pida en ese momento.

## Procedimiento para cambiar un módulo

1. Desglosar la teoría y acordarla conmigo.
2. Cambiar modelo y RTL juntos:
   - misma regla de defaults en los dos: `NBF_x = NB_x − (NB − NBF)`;
   - parámetros nuevos al final, para no romper instancias posicionales;
   - guardas con doble `$fatal` en el RTL (`generate` + `initial`) y `static_assert` en el modelo.
3. Verificar primero los defaults (tienen que dar idéntico a antes, byte a byte) y después
   la configuración nueva: gtest → sim → vm.
4. Respetar el fin de línea de cada archivo (`run_gtest.py` y `run_plot_fft1d.py` son CRLF;
   lo nuevo va en LF). No reformatear archivos enteros.
5. Git: trabajar en la rama en la que estoy; **no hacer commit ni push sin que lo pida**.
   Al terminar, mostrar qué cambió (`git diff --stat` y el diff de lo importante).
6. Cuando se cierra un resultado, una decisión o un cambio de infraestructura, actualizar
   `doc/bitacora.md`: la sección afectada, la tabla de versiones y el registro de
   cambios (§12, el más nuevo arriba).

## Trampas (detalle en la bitácora, §8)

- Twiddles con un ancho distinto a `NB_TW`: el vm pasa igual (RTL y modelo leen el mismo
  error). Solo lo detecta el gtest.
- En la unidad R2²SDF la **primera** mariposa es `BF2II`. El bit *i* de `SCALE_MASK` es la
  etapa *i* en orden de tiempo.
- `cast.v` no puede agregar bits fraccionales (devuelve X).
- Producto del modelo en `double`: exacto solo con `NB + NB_TW ≤ 52`.
- Redondeo *half-up* al descartar un bit: sesgado (+q/4 por etapa escalada). En 2D se
  concentra (fila p = 0; píxel (0, 0) en la ida y vuelta).
- N potencia de 2; con log₂N impar se agrega la etapa radix-2 final (`fft1d_r22sdf_tail`).
- Exceptions en `SetUpTestSuite` de gtest → tests *skipped* → antes el runner daba PASS
  (corregido en `run_gtest.py`). Simular dentro de los tests.
- iverilog ignora `$fatal` en elaboración (por eso el doble `$fatal`).

## Documentación

- `books/amd-docs/`: manuales de AMD/Xilinx y datasheets. Para buscar, `pdftotext -layout`
  (no hay versiones `.txt` guardadas); el PDF para tablas y figuras.
- `books/`: libros y papers (`books/papers/fft`, `books/papers/mri`). `doc/`: documentación
  propia (bitácora, `theory_r22sdf.md`, ecuaciones de SENSE).
- Bibliografía de la tesis: bitácora §11.

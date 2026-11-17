# Infraestructura

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

## 6. Infraestructura

### 6.1 Entorno

- **Plataforma: Linux** (WSL o Ubuntu nativo). MSYS ya no se usa (2026-10-17).
- `FPGA_MRI_ROOT` → `source scripts/set_env.sh` (con `fpga_mri_set`):
  - exporta `PROJECT_ROOT`, `MODULES_ROOT`, `RTLSIM_ROOT` y las rutas de los scripts;
  - detecta la plataforma (`linux` → backend `xsim`);
  - activa `scripts/python/venv`;
  - carga Vivado/Vitis (`VIVADO_SETTINGS_64_SH`, `VITIS_SETTINGS_64_SH`);
  - carga `module_utils.sh`, `flist_utils.sh`, `regression_utils.sh` y `xsim_utils.sh`.
- `help_env` lista los comandos.
- **Claude Code en VS Code (2026-10-25, migración desde claude.ai):** la bitácora pasa a vivir en el repo, en `doc/bitacora.md`, y esa copia es la fuente de verdad. `CLAUDE.md` en la raíz tiene las reglas de trabajo y el índice de la bitácora. `.claude/settings.json` bloquea `make qspi` y pide confirmación para `xsdb`, `xsct`, `git commit` y `git push`; es un resguardo, no una barrera: la protección real sigue siendo `CONFIRM_QSPI=NO`. Los manuales están en `books/amd-docs/` (no hay `.txt`: se busca con `pdftotext -layout`; detalle por PC en §6.2). Cada llamada de Claude a la terminal es una shell nueva: el entorno se carga en el mismo comando.
- **PDF:** `pdftotext` y `pdftoppm` (poppler-utils 24.02; instalados en la PC con Ubuntu, verificado el 2026-11-16). Las tablas y figuras que la capa de texto pierde (pasó con la Tabla 1 de He y Torkelson) se miran como imagen: `pdftoppm -f 5 -l 5 -r 300 -png <pdf> <prefijo>` (página 5).
- **Dos PC (2026-11-16):** Ubuntu nativo (esta, con la placa) y Windows con WSL2. El repo se sincroniza por git; lo que no está versionado (los PDF, `build/`, los proyectos de Vivado/Vitis de `bringup/`) hay que copiarlo a mano.

### 6.2 Árbol del repositorio (`mri-fpga/`, actualizado el 2026-11-16)

```
books/           libros en PDF (+ papers/fft, papers/mri, amd-docs/ con los manuales de AMD, Xilinx DSP Workbook)
bringup/
  scripts/       Makefile, config.mk, boot/, jtag/, tools/, vitis/, vivado/   infraestructura de placa
  v0/            loopback DMA (Tcl de proyecto y block design, firmware)
  v1/, v1.1/     hardware/, firmware/, results/, Makefile                    (v1.1: 250 MHz; estado a confirmar)
doc/             la bitácora separada por archivos (índice: bitacora.md; teoría en teoria_*.md)
model/           modelos y estudios en Python: bit a bit (fft1d_r22sdf_fxp.py, mac_cordic_fxp.py, unfold_ref.py),
                 referencia (fft2d_ref.py), generadores (gen/, gen_fft2d_stim.py, gen_ps_sense_vectors.py),
                 estudios (*_study.py, fft1d_scaling_bound.py), cfxp.py / fxp.py
modules/         btfly_sdf, cast, cmul, cmul_karatsuba, delay_line,
                 fft1d_r2, fft1d_r2sdf, fft1d_r22sdf, tile_transpose (2026-08: RTL y modelo, regresión vacía),
                 fft2d (modelo C++ ciclo a ciclo; rtl/fft2d.v es la plantilla vacía),
                 ps_sense (software del PS, 2026-11-15; sin RTL), unfold (plantilla vacía de create_module, 2026-10-25)
rtlsim/          rtlsim.hpp, rtlsim_core.hpp, rtlsim_trace.hpp, include/ (ap_fixed de Vitis HLS)
scripts/         Makefile, set_env.sh, lint_waivers.txt,
                 python/ (runners + venv), shell/ (utilidades), tcl/ (xsim, synth, impl)
CLAUDE.md, .gitattributes (binarios, 2026-11-16), .gitignore (*.pdf, build/, amd/, ...)
```

**Documentación disponible para consultar.** Desde la migración a Claude Code (2026-10-25), Claude lee directamente `books/` y `doc/` del repo. Los PDF **no están versionados** (`*.pdf` en `.gitignore`; las excepciones son los libros y papers que se agregaron antes de esa regla y PG151): cada PC tiene su copia.

**Ruta única para las dos PC: `books/amd-docs/`** (la que dice `CLAUDE.md`). En la PC con WSL2 los manuales estaban en `doc/amd/`, que git ignora (`amd/` en `.gitignore`); **movidos a `books/amd-docs/` el 2026-11-17** (31 PDF, sin choques de nombre). En `doc/amd/` quedaron, ignorados por git, `pdf2text.sh`, una carpeta `pdf/` vacía y `txt/` con 27 conversiones viejas a texto (de agosto, sin `-layout`): para buscar se usa `pdftotext -layout` sobre el PDF. Al citar, poner la versión del documento: los PDF son de varias versiones (p. ej. UG1085 v2.5 de 2025, DS987 v1.6 de 2026, UG579 v1.11 de 2021, UG573 v1.14 de 2025, los de Vivado 2026.1) y las herramientas, 2024.2.

| PC | Dónde y cuántos | Qué falta |
|---|---|---|
| **Ubuntu nativo** (verificado el 2026-11-16) | 76 PDF en `books/`, todos abren y 75 tienen texto (`books/papers/fft/patent_7415584.pdf` está escaneado). `books/amd-docs/`: **31** — UG1085 (TRM), UG579 (DSP), UG573 (memorias), UG572 (clocking), UG949 (metodología), UG896 (IP), UG994/995 (IP integrator), UG1037 (AXI), UG1118 (IP propio), UG984, DS987 (K26), datasheets del KV260 y del Zynq UltraScale+ (DS891), PG021, PG022, PG034, PG035, PG037, PG059, PG065, PG078, PG085, **PG109 (FFT)**, PG125, PG144, PG151, PG164, PG201, PG247, WP417. Libros (13), papers (7) y el Xilinx DSP Workbook (25, incluido `notes/TheCORDICAlgorithmXilinx.pdf`) | DS925, UG1400, UG1137, UG1089 (están en la otra PC: copiarlos). UG1209 no existe en PDF: solo en línea (ver abajo) |
| **Windows con WSL2** (según la revisión del 2026-11-16; manuales movidos el 2026-11-17) | 52 PDF, todos en `books/`: 32 en `books/amd-docs/` (los 31 que estaban en `doc/amd/`, con DS925, UG1400, UG1137 y UG1089, más PG151) y 20 entre libros y papers | PG109 y el Xilinx DSP Workbook (copiarlos de esta PC). UG1209: solo en línea |

**UG1209** (*Zynq UltraScale+ MPSoC Embedded Design Tutorial*) AMD lo publica solo en HTML, sin PDF (Julián, 2026-11-16): <https://docs.amd.com/r/en-US/ug1209-embedded-design-tutorial/Zynq-UltraScale-MPSoC-Embedded-Design-Tutorial-UG1209>. Al citarlo, poner la URL y la fecha de consulta.

Los `.doc` del DSP Workbook se leen convirtiéndolos con libreoffice.

| Dónde | Qué hay |
|---|---|
| `books/` | Parhami (*Computer Arithmetic*), Khan (*Digital Design of Signal Processing Systems*), Gonzalez & Woods (*Digital Image Processing*), Watkins y Golub & Van Loan (*Matrix Computations*), Akçakaya et al. (*MR Image Reconstruction*), Brown, Cheng & Haacke (*MRI: Physical Principles*), *MRI From Picture to Proton*, *Parallel Imaging in Clinical MR Applications*, Nishimura (*Principles of MRI*), Liang & Lauterbur (*Principles of MRI: A Signal Processing Perspective*), Paul & Mathew (*Regularized Image Reconstruction in Parallel MRI with MATLAB*), Proakis & Manolakis (*Tratamiento digital de señales*) |
| `books/papers/` | `fft/`: He & Torkelson 1996, Garrido 2022 (survey; online 2021), Mukherjee & Choudhury 2018 (FFT 2D), WiMAX, un review de deep learning (no citar), una patente escaneada. `mri/`: Weiger, Pruessmann & Boesiger 2000 |

### 6.3 Estructura de un módulo

```
modules/<mod>/
  rtl/             <mod>.v (+ submódulos, twiddles/)
  cpp/             <mod>.hpp, <mod>.cpp       modelo rtlsim
  testbench/       <mod>_tb.sv, _tb.cpp, _tb.hpp, _gtest.cpp, <mod>_tb_regression.json
  synthesis/       <mod>_wrapper_synth.v, <mod>_synth_regression.json
  implementation/  <mod>_wrapper_impl.v, <mod>_impl_regression.json
  flist/           autogenerado por update_flist, a partir de los `include
  build/<CASE>/    todo lo de un caso: binary/, simulation/vectors/, reports/, plots/
  <mod>_vars.sh
```

- Convención: nombre del módulo = nombre del archivo.
- `` `include "x.v" `` se resuelve primero relativo al archivo y después en `modules/x/rtl/x.v`.

### 6.4 Comandos (desde la carpeta del módulo)

| Comando | Qué hace |
|---|---|
| `run_gtest [--keep-going] [--case X] [--mode fixed\|double]` | gtest del modelo en los dos modos |
| `run_regression_sim` | compila `_tb.cpp` + modelo y genera `stimuli/` y `expected/` |
| `run_regression_vm [--keep-going]` | `update_flist`, XSim con `_tb.sv`, compara `actual/` contra `expected/` |
| `run_plot_fft1d [--src expected\|actual\|both]` | figura por caso: entrada, \|X\|, fase y error contra la FFT ideal |
| `run_plot_fft2d [--only X] [--coil N\|--all-coils] [--full-check]` | figura por caso y bobina de la `fft2d` (lee los vectores de sim). Función agregada el 2026-10-21 (`RUN_PLOT_FFT2D_PY` en `set_env.sh` + función en `regression_utils.sh`) |
| `run_regression_synth` / `run_regression_impl` | Vivado: LUT, FF, DSP, BRAM, WNS |
| `run_compile_rtl [--testbench]`, `run_lint` | chequeos rápidos |
| `create_module <nombre>`, `update_flist`, `cd_module`, `list_modules` | gestión de módulos |

Orden típico: `rm -rf build && run_gtest --keep-going && run_regression_sim && run_regression_vm`.

**`modules/ps_sense`** (software del PS, sin RTL; desde 2026-11-15): no tiene sim, vm ni síntesis. Desde la carpeta del módulo:

| Comando | Qué hace |
|---|---|
| `run_gtest --keep-going` | los 18 casos de `testbench/vec/` contra `unfold_ref.py`, bit a bit (un solo modo, `[fixed]`) |
| `make compile-baremetal` | **compila** `build/baremetal/libps_sense.a` para Vitis (A53 sin sistema operativo) e inspecciona que tenga 0 FMA; no ejecuta nada |
| `make compile-a53-linux` | **compila** el gtest para el A53 con Linux (estático), el que usa `run-qemu` |
| `make compile-host` | **compila** el gtest para la PC con `-O2` |
| `make compile` | los tres |
| `make run-host` | **corre** el gtest en la PC sobre los 18 casos |
| `make run-qemu` | **corre** el gtest del A53 emulado por qemu sobre los 18 casos |
| `make run-full` | genera los casos de 1024 × 1024 (`build/vec_full`, ~600 MB) y los **corre** en la PC y en qemu |
| `make run` | `run-host` + `run-qemu` |
| `make` | la ayuda (las mismas líneas del encabezado del Makefile) |

Los vectores se regeneran con `python3 model/gen_ps_sense_vectors.py --from-json modules/ps_sense/testbench/ps_sense_tb_regression.json` (determinista: mismos bytes).

**Al pasarle comandos a Julián, usar SIEMPRE estas funciones** (no `python3 ../../scripts/python/...`). gtest no depende de sim (genera su propio estímulo); los plots sí leen los vectores de sim.

### 6.5 Convenciones del JSON

- Las claves en MAYÚSCULAS se convierten en `-D<MODULO>_<CLAVE>=<valor>` (`auto_defines`), tanto para C++ como para Verilog.
- `N_CYCLES` es del flujo, no se convierte en define.
- Los valores tienen que valer igual en C++ y Verilog: **máscaras en decimal**; `TW_DIR` sin comillas (lo pasa a string `STRINGIFY`).
- Los `STIM_*` llegan como `FFT1D_R22SDF_STIM_*`, y `_tb.cpp` los traduce a `FFT1D_STIM_*` para `fft1d_stimulus.hpp`. Amplitudes ×1000 en enteros, fases en grados.
- `fft1d_stimulus.hpp` normaliza el pico a ≤ 0,9375, pero solo **hacia abajo**.
- Un solo JSON para gtest, sim y vm: cada runner ignora las claves del otro.

### 6.6 Vectores

```
build/<CASE>/simulation/vectors/
  stimuli/in_ports/{i_valid,i_re,i_im}.dat
  expected/out_ports/{o_valid,o_re,o_im,r_out_count.o,u0_*_count.o}.dat     ← modelo
  actual/out_ports/…                                                        ← RTL
```

- Los `.dat` son enteros crudos en decimal: `i_*` en NB_IN bits, `o_*` en NB_OUT.
- Una línea por ciclo: la entrada aplicada en el ciclo *c* y la salida después del flanco *c*.

### 6.7 `scripts/Makefile` (C++)

- `-std=c++17 -O0 -g -Wall -Wextra -ffp-contract=off`. `HLS_AP_DIR = $RTLSIM_ROOT/include` (los `ap_fixed` están en el repo).
- `-ffp-contract=off` desde el 2026-11-15 (H29): no cambia el código de máquina de ningún módulo existente (56 objetos comparados, [05 §5.8](05_verificacion.md)); lo necesita `ps_sense`.
- Compila todo en un solo comando.
- **Bug encontrado el 2026-10-17:** el binario no dependía de `DEFINES`. Si se cambiaba un parámetro de un caso sin cambiar su nombre ni tocar fuentes, `make` reusaba el binario viejo y **se probaba la configuración anterior sin ningún aviso**. Tampoco seguía los headers que no figuran en `EXTRA_HEADERS`, como `fft1d_stimulus.hpp`.
- **Arreglo** (entrega `makefile_fix`, 2026-10-17):
  - `$(TARGET).flags` guarda la línea de compilación completa; se reescribe solo si cambió, y el binario depende de él.
  - `$(TARGET).d` guarda las dependencias de headers (`g++ -MM -MP`).
  - Necesita GNU make ≥ 4.2.
  - Verificado en 7 escenarios con el Makefile solo, y de punta a punta con `run_gtest.py` y `run_regression_sim.py` reales: al cambiar NB de 20 a 24 en el JSON se recompila solo el caso afectado, y el reporte muestra NB = 24.

### 6.8 Bring-up (Vivado/Vitis)

- Infraestructura propia en `bringup/scripts/` (Makefile y `config.mk`) con `make build`, `make bootbin`, `make elf`, `make run` y `make uart`, más los scripts Tcl de JTAG.
- `make qspi` está protegido (`CONFIRM_QSPI=NO`): no usar hasta validar el firmware por JTAG.

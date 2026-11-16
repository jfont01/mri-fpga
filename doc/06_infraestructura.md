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
- **Claude Code en VS Code (2026-10-25, migración desde claude.ai):** la bitácora pasa a vivir en el repo, en `doc/bitacora.md`, y esa copia es la fuente de verdad. `CLAUDE.md` en la raíz tiene las reglas de trabajo y el índice de la bitácora. `.claude/settings.json` bloquea `make qspi` y pide confirmación para `xsdb`, `xsct`, `git commit` y `git push`; es un resguardo, no una barrera: la protección real sigue siendo `CONFIRM_QSPI=NO`. Los manuales están en `books/amd-docs/` (no hay `.txt`: se busca con `pdftotext -layout`). Cada llamada de Claude a la terminal es una shell nueva: el entorno se carga en el mismo comando.

### 6.2 Árbol del repositorio (`mri-fpga/`, 2026-10-17)

```
books/           libros en PDF (+ papers/fft, papers/mri, amd-docs/ con los manuales de AMD, Xilinx DSP Workbook)
bringup/
  scripts/       Makefile, config.mk, boot/, jtag/, tools/, vitis/, vivado/   infraestructura de placa
  v0/            loopback DMA (Tcl de proyecto y block design, firmware)
  v1/, v1.1/     hardware/, firmware/, results/, Makefile                    (v1.1: 250 MHz; estado a confirmar)
doc/             la bitácora separada por archivos (índice: bitacora.md; teoría en teoria_*.md)
model/           cfxp.py, fxp.py   (modelos de punto fijo en Python)
modules/         btfly_sdf, cast, cmul, cmul_karatsuba, delay_line,
                 fft1d_r2, fft1d_r2sdf, fft1d_r22sdf, tile_transpose
rtlsim/          rtlsim.hpp, rtlsim_core.hpp, rtlsim_trace.hpp, include/ (ap_fixed de Vitis HLS)
scripts/         Makefile, set_env.sh, lint_waivers.txt,
                 python/ (runners + venv), shell/ (utilidades), tcl/ (xsim, synth, impl)
CLAUDE.md
```

**Documentación disponible para consultar.** Desde la migración a Claude Code (2026-10-25), Claude lee directamente `books/` y `doc/` del repo. Verificado: los 76 PDF abren y 75 tienen texto; `books/papers/fft/patent_7415584.pdf` está escaneado (se ve como imagen, no se puede buscar). Los `.doc` del DSP Workbook se leen convirtiéndolos con libreoffice.

| Dónde | Qué hay |
|---|---|
| Repo `books/amd-docs/` | UG1085 (TRM), UG579 (DSP), UG573 (memorias), UG572 (clocking), UG949 (metodología), UG896 (IP), UG994/995 (IP integrator), UG1037 (AXI), UG1118 (IP propio), UG984, DS987 (K26), datasheets del KV260 y del Zynq UltraScale+, PG021, PG022, PG034, PG035, PG037, PG059, PG065, PG078, PG085, PG109 (FFT), PG125, PG144, PG151, PG164, PG201, PG247, WP417. **No están en el repo** (solo estaban en el proyecto de claude.ai): DS925, UG1400, UG1137, UG1089, UG1209 |
| Repo `books/` | Parhami (*Computer Arithmetic*), Khan (*Digital Design of Signal Processing Systems*), Gonzalez & Woods (*Digital Image Processing*), Watkins y Golub & Van Loan (*Matrix Computations*), Akçakaya et al. (*MR Image Reconstruction*), Brown, Cheng & Haacke (*MRI: Physical Principles*), *MRI From Picture to Proton*, *Parallel Imaging in Clinical MR Applications*, Nishimura (*Principles of MRI*), Liang & Lauterbur (*Principles of MRI: A Signal Processing Perspective*), Paul & Mathew (*Regularized Image Reconstruction in Parallel MRI with MATLAB*), Proakis & Manolakis (*Tratamiento digital de señales*), PG151 (Divider Generator) |

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

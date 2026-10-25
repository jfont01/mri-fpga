# Bitácora técnica — Reconstrucción SENSE en tiempo real sobre Kria KV260

*Tesis de grado de Ingeniería Biomédica (UNC). Última actualización: 2026-10-25.*

Este documento junta el contexto de la tesis, las decisiones de diseño, los resultados medidos, la infraestructura y los pendientes. Tiene dos usos:

1. **Material para escribir la tesis.** Cada resultado dice cómo se obtuvo y con qué herramienta.
2. **Retomar el contexto** al empezar una sesión nueva (vos o Claude).

**Convenciones**

| Marca | Significa |
|---|---|
| ✅ | verificado: simulado, medido o comparado bit a bit |
| 🟡 | estimado: cuenta en papel o modelo simplificado (NumPy ideal) |
| ⏳ | pendiente |
| ❌ | descartado |

Los SQNR están en dB. "LSB" sin aclaración es el LSB del puerto de salida (2⁻¹⁵ en Q1.15).

**Estado al 2026-10-17, en una línea.** La FFT 1D R2²SDF de N = 1024, con datapath de 20 bits, twiddles de 18, puertos de 16 y máscara 784, está verificada (gtest 28/28, vm 14/14 bit a bit en XSim) y cierra a 200 MHz después del ruteo, pero con solo 50 ps de margen. Para ganar margen se agregó `CMUL_PIPE` (registro antes del `cast` del `cmul`, latencia 1040). Con él cierra con 0,903 ns de margen después del ruteo (Fmax 244 MHz), a cambio de 168 FF; desde el 2026-10-18 es el default. El camino crítico pasó a la línea de retardo de BF2I.

**Versiones vigentes de los archivos** (todas aplicadas en el repo):

| Archivo | Entrega |
|---|---|
| `modules/cmul/rtl/cmul.v`, `rtl/fft1d_r22sdf.v`, `fft1d_r22sdf_unit.v`, `cpp/fft1d_r22sdf.hpp`, `testbench/fft1d_r22sdf_tb.sv`, `_tb.cpp`, `_gtest.cpp`, `_tb_regression.json`, wrappers y JSON de síntesis e implementación | **`cmul_pipe` (2026-10-17, aplicado)**: reemplaza a las entregas de abajo para esos archivos |
| `rtl/fft1d_r22sdf.v`, `fft1d_r22sdf_unit.v`, `cpp/fft1d_r22sdf.hpp`, `testbench/fft1d_r22sdf_tb.sv`, `_tb_regression.json`, wrappers y JSON de síntesis e implementación | **`cmul_pipe_default` (2026-10-18)**: default `CMUL_PIPE = 1`; va encima de `cmul_pipe` |
| `rtl/fft1d_r22sdf.v`, `cpp/fft1d_r22sdf.hpp`, `testbench/fft1d_r22sdf_tb.sv`, `_tb.cpp`, `_gtest.cpp`, `_tb_regression.json`, wrappers, `scripts/python/run_plot_fft1d.py`, `model/fft2d_ref.py` | **`fft_inverse` (2026-10-18, aplicado)**: parámetro `INVERSE`; va encima de `cmul_pipe_default` |
| `rtl/fft1d_r22sdf.v`, **`rtl/fft1d_r22sdf_tail.v` (nuevo)**, `cpp/fft1d_r22sdf.hpp`, `testbench/_gtest.cpp`, `_tb.cpp`, `_tb_regression.json`, JSON de síntesis e implementación, twiddles `n8`, `n32`, `n128`, `n512`, `n512_tw18`, **`model/fft2d_theory.py` (nuevo)** | **`fft_n512` (2026-10-18, falta aplicarlo)**: N = 2^impar; va encima de `fft_inverse` |
| `model/fft1d_r22sdf_fxp.py` (modelo bit-exacto vectorizado), `model/fft2d_mask_study.py` (estudio de máscaras) | **`fft2d_mascaras` (2026-10-18, nuevos)** |
| `rtl/fft1d_r22sdf.v`, `fft1d_r22sdf_unit.v`, `cmul.v` (+ `btfly_bf2i/ii.v` con `SCALE`) | `fft_puertos` (2026-10-11) |
| `cpp/fft1d_r22sdf.hpp` | `gtest_double` (2026-10-12) |
| `testbench/fft1d_r22sdf_gtest.cpp` | `gtest_mascara` (2026-10-14) |
| `testbench/fft1d_r22sdf_tb.sv`, `_tb.cpp`, `_tb_regression.json` | `vm_json_unico` (2026-10-17) |
| `scripts/python/run_plot_fft1d.py` | `plot_fft1d` (2026-10-17) |
| `scripts/Makefile` | `makefile_fix` (2026-10-17, falta aplicarlo) |
| `synthesis/fft1d_r22sdf_wrapper_synth.v`, `implementation/fft1d_r22sdf_wrapper_impl.v` y sus JSON | `synth_impl` (2026-10-17) |
| `rtl/fft1d_r22sdf.v` con `` `ifndef __SYNTHESIS__ `` (agregado por Julián) y el `` `endif `` corregido | `synth_fix` (2026-10-17) |
| `gen_twiddles.py` | `fft_puertos` |
| `model/fft2d_ref.py` (referencia y convenciones de la FFT 2D; los `gen_*.py` están en `model/gen/`) | `fft_inverse` (2026-10-18). Self-test en el repo: `TODO OK` ✅ |
| **`CLAUDE.md`, `.claude/settings.json`, `docs/bitacora_tesis.md`, `model/fft2d_roundtrip_study.py` (nuevos)** | **`migracion_vscode` (2026-10-25, falta aplicarlo)** |

---

## Índice

1. [Contexto de la tesis](#1-contexto-de-la-tesis)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [FFT 1D R2²SDF: diseño](#3-fft-1d-r2sdf-diseño)
4. [Estudio de precisión](#4-estudio-de-precisión)
5. [Modelo C++ y verificación](#5-modelo-c-y-verificación)
6. [Infraestructura](#6-infraestructura)
7. [Bring-up en placa](#7-bring-up-en-placa)
8. [Trampas conocidas](#8-trampas-conocidas)
9. [Pendientes y próximos pasos](#9-pendientes-y-próximos-pasos)
10. [Riesgos y preguntas abiertas](#10-riesgos-y-preguntas-abiertas)
11. [Bibliografía](#11-bibliografía)
12. [Registro de cambios](#12-registro-de-cambios)

---

## 1. Contexto de la tesis

### Objetivo

Reconstruir imágenes de resonancia magnética con **SENSE** (Pruessmann et al. 1999) a **20 fps** (50 ms por trama) en una **Kria KV260**: SoM K26, chip XCK26, Zynq UltraScale+ MPSoC.

### Parámetros del sistema

| Parámetro | Valor | Nota |
|---|---|---|
| Imagen final | **1024 × 1024** | Decidido el 2026-10-10. La descripción del proyecto todavía dice 512×512; hay que actualizarla |
| Bobinas | L = 8 | |
| Factor de aceleración | Af = 2, submuestreo en *phase-encode* | |
| k-space por bobina | 1024 (readout, completo) × 512 (PE, submuestreado) | La **FFT2D es de 1024 × 512** por bobina; el unfolding lleva la imagen plegada de 1024 × 512 a la final de 1024 × 1024 |
| Formato base | Q1.15 (16 bits) en los puertos y en DDR | El datapath interno de la FFT es más ancho (§3.4) |
| Datos | Dataset offline o simulación | No hay escáner en vivo por ahora |

**Por qué 1024 y no 512.** A 512, el acumulador del unfolding (1 MiB) y el intermedio de una bobina (512 KiB) entran en chip; a 1024 el acumulador son 4 MiB, contra 2,88 MiB de BRAM + URAM. La memoria externa pasa a ser **estructural**, no una elección, y eso hace al problema más representativo. 🟡

### SENSE en una línea

Con Af = 2, cada píxel de la imagen plegada `y_l` (bobina *l*) es la suma de dos píxeles verdaderos pesados por la sensibilidad de la bobina:

```
y = S · m        y ∈ ℂ^L,  S ∈ ℂ^(L×Af),  m ∈ ℂ^Af
m̂ = (Sᴴ S)⁻¹ Sᴴ y = U · y      con  U ∈ ℂ^(Af×L)
```

`U` es la matriz de *unfolding* de Pruessmann.

### Glosario de símbolos (2026-10-24)

Versión legible, con las ecuaciones renderizadas y un diagrama: doc **"SENSE en el FPGA: ecuaciones en limpio"** (https://claude.ai/artifact/H3aqtmtvK5coLz5koBDyZk).

El camino de una trama, de punta a punta:

```
escáner ─► K_l ──×g_l, a 16 bits──► K_q,l ──IFFT 2D (PL)──► A_l ──MAC con U''──► m̂ ──CORDIC──► |m̂| (imagen final, u16)
           (k-space, 8 bobinas)      (entra al FPGA)          (imagen plegada,      (imagen       (magnitud)
                                                               una por bobina)       compleja)
                       el PS calcula U'' una vez por sesión a partir de los mapas s_l
```

| Símbolo | Qué es | Dónde aparece |
|---|---|---|
| l, L | número de bobina (0 … 7) y cantidad de bobinas (L = 8) | todo |
| Af | factor de aceleración (2): se mide 1 de cada Af líneas de phase-encode | todo |
| m | la imagen verdadera (lo que se quiere reconstruir), 1024 × 1024 | §1 |
| s_l | mapa de sensibilidad de la bobina l: cuánto "ve" esa bobina de cada píxel | §1, PS |
| K_l | k-space medido por la bobina l (ya submuestreado): 512 filas × 1024 muestras, complejo | entrada |
| g_l | ganancia por bobina que lleva el pico de K_l a 0,9 del fondo de escala de 16 bits. Fija por sesión | §4.8, §4.9 |
| K_q,l | K_l·g_l redondeado a Q1.15 (16 bits): lo que realmente entra al FPGA | §4.9 |
| S1, S2 | cuántas etapas de la pasada 1 y de la pasada 2 dividen por 2 (máscaras 801 y 384 → S1 = 4, S2 = 2). **No confundir con la matriz S** | §3.5, §4.8 |
| A_l | salida de la IFFT 2D del FPGA para la bobina l: la imagen **plegada** de esa bobina, 512 × 1024, Q1.15 | §3.8, §4.9 |
| y_l | lo mismo que A_l pero en unidades físicas: y_l = A_l · 2^(S1+S2−19) / g_l | §4.9 |
| x, p, a | x: columna (readout, 0 … 1023); p: fila de la imagen plegada (0 … 511); a: cuál de los Af píxeles verdaderos se superponen en (x, p) (filas p y p + 512) | §4.9 |
| S | matriz de sensibilidades de un píxel plegado: L × Af (8 × 2), S[l, a] = s_l en el píxel a | §1, §4.10 |
| U | matriz de desplegado de Pruessmann, Af × L (2 × 8) por píxel plegado: U = (SᴴS)⁻¹Sᴴ. Combina las 8 bobinas para separar los 2 píxeles superpuestos | §1, §4.10 |
| U' | U con la ganancia de cada bobina adentro: U' = U / g_l (así el PL usa A_l directo, sin deshacer g_l) | §4.9 |
| E | exponente común (por sesión) para que U' entre en 16 bits: U'' = U'·2^−E. Como el exponente de un punto flotante, pero uno solo para toda la tabla | §4.9 |
| U'' | lo que el PS le entrega al PL: U' escalada por 2^−E (y por 1/Kc) en Q1.15. 32 MiB | §4.10 |
| λ | regularización de Tikhonov: (SᴴS + λI)⁻¹ en vez de (SᴴS)⁻¹, para que no se divida por cero donde los mapas se anulan | §4.10 |
| m̂ ("m sombrero") | la imagen reconstruida (estimación de m), compleja: m̂ = Σ_l U·y_l | §1, §4.9 |
| CORDIC | algoritmo de sumas y corrimientos que calcula la magnitud √(re² + im²) sin multiplicadores ni raíz: gira el vector hacia el eje x en n pasos de ángulo arctan(2^−i) y al final la coordenada x es la magnitud | §4.11 |
| Kc | ganancia del CORDIC (≈ 1,647 con 12 pasos): sale x = Kc·magnitud. 1/Kc se mete en U'' | §4.11 (allí figura como K_n) |
| G | bits de guarda del CORDIC (fracción extra para que los truncamientos no se acumulen) | §4.11 |
| D, F_OUT | D: corrimiento de sesión que lleva la suma del MAC a la escala de salida; F_OUT: bits fraccionales de la salida u16 (15) | §4.11 |
| T | muestras por bobina = N1·N2 = 512 × 1024 = 524 288 | §5.7 |

La cuenta del MAC, en palabras: **en cada píxel plegado (x, p), y para cada uno de los 2 píxeles verdaderos que se le superponen, la imagen final es la suma, sobre las 8 bobinas, de (coeficiente de desplegado) × (imagen plegada de esa bobina), por una potencia de 2 fija**:

  m̂(x, p + a·512) = 2^(S1+S2−19) · Σ_l U'(x, p, a, l) · A_l(x, p)

El 19 es log2(1024·512): la IFFT normaliza por el tamaño y el hardware no lo hace (solo divide por 2 en las S1 + S2 etapas que escalan).

---

## 2. Arquitectura del sistema

### Partición PS / PL

**PS (ARM A53), una vez por sesión:**
1. Prescan → mapas de sensibilidad S (RSS o ESPIRiT).
2. A = SᴴS (hermitiana definida positiva) y factorización A = LDLᴴ, sin raíces cuadradas.
3. Resolver para obtener U = A⁻¹Sᴴ, de 2 × 8 complejos por píxel alias.
4. Pasar U a DDR4.

**PL (fabric), en cada trama:**
1. Por cada bobina, en serie sobre un único core: IFFT2D del k-space submuestreado → imagen plegada `y_l`. Pasada por filas (N = 1024), *corner-turn*, pasada por columnas (N = 512).
2. MAC de unfolding: `m̂ = U · y`, una matriz-vector de 2 × 8 por posición alias. No hay divisiones ni sustituciones en tiempo real.
3. CORDIC para la magnitud `|m̂|`.

**Idea clave.** S no cambia entre tramas, así que todo el Cholesky (que originalmente iba a correr en tiempo real) se reduce a U precalculado. En tiempo real solo quedan la IFFT2D y un MAC complejo.

**Mapas en el PS.** Se calculan una vez por estudio, en punto flotante (A53 a 1,33 GHz con NEON). Estimación: ~79 Mflop, decenas de ms. 🟡 La dificultad real no es el cómputo:
- La división `I_c / I_ref` se va al infinito fuera del objeto, así que hace falta máscara o regularización.
- El cociente crudo trae ruido y hay que suavizarlo.
- Se necesita el centro del k-space completo (líneas ACS; fastMRI las trae).

### Memoria

- Todo lo del tamaño de una imagen o más (k-space, intermedios, `y`, U, `m̂`) va a **DDR4**.
- En chip (BRAM/URAM) quedan solo las estructuras internas de los cores: ROM de twiddles, líneas de retardo, FIFOs.
- Hay una FIFO en cada frontera donde un core de streaming toca AXI/DDR.

### Presupuesto de tiempo (1024 × 1024, 200 MHz) 🟡

```
FFT2D:      8 bobinas × (512 filas × 1024 + 1024 columnas × 512) = 8,39 M muestras
Unfolding:  1024 × 512 posiciones × 16 MAC complejos            = 8,39 M MAC
A 200 MHz, una muestra o un MAC por ciclo:  41,9 ms / P    y    41,9 ms / M
```

| Configuración | FFT | Unfolding | Total | |
|---|---|---|---|---|
| P = 4, M = 1 | 10,5 ms | 41,9 ms | 52,4 ms | ✗ no entra |
| **P = 4, M = 4** | 10,5 ms | 10,5 ms | **21 ms** | ✓ |

**El unfolding cuesta lo mismo que la FFT**: paralelizar solo la FFT no alcanza. Los DSP no son la restricción: P = 4 y M = 4 son unos 80 de 1248 (6 %).

### Tráfico por trama 🟡

| Tráfico | Tamaño |
|---|---|
| Cuenta inicial, sin los coeficientes | ~82 MiB (~50 MiB con *corner-turn* en chip) |
| Del unfolding: imágenes plegadas (8 bobinas) | 16 MiB |
| **Coeficientes de unfolding (U)** | **32 MiB** = 524 288 × 16 complejos × 4 B |
| Salida (magnitud) | 2 MiB |

- **Los coeficientes son el doble que las imágenes**: 640 MiB/s a 20 fps.
- Formas de bajarlos: guardar S y resolver el 2 × 2 en la PL (16 MiB); submuestrear los mapas, que son suaves, e interpolar; o reducir la precisión de U.
- Con P = 4 y 16 bits, y antes de contar los coeficientes, el sistema usaba ~80 % de un puerto HP y ~37 % de la DDR4. DS987 especifica 4 GB de DDR4 de 64 bits a 2400 Mb/s: **19,2 GB/s de pico teórico** (17,88 GiB/s).
  - La cifra de 17,07 GB/s usada antes coincide con el pico teórico de una DDR4-2133 (2133,3 MT/s × 8 B = 17,07 GB/s). Probablemente es una cuenta con 2133 MT/s y no una medición. ⏳ Verificar a qué velocidad configura la DDR el PS del diseño.
  - El "~37 %" sale de 6,4 GB/s (P = 4 núcleos × 200 MHz × 4 B × lectura + escritura) sobre 17,07. Contra 19,2 GB/s es el 33 %.
  - ⏳ Medir el ancho de banda **agregado** con varios puertos HP en paralelo (§7). El barrido de un solo puerto no dice cuánto da la DDR en total.

### Estrategias de paralelismo (en orden de valor)

1. ***Corner-turn* en chip con URAM** (URAM288 = 4K × 72, dos puertos: se escribe por filas y se lee por columnas en el mismo ciclo). Ahorra una pasada por DDR y evita el acceso con *stride* de 4 KiB. El tile mínimo lo fija N½ (§7): tiene que ser bastante más grande que 3 KiB.
2. **Más puertos HP.** Hay tres caminos independientes al DDRC: Port3 = HP0, Port4 = HP1 + HP2 (compartido), Port5 = HP3 (UG1085). Conviene usar HP0 y HP3, y separar lecturas de escrituras.
3. **Paralelismo por bobina** (instancias independientes).
4. **Radix mayor** (R2⁴SDF, *multipath*): RTL nuevo, queda como última opción.

---

## 3. FFT 1D R2²SDF: diseño

### 3.1 Estructura

- R2²SDF (*radix-2², single-path delay feedback*): pipelined, una muestra por ciclo. Usa la mitad de multiplicadores que un R2SDF, porque una de cada dos etapas solo multiplica por −j.
- Para N = 1024: **5 unidades**, **10 etapas radix-2** y **4 multiplicadores complejos** (log₄N − 1; la última unidad no multiplica).
- **N = 2^impar** (desde `fft_n512`, 2026-10-18): ⌊log₂N/2⌋ unidades (M = N, N/4, …, 8), **todas con multiplicador** por la misma regla `LOG_M != 2`, más **`fft1d_r22sdf_tail`**: una etapa radix-2 de span 1 sin twiddle (la `btfly_bf2i` con DEPTH = 1, el mismo control de etapa y un registro de salida). Para N = 512: 4 unidades más la etapa final, 4 multiplicadores (16 DSP), igual que 1024. La etapa final usa el bit log₂N − 1 de `SCALE_MASK`.
  - Única diferencia de control con las unidades: con DEPTH = 1 la condición de arranque `di_count == DEPTH − 1` también vale en reposo, así que la etapa final exige además `i_valid`.
  - Latencia con log₂N impar: **N + (3 + CMUL_PIPE)·⌊log₂N/2⌋ − 1**, o sea **527** para N = 512 (523 sin registro). Verificada en el vm, ciclo a ciclo, para N = 8, 32, 128 y 512.
- **Latencia: N + (3 + CMUL_PIPE)(log₄N − 1)** hasta el primer `o_valid` ✅: **1040** ciclos con `CMUL_PIPE = 1` (default desde el 2026-10-18) y **1036** con `CMUL_PIPE = 0`. El tb cuenta 2 más por convención de conteo, igual en todas las versiones.
- Salida en orden **bit-reversed**.
- Cada unidad, en orden de tiempo. **Ojo con los nombres:** en el RTL la primera mariposa se llama `BF2II`, al revés que en He & Torkelson.

```
i_din → BF2II (línea M/2, −j trivial) → reg → BF2I (línea M/4) → reg → cmul → reg → o_dout
```

- **Control por etapa.** Cada mariposa tiene su propio contador (`di_count`, `bf1_count`, `bf2_count`), que arranca cuando le llegan los datos.
- **Dirección de twiddle:** `tw_addr = tw_num · tw_sel`. Con `tw_sel = 0`, o sea W⁰ = 1, **se saltea el multiplicador**.
- **Restricción: N potencia de 2 y N ≥ 4** (antes de `fft_n512`: potencia de 4; con N = 512 elaboraba sin error y daba basura). Doble `$fatal` en el RTL y `static_assert` en el modelo; probado con N = 2 y N = 12.

### 3.2 Multiplicador complejo

- 4 multiplicadores reales (`(a+jb)(c+jd) = (ac−bd) + j(ad+bc)`), producto exacto y un solo `cast` a la salida. Es más fácil de verificar bit a bit.
- Karatsuba ❌: WNS −0,918 ns a 200 MHz. El `cmul` directo: 4 DSP, Fmax 211 MHz ✅.
- Operandos con anchos independientes (`NB_IN1`/`NB_IN2`): operando 1 = twiddle, operando 2 = dato.
- DSP48E2 = multiplicador 27 × 18 (UG579): con dato ≤ 27 bits y twiddle ≤ 18 bits, cada producto real usa 1 DSP. **Confirmado en síntesis ✅ (2026-10-17): 16 DSP para la FFT de 1024 con datos de 16, 20 y 27 bits.**
- Cómo lo mapea Vivado (log de síntesis, por `cmul`):
  - `a·c` en un DSP; el siguiente DSP, encadenado por `PCIN`, calcula `(a·c) − b·d`. Lo mismo para `a·d + b·c`. **La suma y la resta quedan adentro de los DSP.**
  - Los registros del dato (`r_bf2_do`) y del twiddle (`r_tw_re/im`) se absorben como AREG/BREG = 1. **MREG = PREG = 0**: el camino registro → multiplicador → suma encadenada → `cast` en fábrica → registro se hace en un ciclo.
- **Parámetro `PIPE` del `cmul`** (entrega `cmul_pipe`, 2026-10-17). En la FFT se controla con `CMUL_PIPE`: **default 1 desde el 2026-10-18** (decisión de Julián; antes era 0). El `PIPE` del `cmul` solo sigue con default 0, porque lo usa también la regresión propia de `modules/cmul`; la FFT se lo pasa explícito.
  - `PIPE = 0`: combinacional, idéntico a antes byte a byte.
  - `PIPE = 1`: registra la suma completa (`NB_FULL` bits) **antes** del `cast`, con reset sincrónico y sin enable, para que Vivado lo absorba como **PREG** del segundo DSP de la cascada. El valor es el mismo; llega un ciclo después.
  - Puertos nuevos `i_clock` e `i_rst` **al final** de la lista; con `PIPE = 0` pueden quedar sin conectar.
  - En la unidad, el *bypass* de W⁰ (`r_bf2_do_d`), su selección (`r_mu_en_d`) y el valid (`r_bf2_do_en_d`) se atrasan el mismo ciclo: 2·NB + 2 FF por unidad con multiplicador.
  - Motivo: el camino crítico después del ruteo era DSP sin PREG → `cast` en fábrica (§5.6).

### 3.3 `cast.v`

- Redondeo *half-up* (`floor(x + ½)`) y saturación.
- Equivale a `ap_fixed<…, AP_RND, AP_SAT>`: 800 324 vectores sin diferencias ✅.
- **No puede agregar bits fraccionales**: con NBF_OUT > NBF_IN devuelve X. Por eso la entrada de la FFT se ensancha por cableado y no con `cast` (⏳ agregar `$fatal`).

### 3.4 Formatos y parámetros

| Parámetro | Rol | Default |
|---|---|---|
| `N` | tamaño (potencia de 2, ≥ 4) | 64 |
| `NB`, `NBF` | datapath interno Q(NB, NBF) | 16, 15 |
| `NB_TW`, `NBF_TW` | twiddles | NB, NB_TW − (NB − NBF) |
| `NB_IN`, `NBF_IN` | puerto de entrada | NB, NB_IN − (NB − NBF) |
| `NB_OUT`, `NBF_OUT` | puerto de salida | NB, NB_OUT − (NB − NBF) |
| `SCALE_MASK` | qué etapas escalan ×½ | `0xFFFFFFFF` (todas) |
| `CMUL_PIPE` | registro antes del `cast` de cada `cmul` (§3.2) | 1 (desde el 2026-10-18) |
| `INVERSE` | IDFT sin normalizar: Σ X[k] e^{+j2πkn/N} por 2^−S, con swap real/imag a la entrada y a la salida | 0 (2026-10-18) |
| `TW_DIR` | carpeta con `tw_re.mem` / `tw_im.mem` | `twiddles` |

- **Regla de los valores por defecto** (la misma en RTL, modelo y tb): si no se fija el ancho, es el del datapath; si no se fijan los fraccionales, el formato **hereda los bits enteros del datapath**. Con datapath Q1.x, todo queda Q1.x.
- **Configuración elegida: `NB = 20`, `NB_TW = 18`, `NB_IN = NB_OUT = 16`, `SCALE_MASK = 784`.** El dato entra en 16 bits, se ensancha a 20 adentro de la FFT, se multiplica por twiddles de 18 y sale con un único `cast` a 16 bits antes de la DDR. 27 no es el valor elegido: es el **techo** que permite el DSP sin agregar bloques, por si la entrada tiene que pasar a 18 bits.
  - Datapath Q1.19, twiddles Q1.17, puertos Q1.15.
  - 16 bits a la entrada porque es lo que viene de la DDR; 18 para el twiddle porque es el puerto B del DSP; 16 a la salida para mantener 4 B por muestra hacia la DDR.
  - Si los datos reales piden entrada de 18 bits, NB puede subir hasta 27 sin agregar DSP.
  - **Por qué no 27 con entrada de 16** (resumen para la tesis): no mejora nada y cuesta.
    - SQNR total: 70,32 dB con NB = 20 contra 70,33 con NB = 27 (§4.3). Lo que falta hasta el techo de 70,50 es el `cast` de salida a 16 bits, y ningún ancho interno lo saca.
    - Implementación: +23 % de LUT (3637 contra 2955) y +22 % de FF (1101 contra 902). Las líneas de retardo pasan de 40 a 54 bits por muestra.
    - Timing: sin `CMUL_PIPE` no cerró a 200 MHz (−0,017 ns, §5.6).

### 3.5 Escalado selectivo (`SCALE_MASK`)

- **El bit *i* corresponde a la etapa radix-2 número *i* en orden de tiempo** (0 = la primera). En la unidad *u*: bit 2u = primera mariposa (`BF2II`), bit 2u + 1 = segunda (`BF2I`).
- La salida es `X[k] = 2^−S · Σ x[n] W^(nk)`, con `S = popcount(SCALE_MASK & (N−1))`.
- **784 = 0b11_0001_0000**: escalan las etapas 4, 8 y 9 (contando desde 0), o sea S = 3 y salida = X/8.
- **De dónde sale 784.** Escalar todas las etapas (las 19 de la 2D) dejaba el pico de la imagen en 0,9 LSB: la señal se perdía (§4.5). Una búsqueda exhaustiva de máscaras de 3 etapas sobre filas de k-space del fantoma encontró que 784 (0x310) maximiza la señal sin desbordar.
- **Ojo:** con ruido blanco de amplitud alta, 784 satura. La segunda pasada de la 2D necesita su propia máscara (⏳).
- En el JSON la máscara va **en decimal**, porque el mismo literal tiene que valer en C++ y en Verilog.

### 3.6 Redondeo por etapa

| Punto | Operación | ¿Redondea? |
|---|---|---|
| Entrada | Q(NB_IN, NBF_IN) → Q(NB, NBF), corrimiento | No: exacto |
| Mariposa que no escala | `a ± b` | No; solo puede saturar |
| Mariposa que escala | `(a ± b)/2`: se descarta 1 bit, *half-up* | Sí. Error ∈ {0, +q/2}: **sesgado** (§4.6) |
| `−j` | intercambio y cambio de signo | No (salvo −(−1), que satura) |
| `cmul` | producto exacto → Q(NB, NBF) | Sí; se saltea con W⁰ |
| Salida | Q(NB, NBF) → Q(NB_OUT, NBF_OUT) | Sí, si NBF > NBF_OUT (`cast` combinacional) |

### 3.7 Twiddles

- `gen_twiddles.py --n N --nb NB_TW [--nbf NBF_TW] --out <dir>` genera `tw_re.mem` y `tw_im.mem`: N valores hex de NB_TW bits, con `W^k = exp(−j2πk/N)`, redondeo y recorte (+1 se recorta a 1 − 2^−NBF_TW).
- Primeras líneas en Q1.17: `tw_re` = `1ffff`, `1fffe`; `tw_im` = `00000`, `3fcdc`.
- Carpetas: `rtl/twiddles/n8`, `n32`, `n64`, `n128`, `n256`, `n512`, `n1024` (Q1.15) y `n512_tw18`, `n1024_tw18` (Q1.17). `gen_twiddles.py` reproduce byte a byte las que ya estaban.
- **`gen_twiddles_r22sdf.py`** (el de Julián, 2026-10-20, mismos argumentos) reproduce byte a byte las 10 carpetas que había, incluida `n1024_q1_17`. Con él se generaron `n8_tw18`, `n16_tw18`, `n32_tw18` y `n64_tw18` para el gtest chico de la `fft2d` (mismos formatos que producción).
- **El ancho del archivo tiene que coincidir con `NB_TW` del caso.** Si no, RTL y modelo leen el mismo error y el vm pasa igual. Solo lo detecta el gtest (§8).

### 3.8 FFT 2D: convenciones (paso 1, 2026-10-18) ✅

Referencia: `model/fft2d_ref.py`. Importa las funciones de `gen_*.py` (las ya verificadas) y comprueba todo con N = 1024, L = 8, Af = 2, Shepp-Logan; también con N = 64 y 256 y con Af = 4.

**Dimensiones** (confirmadas por Julián): por bobina, 512 líneas de phase-encode adquiridas × 1024 muestras de readout. La imagen plegada es de 512 × 1024; la final, de 1024 × 1024, sale del unfolding.

**Convención de los scripts:** `K[l, r, e] = fft2(coils)`, sin fftshift. `gen_aliased_kspace` (eje `y`, default) se queda con e = 0, Af, 2Af, … del último eje. Ojo con los nombres: en `gen_phantom`/`gen_smaps` las imágenes son `[Y, X]` (por `np.mgrid`), así que el eje "Ny" de los scripts, el que se submuestrea, es la **columna X** de la imagen.

**Convención del hardware:**

| Qué | Definición |
|---|---|
| Layout del k-space | `K_hw[l, k', r] = K_alias[l, r, k']`: cada línea adquirida tiene sus 1024 muestras de readout contiguas, que es como llega una adquisición |
| Pasada 1 | IDFT de 1024 sobre r, fila por fila (512 filas) |
| Pasada 2 | IDFT de 512 sobre k', columna por columna (1024 columnas) |
| Salida | imagen plegada `A[l, p, x] = ifft2(K_alias)[l, x, p]`, transpuesta respecto de los scripts |
| Escalado | cada pasada calcula 2^−S·Σ, sin 1/N: **A = 2^(19 − S1 − S2) · ifft2**. Con máscaras completas (S1 = 10, S2 = 9) es exactamente `np.fft.ifft2` |
| IFFT | swap → FFT → swap con el mismo núcleo (§9, verificado) |
| Orden | bit-reversed en el eje de cada pasada: `A_hw[l, j2, j1] = A[l, bitrev9(j2), bitrev10(j1)]` |
| Shifts | ninguno: el hardware calcula la IDFT pura |

**Checks** (todos con error ≤ 10⁻¹⁵ relativo):
- **A. Plegado de Pruessmann:** `ifft2(K_alias)[x, p] = Σ_a c[x, p + a·N/Af]`. Con la normalización de `ifft2` sobre (N, N/Af) **no aparece 1/Af**.
- **B. Flujo del hardware:** swap-FFT, bit-reversed por pasada y *corner-turn* dan la referencia; con máscaras parciales solo cambia la potencia de 2.
- **C. Unfolding SENSE en float**, con U = (SᴴS)⁻¹Sᴴ por posición plegada y pares (p, p + 512): recupera m. Lo mismo trabajando todo en orden hardware, con **U guardada en ese orden** y solo la escritura final reordenada. Conclusión: **el datapath no necesita hardware de reordenamiento.**
- **D. k-space centrado** (fastMRI, `gen_kspace --fftshift`): la imagen plegada es (−1)^(x+p) por la no centrada, y **|m̂| es idéntico**. Como la salida es en magnitud, centrar o no da igual y el hardware no hace shifts.
- Con L = Af = 4 y estos mapas, SᴴS es singular en algunas posiciones; con L = 8 no pasa (Af = 2 y 4).

**Rango dinámico** (pico/rms por componente; para el paso 4):

| Punto | 8 bobinas juntas | Por bobina |
|---|---|---|
| k-space | 54,3 dB | 52,3 a 52,6 dB |
| Tras la pasada 1 (readout) | 38,4 dB | 26,3 a 36,6 dB |
| Imagen plegada | 20,6 dB | 17,7 a 20,8 dB |

El "29 dB tras la primera pasada" de §4.3 era por bobina y es consistente con este rango. Lo que importa para elegir las máscaras es si la normalización de entrada es común a todas las bobinas (entonces cuenta la columna "juntas") o por bobina; eso se define en el paso 4.

---

## 4. Estudio de precisión

### 4.1 Métrica

**SQNR** (*signal-to-quantization-noise ratio*): cuánto más potente es la señal que el error que introduce la aritmética finita.

```
SQNR = 10 · log10( Σ |X_ref[k]|² / Σ |X_hw[k] − X_ref[k]|² )   [dB]
```

- `X_ref` es la salida ideal (en `double`); `X_hw` es la salida en punto fijo.
- Cuantizar uniformemente con N bits una senoidal a fondo de escala da `6,02·N + 1,76 dB`, o sea ~6 dB por bit. Verificado en NumPy: 49,93 / 74,03 / 98,09 dB con 8, 12 y 16 bits, contra 49,92 / 74,00 / 98,08 de la fórmula.
- Si la señal está *A* dB por debajo del fondo de escala, la SQNR baja *A* dB, porque el error es el mismo y la señal es más chica. Por eso importan el PAR del estímulo y el escalado.

**SNR contra SQNR.** El SNR mide la señal contra el ruido **físico** de la adquisición: térmico, del paciente y de la electrónica de las bobinas, gaussiano en k-space. La SQNR mide la señal contra el error que agrega **nuestra aritmética**. En SENSE el SNR además cae por el submuestreo: `SNR_SENSE = SNR_full / (g·√R)`, con `g ≥ 1` el factor geométrico (Pruessmann 1999; la notación exacta hay que verificarla en el paper). Como los dos ruidos son independientes, sus potencias se suman:

```
1 / SNR_total = 1 / SNR_térmico + 1 / SQNR      (en potencia, no en dB)
```

Este modelo reproduce la tabla de ruido térmico de §4.4 con una diferencia de 0,2 dB o menos. Con SQNR = 44,16 dB (entrada de 16 bits en 2D) predice pérdidas de −0,02 / −0,07 / −0,40 / −1,41 / −4,03 dB para SNR térmico de 20 / 26 / 34 / 40 / 46 dB; las medidas fueron −0,02 / −0,06 / −0,37 / −1,32 / −3,85. **Lo que importa es el margen entre los dos:** si la SQNR está 20 dB por encima del SNR térmico se pierden 0,04 dB; con 10 dB de margen, 0,41 dB; con los dos iguales, 3 dB.

Las dos variantes de SQNR usadas:

- **SQNR total**: contra la señal verdadera, sin cuantizar. Incluye la cuantización de la entrada.
- **SQNR del pipeline**: contra la FFT ideal de la entrada **ya cuantizada**. Mide solo lo que agrega la FFT.
- Estímulo base del estudio 1D: fila del fantoma, N = 1024, máscara 784, entrada de 16 bits.

### 4.2 Mapa datos × twiddle (SQNR del pipeline, dB) ✅

Modelo bit-exacto, fila del fantoma, máscara 784:

| datos ↓ / twiddle → | 12 | 14 | 16 | **18** | 20 | 24 |
|---|---|---|---|---|---|---|
| 16 | 68,6 | 71,5 | 72,2 | 72,2 | 72,1 | 72,0 |
| 18 | 71,4 | 80,5 | 84,1 | 84,5 | 84,3 | 84,4 |
| 20 | 71,6 | 83,0 | 93,2 | 96,4 | 96,4 | 96,4 |
| 22 | 71,6 | 83,3 | 95,5 | 105,8 | 108,2 | 108,3 |
| 24 | 71,6 | 83,3 | 95,8 | 108,7 | 116,8 | 120,6 |
| 27 | 71,6 | 83,3 | 95,8 | **108,9** | 119,5 | 137,2 |

- **Hay dos techos independientes y manda el menor**: uno lo fijan los datos y otro el twiddle. Cada uno sube ~6 dB por bit.
- **El twiddle necesita ~4 bits menos que el dato** para aportar el mismo ruido. La explicación está en la cota de §4.7:
  - El error del dato es **absoluto** y se amplifica 2^(r−s) aguas abajo.
  - El error del twiddle es **relativo a la señal**, y la señal está atenuada aguas arriba exactamente en ese mismo factor.

### 4.3 ¿Full precision adentro y redondear solo al final?

**Pregunta** (2026-10-10): hacer toda la FFT sin redondear y redondear una sola vez antes de la DDR.

**Literal no entra** 🟡. Sin redondear nunca, el ancho crece así por componente:

| Punto | Bits |
|---|---|
| Entrada | 16 |
| Tras la FFT de 1024 | 90 |
| Tras la FFT de 512 | 147 |
| Tras el unfolding | 167 |

Son **818 MiB por trama, 17,15 GB/s**: el 89 % del pico teórico de la DDR4 (19,2 GB/s, DS987) solo para mover datos de la FFT. En la práctica una DDR4 no sostiene ese porcentaje, y faltaría el resto del sistema. Además, el DSP es de 27 × 18.

**Y aunque entrara, no serviría: la pared es la entrada.** SQNR total con entrada de 16 bits ✅:

| Datapath | SQNR total |
|---|---|
| 16 bits (todo) | 68,23 dB |
| 18 bits | 70,41 dB |
| 20 / 24 / ∞ | 70,50 dB (techo de la entrada de 16 b) |
| *Contraste: 24 b con entrada de 24 b* | *116,47 dB* |

**Versión práctica, la que quedó implementada** ✅: datapath de 20 bits, twiddle de 18, puertos de 16, con el único redondeo hacia 16 bits en el `cast` de salida.
- **De 68,23 a 70,32 dB**, a 0,18 dB del techo.
- Con puertos de 16 bits, NB = 20, 24 o 27 dan lo mismo (70,32–70,33 dB). Sin el redondeo de salida da 70,50.

**En 2D, el ancho de almacenamiento da igual** 🟡 (NumPy, pipeline ideal, 1024 × 512, una bobina):

| Intermedio en DDR | Imagen | SQNR |
|---|---|---|
| 16 b | 16 b | 44,15 dB |
| 32 b | 32 b | 44,16 dB |
| ∞ | ∞ | 44,16 dB |

El límite es la entrada. El k-space tiene un PAR de 53 dB y, al normalizar el pico a 0,9, el 2,4 % de las muestras queda debajo de 1 LSB. Después de la primera pasada el PAR baja a 29 dB y el intermedio se cuantiza sin problema.

**Conclusión:**
- El ancho interno y el de almacenamiento son variables independientes.
- Conviene ensanchar adentro (gratis en DSP) y guardar en 16 bits.
- Lo que decide la calidad final es el ancho de la **entrada**, y eso lo define el SNR de los datos reales (§4.4).

### 4.4 Ruido térmico: cuándo importa la cuantización 🟡

Agregando ruido térmico al fantoma (NumPy), pérdida de la imagen por cuantizar la entrada:

| SNR de la adquisición | Pérdida con entrada de 16 b | Con 18 b | σ del ruido en LSB |
|---|---|---|---|
| 20 dB | −0,02 dB | 0,00 | 4,8 |
| 26 dB | −0,06 dB | 0,00 | 2,4 |
| 34 dB | −0,37 dB | −0,02 | 1,0 |
| 40 dB | −1,32 dB | −0,10 | 0,5 |
| 46 dB | −3,85 dB | −0,37 | 0,24 |

**Regla: la cuantización solo importa si el ruido térmico queda por debajo de ~1 LSB.** Para las curvas de SNR contra bits alcanza la referencia en float; con datos reales, la referencia que cuenta es el piso de ruido de la adquisición. Hay que medirlo en fastMRI (⏳): si el SNR da menos de ~30 dB, 16 bits sobran; si da 40 dB o más, la entrada pasa a 18 bits.

### 4.5 Escalado: historia

- Con ½ en todas las etapas (19 en la 2D), el pico de la imagen del Shepp-Logan quedaba en **0,9 LSB**: señal destruida.
- El crecimiento real fue de +3,87 bits pico, así que 4 de las 19 etapas escalando alcanzan (pico a −0,13 bits del fondo de escala). 🟡
- Se agregó `SCALE` por mariposa (`NBF_SUM = NBF + SCALE`) y `SCALE_MASK` en el top. Para la 1D de 1024 se eligió 784 (§3.5).

### 4.6 Sesgo del redondeo *half-up* ✅ (2026-10-17)

**Qué se ve.** En el ploteo de fase, los bins sin señal solo toman los valores 0, π/4 y π/2: el residuo nunca es negativo. Medido sobre la salida del modelo (Q1.15, multitono, N = 1024), en los bins sin tono:
- la parte real vale 0 o +1 LSB;
- la parte imaginaria vale 0, +1 o +2 LSB.

**Causa.** La mariposa que escala calcula `(a ± b)/2` y descarta un bit. Si el bit descartado es 1, el valor queda **exactamente** en el empate, y *half-up* siempre sube. El error está en {0, +q/2}, con media +q/4 por etapa escalada. En la última etapa: −½ → 0 y +½ → +1.

Error contra la FFT ideal de la entrada cuantizada:

| Caso | Sesgo (Re / Im) | rms | Sesgo² / MSE |
|---|---|---|---|
| Q1.15, multitono, N = 1024 | +0,37 / +0,40 LSB | 0,88 | 39 % |
| Q1.15, ruido, N = 1024 | +0,51 / +0,48 | 1,01 | 49 % |
| Q1.15, ruido, N = 64 | +0,59 / +0,50 | 1,03 | 57 % |
| 20/18, ruido | +0,06 / +0,06 | 0,42 | 4 % |
| 20/18 con 784, multitono | +0,05 / +0,05 | 0,43 | 3 % |

**Implicancias:**
- **En 16 bits, casi la mitad de la potencia de error es sesgo.** El datapath de 20 bits casi lo elimina: los redondeos internos son 16 veces más chicos y en el `cast` final (4 bits descartados) el empate exacto es raro. Parte de la mejora de los 20 bits viene de acá.
- **En la imagen, el sesgo es un offset complejo** de algunas décimas de LSB en todos los píxeles: sube el fondo. Con dos pasadas se acumula.
- **Medido en 2D (2026-10-21, fantoma 1024 × 512, 801 / 384, `run_plot_fft2d.py --full-check`):** el sesgo **no** queda repartido. La salida de la pasada 1 trae un sesgo de ≈ +0,055 LSB por muestra. Para la pasada 2 es una constante a lo largo de cada columna, y la IDFT de una constante es un delta en p = 0. Resultado: en la fila p = 0 el error medio es **+7,5 LSB** en Re y en Im (≈ 512 / 2^S2 × 0,055 = 128 × 0,055 más el sesgo propio de la pasada 2), contra +0,055 en el resto. Después del unfolding, la fila p = 0 cae en y = 0 y en y = N/2: se ve como una línea horizontal en el mapa de error de la imagen. Es un argumento más para el redondeo convergente.
- **Ida y vuelta (2026-10-25, prueba rápida con `fft1d_r22sdf_fxp.py`, N = 256, 20/18/16):** imagen → FFT 2D con escalado completo → IFFT 2D. Con escalado completo también en la inversa, la ganancia total es N1·N2 / 2^(S_ida + S_vuelta) = 2^16 / 2^32 y la imagen desaparece (pico 1 LSB). Con la inversa sin escalar (ganancia 1) vuelve la imagen, con el SQNR de la FFT directa (Shepp 36,5 dB, seno rotado 49,3, anillos 53,3, *zone plate* 48,7), **pero el sesgo del espectro (≈ +0,06 LSB de media) se convierte en un pico en el píxel (0, 0)**: error ≈ N² × (media del error del espectro) ≈ 65 536 × 0,062 ≈ +4 050 LSB en Shepp (coincide al LSB). En el seno y la *zone plate* ese píxel satura. Es el mismo mecanismo que la fila p = 0 de arriba, llevado a 2D sin escalado que lo atenúe.
- **Se corrige con redondeo convergente** (*round-half-even*). Al descartar un bit: `resultado = m + (bit_descartado AND lsb(m))`, una compuerta por mariposa. UG579 describe cómo hacerlo en el DSP48E2 con el *pattern detector*, pero las mariposas están en fabric, así que acá serían LUTs.
- ⏳ Experimento pendiente: agregar un modo convergente al `cast` y al modelo, y medirlo. Cambia la salida bit a bit.

### 4.7 Cota de peor caso del error de salida ✅

Para usar como tolerancia del gtest (⏳ reemplazar el `5e-3` fijo).

El error de salida es la suma de los errores de cada redondeo, propagados por el resto de la FFT, que es lineal. Para un punto *p*:
- `r_p` = etapas de mariposa aguas abajo, y `s_p` = cuántas de ellas escalan;
- `c_p` = √2 si queda algún multiplicador no trivial aguas abajo (mezcla Re/Im); 1 si no;
- ganancia por componente: `g_p = c_p · 2^(r_p − s_p)`, porque cada salida combina 2^r valores con coeficientes de módulo 2^−s.

| Fuente | Error inyectado | Aporte a la salida |
|---|---|---|
| Mariposa que escala | ∈ [0, q_d/2] | (q_d/2) · g_p |
| Redondeo del `cmul` | ≤ q_d/2 | (q_d/2) · g_p |
| Twiddle cuantizado | ≤ X_p · q_tw/√2 | (q_tw/√2) · A_out · c_p |
| `cast` de salida | ≤ q_out/2 | q_out/2 |

- Sin desborde, `X_p ≤ A_out · 2^−(r_p − s_p)`, así que **el término del twiddle no depende de la máscara**, mientras que el del dato crece como 2^(r−s).
- Con 784, el redondeo del primer `cmul` sale multiplicado por 2^(8−3) = 32.

Verificación con Monte Carlo: ruido, tono, DC con fase y espectro aleatorio; A_out = 0,65 y 0,9; 60 tramas por configuración (400 en N = 64). Nunca se superó la cota; el margen mínimo fue de 2,4 veces.

| Configuración (A_out = 0,65) | Cota B (LSB) | Máximo medido (Monte Carlo o gtest) |
|---|---|---|
| Q1.15, N = 64 | 6,15 | 2,60 |
| Q1.15, N = 1024 | 11,69 | 3,27 |
| 20/18, máscara completa | 1,68 | 0,66 |
| 20/18, 784 | 3,49 | 0,74 |
| 20/18, máscara 0 | 16,08 | 3,01 |
| 16 bits, 784 | 40,68 | 5,05 |
| 27/18, 784 | 1,12 | 0,51 |

*Corrección del 2026-10-17:* la primera versión de la cota tenía invertido el orden de las dos mariposas dentro de cada unidad. Corregida, la de 784 sube de 3,32 a 3,49 y la de 16 bits con 784 de 37,9 a 40,7. Las conclusiones no cambian.

### 4.8 Máscaras de la FFT 2D con el fantoma (paso 4, 2026-10-18) 🟡

**Herramientas.**
- `model/fft1d_r22sdf_fxp.py`: el núcleo en aritmética entera, vectorizado (todas las filas de una pasada en una llamada, ~1,4 s para 8 bobinas). **Bit a bit igual al RTL y al modelo C++**: 35/35 casos de la regresión y 16 casos de estrés con máscaras al azar y saturación fuerte (RTL = C++ = Python). Se revalida con `python3 model/fft1d_r22sdf_fxp.py --validate modules/fft1d_r22sdf` después de `run_regression_sim`.
- `model/fft2d_mask_study.py`: k-space → normalización → Q1.15 → pasada 1 (IFFT 1024) → Q1.15 → pasada 2 (IFFT 512) → Q1.15 → unfolding SENSE en float. Mide la SNR de la imagen desplegada contra el fantoma, los techos, las saturaciones y el margen (20 log₁₀(1/pico) del peor punto interno).

**Setup:** Shepp-Logan 1024, L = 8, Af = 2, núcleo 20/18/16, sin ruido térmico.

**Picos sin escalar por etapa** (dB respecto del fondo de escala, entrada normalizada a 0,9):

| Pasada | Etapas 0 → final |
|---|---|
| 1 (1024) | −0,9 −0,9 −0,9 −0,9 −0,8 −0,1 +1,5 +4,3 +7,8 +13,3 |
| 2 (512, sobre la salida de la pasada 1 con S1 = 3) | −4,7 −4,7 −4,6 −4,6 −4,4 −4,2 −3,9 +1,2 +4,6 |

El k-space concentra la energía en pocas muestras, así que las primeras etapas no crecen y casi todo el crecimiento (×4,6 en la pasada 1) está en las últimas. **Hacen falta S1 ≥ 3 y S2 ≥ 1**: con menos satura siempre (barrido exhaustivo).

**Resultados** (SNR de la imagen desplegada, dB):

| Máscaras (m1 / m2) | S1 / S2 | Normalización común | Por bobina | Margen p1 / p2 (por bobina) |
|---|---|---|---|---|
| 1023 / 511 (todas escalan, el default 1D) | 10 / 9 | 1,5 | 1,9 | — |
| 784 / 511 (el 784 de la FFT 1D) | 3 / 9 | 35,7 | 37,7 | — |
| 832 / 128 (lazy, θ = 1) | 3 / 1 | 50,00 | — | 0,1 / 1,5 (común) |
| 641 / 384 | 3 / 2 | 49,98 | 51,88 | 4,3 / 1,9 |
| **801 / 384 (propuesta)** | **4 / 2** | **49,12** | **51,03** | **6,5 / 7,9** |
| techo (a): solo cuantizar la entrada | | 50,33 | 52,22 | |

**Conclusiones (provisorias, con el fantoma):**
- **Lo que decide es S1, S2 y el margen, no la ubicación.** Con el mismo S y sin saturar, la imagen cambia menos de 0,03 dB (barrido exhaustivo: 375 máscaras de la pasada 1 y 46 de la pasada 2). La aritmética de 20 bits cuesta ~0,02 dB y el almacenamiento en 16 bits ~0,3 dB; **el techo es la cuantización de la entrada**.
- **Las máscaras de la FFT 1D no sirven para el 2D:** la completa deja la imagen en el ruido (1,5 dB) y el 784 con la pasada 2 completa da 35,7 dB.
- **Normalizar por bobina da +1,9 dB gratis:** las bobinas débiles usan más bits. La ganancia g_l se compensa en U (U'[a, l] = U[a, l] / g_l), que ya precalcula el PS.
- **Propuesta provisoria: m1 = 801 (etapas 0, 5, 8, 9), m2 = 384 (etapas 7, 8), normalización por bobina.** 51,0 dB con más de 6,5 dB de margen en las dos pasadas. Cuesta 0,85 dB frente a la opción sin margen (641/384), que con datos reales (ruido térmico de 30–40 dB) no se nota: con 20 dB entre ruido y cuantización, la pérdida total es de ~0,04 dB (§4.1).
- **`SCALE_MASK` es un parámetro de síntesis**, no se puede cambiar por estudio. Por eso importa el margen, y la decisión final se toma con datos reales (fastMRI).
- **Timing de N = 512** (2026-10-20, máscara completa): cierra con +0,192 ns, contra +0,903 ns de N = 1024 con 784. Camino crítico confirmado con `timing_summary_impl.rpt` (ver §5.6): la BF2I **escalada** de la unidad 2 (etapa 5) suma dos carry chains en serie, la suma y el `+BIAS` del redondeo, antes de la saturación y de la entrada A del DSP. La segunda cadena explica la diferencia: suma unos 0,82 ns, y la diferencia de WNS es de 0,71 ns.
  - **Corrección:** las máscaras propuestas **sí** escalan una BF2I que alimenta un DSP: 801 en 1024 escala la etapa 5 (unidad 2, M = 64) y 384 en 512 escala la etapa 7 (unidad 3, M = 8). Con 784 no pasa porque escala solo la etapa 4 (primera mariposa) y la unidad 4, que no tiene multiplicador. Hay que esperar un WNS del orden de +0,2 ns con las máscaras finales: cumple 200 MHz, pero sin el colchón de 0,9 ns. ⏳ Agregar casos de implementación con 801 (1024) y 384 (512).
- **Opción a evaluar:** máscara configurable en tiempo de ejecución (un registro escrito por el PS por sesión, como U). Cuesta un mux por mariposa y por componente.

### 4.9 MAC de desplegado: escalas, formato de U'' y acumulación (2026-10-21) ✅ fantoma

`model/mac_unfold_study.py`, con el fantoma 1024 × 1024 (L = 8, Af = 2) y la salida **bit-exacta** de la `fft2d` (801 / 384, normalización por bobina).

**(1) Escalas.** La entrada es K_q,l = Q15(g_l·K_l). La `fft2d` da A_l = 2^−S·Σ K_q,l·e^{+j…} = 2^(19−S)·ifft2(K_q,l), con S = S1 + S2 y 19 = log2(N1·N2). Como ifft2(K_l) = y_l (imagen plegada, sin 1/Af), queda y_l = 2^(S−19)·A_l / g_l y

  m̂[x, p + a·N/Af] = Σ_l U[x,p,a,l]·y_l = 2^(S−19) · Σ_l U'[x,p,a,l]·A_l,  U' = U / g_l.

Con un exponente común por sesión, U'' = U'·2^−E con las componentes en [−1, 1): **m̂ = 2^(S−19+E) · Σ_l U''_l·A_l**. Verificado contra el camino del estudio de máscaras: error relativo 5,5·10⁻¹⁶, misma SNR (51,03 dB). Con el fantoma, E = 16 y el factor final es 2^3. El 1/g_l (relación 1,42 entre bobinas) y el 2^(S−19+E) no cuestan nada en hardware: el primero va en U'' (lo calcula el PS) y el segundo es un corrimiento.

**(2) Formato de U''.** El máximo por posición plegada varía solo 5,6 dB (percentil 1 contra el máximo): **alcanza un E común por sesión**; un exponente por posición ganaría menos de un bit.

| NB_U | SNR imagen | Pérdida | SNR del error de U solo | Bits del acumulador usados / cota |
|---|---|---|---|---|
| 10 | 48,41 | 2,63 | 51,8 | 23 / 30 |
| 12 | 50,82 | 0,22 | 63,9 | 25 / 32 |
| 14 | 51,02 | 0,014 | 75,9 | 27 / 34 |
| **16** | **51,03** | **0,001** | 88,0 | 29 / 36 |
| 18 | 51,03 | 0,000 | 100,0 | 31 / 38 |

El error de U solo sube 6,02 dB por bit, como tiene que ser. **Propuesta: NB_U = 16** (re + im en una palabra de 32 bits; entra en el puerto de 27 bits del DSP48E2 con A_l de 16 bits en el de 18). En memoria: Af·L·T = 8 388 608 complejos = **32 MiB**, leídos una vez por trama: **671 MB/s** a 20 fps.

**(3) Acumulación.** Producto complejo de 16 × NB_U bits y suma de 8 bobinas: cota de peor caso **NB_U + 20 bits** (36 con NB_U = 16; los datos usan 29). Refinada el 2026-10-21: la cota exacta es NB_A + NB_U + 3 = **35 bits** (cada parte de un producto complejo es < 2^31 porque +2^15 no es representable; por 8 bobinas, < 2^34). Entra con margen en el acumulador de 48 bits del DSP48E2. La suma en enteros es exacta, así que **el resultado no depende del orden de las bobinas** (verificado: natural, inverso y permutado, idénticos): el modelo del MAC se puede escribir sin haber elegido la memoria.
- Si la suma parcial se guarda redondeada (acumular bobina por bobina con el acumulador en memoria), con punto binario fijo para la cota y sin saturación **sigue sin depender del orden**: redondear a una grilla conmuta con sumar múltiplos de la grilla. Lo que cuesta es precisión: W = 20 bits pierde 1,83 dB, **W = 24 pierde 0,011 dB**, W = 28 nada. Ese acumulador (Af·T complejos de 24–32 bits) es un dato para el paso 6.
- Corrección: antes había dicho que redondear las sumas parciales haría depender el resultado del orden. Los números muestran que no, mientras no sature.

**⚠ Válido para el fantoma.** Sus mapas son lisos y nunca se anulan: cond(SᴴS) entre 2,6 y 5,8, g-factor medio 1,24 y máximo 1,41. Con mapas reales (ESPIRiT, que se anulan fuera del objeto) U puede crecer mucho donde SᴴS está mal condicionada: hará falta regularización (Tikhonov, Paul & Mathew cap. 2) o enmascarar, y repetir E y NB_U con fastMRI.

**Código que va al PS** (U'' por sesión: SᴴS, LDLᴴ, sustituciones, /g_l, E, cuantización y empaquetado en el orden del hardware). Propuesta de infraestructura, a confirmar cuando se empiece:
- Un "módulo de software" dentro de la infraestructura de siempre (p. ej. `modules/ps_unfold/` con `cpp/` y `testbench/`, sin RTL): `run_gtest` y el JSON andan sin cambios, y la aplicación de Vitis agrega esos `.cpp`.
- C++17 portable: sin `ap_fixed`, sin excepciones en lo que va al PS, buffers provistos por quien llama, `double`.
- El mismo código compila nativo (gtest, y genera el U'' que usa el modelo del MAC) y para el A53: lo que recibe el PL es lo que se probó.
- Bit-exacto entre x86 y A53: `-ffp-contract=off` explícito y sin `-ffast-math` en los dos (el A53 tiene FMA y el compilador puede fusionar a·b + c), y nada de funciones trascendentes de libm en lo que alimenta la cuantización. Se verifica volcando el buffer por JTAG y comparando bit a bit.
- Primero la referencia en NumPy (`model/`), como siempre. Supuesto: PS en *bare-metal* (standalone), como el bring-up.

### 4.10 MAC, punto 4: U en el PS por L D Lᴴ y el buffer para el PL (2026-10-21) ✅ fantoma

`model/unfold_ref.py` (referencia del código del PS, con self-test) y `model/mac_output_study.py`, sección A a C.

**Algoritmo**, por posición plegada (x, p), con S[l, a] = s_l[x, p + a·N/Af] (L × Af):
- A = SᴴS + λI (hermitiana definida positiva; λ = 0 es Pruessmann con Ψ = I).
- A = L D Lᴴ, por columnas, sin raíces: d_j = A_jj − Σ_{k<j} |L_jk|² d_k y L_ij = (A_ij − Σ_{k<j} L_ik d_k conj(L_jk)) / d_j. A es definida positiva si y solo si todos los d_j > 0: es el chequeo de validez.
- U = A⁻¹Sᴴ resolviendo, por bobina, L z = conj(S[l, :]), w = D⁻¹z, Lᴴu = w.
- Para Af = 2: 328 flops + 2 divisiones + 32 `rint` por posición; **172 Mflop por sesión** (estimación de tiempo en el A53: menos de un segundo; a medir).

**Regla de la referencia:** todo con re / im separados y en el orden de operaciones del futuro C (sumas secuenciales sobre l, 1/d y 1/g una vez, potencias de 2 con `ldexp`, `rint` al par). Así el C nativo se puede exigir **bit a bit**.

**Verificado:**
- L D Lᴴ = A y U = `np.linalg.solve` con error relativo 10⁻¹⁵ a 10⁻¹⁶, U·S = I, para Af = 2, 3 y 4, con y sin λ.
- A tamaño completo: U'' cuantizado **idéntico** (0 de 16 777 216) al que sale de cuantizar el U de `solve`.
- Pivotes del fantoma entre 0,505 y 1,0 (mapas normalizados).

**Mapas que se anulan (tipo ESPIRiT; soporte del fantoma dilatado 8 píxeles).**
- 25 % de las posiciones tienen los dos píxeles plegados afuera, 58 % uno adentro y uno afuera, y 16,6 % los dos adentro. **Sin regularizar, L D Lᴴ es válida solo en el 16,6 %**: en el resto d = 0 y se divide por cero.
- Con Tikhonov, λ = μ·(mediana de diag SᴴS, que vale 1 con mapas normalizados), queda U_λ·S = I − λ(SᴴS + λI)⁻¹ (verificado a 10⁻¹⁵): el sesgo relativo es del orden de λ / d_min.

| μ | SNR adentro del soporte | E |
|---|---|---|
| 10⁻⁶ | 54,78 dB | 15 |
| 10⁻⁴ | 54,76 dB | 15 |
| 10⁻³ | 53,53 dB | 15 |
| 10⁻² | 39,50 dB | 15 |

- Referencia con mapas completos: 53,92 dB adentro. Enmascarar **mejora** (afuera m̂ = 0 exacto y desaparece el aliasing que venía de afuera), y con mapas normalizados el rango de U no explota.
- **Propuesta: λ = 10⁻⁴·escala siempre** (λ > 0 hace que no haya divisiones por cero en ningún caso).
- ⚠ El fantoma no tiene ruido: con datos reales λ también regula la amplificación de ruido (g-factor). Elegirlo con fastMRI.

**Interfaz PS → PL (provisoria hasta el paso 6):**
- **k-space de entrada:** Q1.15 = sat(rint(g_l·K)). Las g_l son fijas por sesión: entran en U''.
- **Buffer U''.** Palabras de 32 bits {im[15:0], re[15:0]} (como los puertos de la FFT), Q1.15.
  - Contenido: U' = U·(1/g_l)·(1/K_n), con E por `frexp` del máximo de las componentes (exacto, sin `log2`); E = 15 en el fantoma con 1/K plegado.
  - Orden: el del **stream de la fft2d**, o = t1·N2 + t2 ↔ (x = br(t1), p = br(t2)).
  - Layout `pos_coil_a` ([o][l][a], lo de una posición junto) o `coil_pos_a` ([l][o][a]); se elige en el paso 6.
  - Af·L·T = 8 388 608 palabras = **32 MiB**. La descripción del proyecto dice ~8 MB, pero eso era para 512 × 512.
- **Registro D** (6 bits), ver §4.11.
- Verificado: el MAC con el buffer empaquetado y el stream en orden hardware da **exactamente** el mismo P que en orden natural, para los dos layouts.

### 4.11 MAC, punto 5: salida y CORDIC (2026-10-21) ✅ fantoma

`model/mac_cordic_fxp.py` (modelo bit-exacto del PL después de la fft2d, con self-test) y `model/mac_output_study.py`, sección D.

**Cadena:**
1. P = Σ_l U''_q·A_l, exacto (35 bits como máximo; usa 29).
2. M = sat_{17+G}(redondeo half-up de P / 2^(D−G)): G bits de guarda.
3. CORDIC vectoring sobre (|Re M|, |Im M|) con n iteraciones y corrimientos aritméticos: si y ≥ 0, x += y>>i y y −= x>>i; si no, al revés. x_n = K_n·|M|, con K_n = Π sqrt(1 + 2^−2i).
4. Salida u16 = redondeo de x_n / 2^G.

**Escala:** m̂ = 2^(S−19+E)·P·2^−30, así que **salida ≈ |m̂|·2^F_OUT con D = 49 − S − E − F_OUT** (D = 13 con S = 6, E = 15, F_OUT = 15: UQ1.15; el máximo del fantoma da 32 938, o sea 1,005). D lo fija el PS por sesión; con datos reales la escala de la imagen no se conoce de antemano y hay que calibrarla (prescan o primera trama).

**Ganancia del CORDIC:** 1/K_n se pliega en U'' (lo multiplica el PS), así que no hay multiplicador en el PL. E baja de 16 a 15.

**Error del CORDIC:**
- Residuo angular ≤ |v|·(1 − cos(atan 2^−(n−1))) ≈ |v|·2^−(2n−1): 1 LSB a fondo de escala con n = 8, 0,008 LSB con n = 12.
- Cada corrimiento trunca: el error crece con n y baja con G.

| n | G | Error máx (LSB) | rms | Media | SNR imagen |
|---|---|---|---|---|---|
| 8 | 0 | 7,17 | 2,38 | +1,31 | 51,47 |
| 8 | 4 | 1,37 | 0,34 | +0,01 | 51,97 |
| 10 | 4 | 1,00 | 0,33 | +0,14 | 51,95 |
| 10 | 5 | 0,73 | 0,30 | +0,06 | 51,97 |
| **12** | **5** | **0,81** | **0,31** | +0,09 | **51,96** |
| 16 | 0 | 15,17 | 7,35 | +5,69 | 50,34 |

Referencias: |m̂| exacta 51,99 dB; el redondeo ideal a u16 tiene rms 0,289 LSB. **Propuesta: n = 12, G = 5** (error < 1 LSB, residuo angular despreciable aun a fondo de escala), M de 22 bits y unos 24 sumadores de ~23 bits.

**La magnitud sube la SNR** de 51,03 dB (m̂ complejo) a 51,99 dB: con un objeto real, descarta la componente del error en cuadratura, donde |m| ≫ ruido. En el fondo (m = 0) queda |ruido| ≈ 30 LSB de u16 (distribución de Rayleigh), así que la salida de 16 bits resuelve el piso de ruido.

**La imagen sale en el orden del stream** (bit-reversed en x y en p). Para mostrarla hay que reordenar: en el PS o con direcciones en el escritor (el *scatter* bit-reversed le pega a la eficiencia de las ráfagas en DDR). Es del paso 6.

### 4.12 Ida y vuelta FFT 2D → IFFT 2D: reparto de la escala (2026-10-25) 🟡

Estudio con `fft1d_r22sdf_fxp.py` (20/18/16, N = 256, imágenes de `gen_fft2d_stim.py`, reordenamiento ideal entre las dos transformadas). Script: `model/fft2d_roundtrip_study.py` (`-N`, `--images`; reproduce la tabla de abajo). Todavía no hay casos de regresión.

**Lo que hay en cada frontera entre pasadas se conoce exacto** (sin contar el redondeo). Pasadas 1 y 3 van por filas, 2 y 4 por columnas; S_i = etapas que dividen en la pasada i:

| Después de | Contenido |
|---|---|
| pasada 1 (FFT de filas) | FFT de cada fila / 2^S1 |
| pasada 2 (FFT de columnas) | FFT 2D / 2^(S1+S2) |
| pasada 3 (IFFT de filas) | N · FFT de cada columna / 2^(S1+S2+S3) |
| pasada 4 (IFFT de columnas) | N² · imagen / 2^(S1+S2+S3+S4) |

Para que vuelva la imagen, S1 + S2 + S3 + S4 = 2·log2 N (16 con N = 256). En cada frontera el pico tiene que quedar por debajo de 1, y eso da el **mínimo acumulado de S** de cada frontera. Los picos medidos coinciden con los predichos al tercer decimal.

**Regla que funciona: dividir lo justo, y al principio de cada pasada.** Dividir al final satura: la primera mariposa suma dos píxeles (n y n + N/2) y, si los dos están cerca de 0,9, da 1,8 (medido: pico 1,80 después de la etapa 0 de la pasada 1 sin escalar, en Shepp).

| Imagen | S1..S4 mínimos | Completa / sin escalar | Mínimo, divide al principio | Mínimo, divide al final |
|---|---|---|---|---|
| Shepp | 6, 7, 1, 2 | 36,5 dB | **54,3 dB** | 13,3 (satura) |
| seno rotado 25° | 7, 8, 0, 1 | 49,3 | **53,9** | 9,3 (satura) |
| anillos | 8, 7, 1, 0 | 53,3 | **58,0** | 58,2 |
| *zone plate* | 4, 4, 4, 4 | 48,7 | **91,2** | 1,0 (satura) |

- El mínimo depende de la imagen. En hardware la máscara es fija salvo que se haga configurable (§9, paso 4). Para SENSE solo importa la inversa (§4.8). La ida y vuelta es un caso de prueba.
- **El pico en el píxel (0, 0) sigue** con el reparto mínimo (Shepp: 729 LSB de módulo, contra 5 700 con completa / sin escalar; seno: 3 448, con 2 saturaciones). Es el sesgo *half-up* (§4.6). Es la medida para el redondeo convergente.
- Reordenamiento entre las dos: la salida de la `fft2d` sale en el orden o = t1·N2 + t2, con kx = br(t1) y ky = br(t2). La entrada de la inversa va en orden natural por filas: entrada[ky·N1 + kx] = salida[br(kx)·N2 + br(ky)]. Es transponer y dar vuelta los bits de los dos índices.

---

## 5. Modelo C++ y verificación

### 5.1 Modelo (`cpp/fft1d_r22sdf.hpp`)

- Escrito sobre **rtlsim**: `Reg<T>` con `.i`/`.o`, `hold()`/`commit()`; `Module` con *hooks*; ciclo `hold → comb → seq → commit → comb`. Usa `ap_fixed` de Vitis HLS.
- Tipos `in_t`, `data_t`, `out_t` y `tw_t`, con los mismos `static_assert` que el RTL.
- El producto del `cmul` se evalúa en `double`: es exacto solo si `NB + NB_TW ≤ 52` (`static_assert`).
- **Modo `fixed`**: bit-exacto con el RTL; lee los mismos `.mem` que el RTL.
- **Modo `DOUBLE`**: todo en `double` y **twiddles exactos** (calculados, sin `.mem`) desde 2026-10-12. No tiene ninguna cuantización.
- **Parámetros por instancia** (desde 2026-10-20, entrega `fft_params`). Los **formatos** (`NB`, `NBF`, `NB_TW`, `NB_IN`, `NB_OUT` y sus fraccionales) siguen siendo de compilación, porque definen los tipos `ap_fixed`. **N, `SCALE_MASK`, `INVERSE`, `CMUL_PIPE` y `TW_DIR`** pasan a ser de elaboración, en un `Params` que recibe el constructor, como los `parameter` de un módulo Verilog. Así un mismo binario tiene la FFT de 1024 y la de 512 (la FFT2D).
  - `Params{}` toma los macros, así que `fft1d_r22sdf_model dut;` es el modelo de siempre, y las constantes del namespace (`N`, `LATENCY`, …) describen esa configuración por defecto. El tb y el gtest no cambiaron.
  - `Params::validate()` aplica la misma guarda que el RTL (N potencia de 2 y ≥ 4) con `std::invalid_argument`. `latency()` y `scale_shift()` (la S de 2^−S) por instancia; `latency_of()` es la fórmula común, con un `static_assert` contra `LATENCY`.

### 5.2 Las etapas de verificación

| Etapa | Compara | Detecta |
|---|---|---|
| `run_gtest` | modelo ↔ DFT de referencia (modos `fixed` y `double`) | errores de algoritmo |
| `run_regression_sim` | — (corre el modelo y genera los vectores `expected`) | — |
| `run_regression_vm` | RTL ↔ modelo, bit a bit | errores de implementación |
| `run_regression_synth` | — | recursos y camino crítico |
| `run_regression_impl` | — | timing después de *place & route* |

**Por qué hacen falta los dos primeros.** El vm compara el RTL contra el modelo: si el modelo está mal pero coincide con el RTL, el vm pasa. El gtest es la única etapa que valida el modelo en sí.

Lectura del gtest:
- Si falla `double`, el bug es **estructural**: índice, signo o control.
- Si pasa `double` y falla `fixed`, el bug es **numérico**: formato, redondeo o saturación.

### 5.3 gtest (`testbench/fft1d_r22sdf_gtest.cpp`)

Compara bin por bin, real e imaginario por separado, contra `X[k] = 2^−S Σ x[n] W^(nk)`, con estos detalles:
- **`INPUT_GAIN = 2^(S − log₂N)`.** Un tono crece ×2 por cada etapa sin escalar y el máximo está a la salida, así que esta ganancia evita la saturación para cualquier máscara. Con la máscara completa vale 1.
- **La referencia usa la entrada ya cuantizada** (`quantize_input`).
- **`(k·i) mod N` antes del coseno.** Sin esa reducción, la referencia perdía ~10⁻¹³ en N = 1024.

| Test | Estímulo | Salida esperada | Qué agarra |
|---|---|---|---|
| `ComplexSineLowFreq` | tono en k = 1 | 0,5 en k = 1 | signo del exponente, reordenamiento |
| `ComplexSineMidFreq` | tono en k = N/4 | 0,5 en N/4 | camino de rotaciones triviales |
| `ComplexSineHighFreq` | tono en k = N − 3 | 0,5 en N − 3 | frecuencias negativas, bits altos |
| `RealCosine` | coseno en k = 5 | 0,25 en 5 y en N − 5 | simetría hermítica |
| `DcInput` | constante 0,5 | 0,5 en k = 0 | crecimiento coherente, bypass de W⁰ |
| `ZeroInput` | todo cero | todo cero | offsets, basura inicial |
| `Multitone` | 0,3 / 0,2 / 0,15 en k = 3, 7, 12 | esos valores | linealidad (pico de 0,65) |
| `TwoConsecutiveFrames` | dos tramas pegadas | la 2.ª igual a una aislada | que el estado no contamine la trama siguiente |
| `LatencyMatchesFormula` (2026-10-17) | una trama | primer `o_valid` en `LATENCY`, N válidas seguidas, un solo `o_last` en la última | que la latencia sea N + (3 + CMUL_PIPE)(log₄N − 1). El vm compara ciclo a ciclo contra el RTL, así que entre los dos atan fórmula, modelo y RTL |

- **Tolerancia actual:** `1e-12` en `double` (error medido ≤ 3 · 10⁻¹⁶) y `5e-3` en `fixed`, o sea 164 LSB, contra errores reales de 1 a 5 LSB. ⏳ Reemplazarla por la cota B.
- **Lo que no cubre:** saturación, señales tipo k-space, huecos en `i_valid`, reset en medio de una trama, comparación contra el RTL.

### 5.4 Casos de regresión (un solo JSON para gtest, sim y vm)

`testbench/fft1d_r22sdf_tb_regression.json`. En el gtest el caso solo fija la configuración (el estímulo lo ignora); en sim y vm el estímulo es la trama que entra.

| Caso | Configuración | Estímulo / propósito |
|---|---|---|
| 8 originales | Q1.15, escalado completo, N = 64 / 256 / 1024 | tono, multitono, impulso, ruido, chirp: el diseño de siempre |
| `n1024_q1_15_m784_impulse` | 16 bits, 784 | impulso en n = 5 |
| `n1024_nb20_tw18_io16_noise` | 20/18/16, máscara completa | ruido 0,7 |
| `…_m784_multitone` | 20/18/16, 784 | tonos de 0,004 / 0,002 / 0,001: con 784 la suma tiene que ser < 2^(3−10) ≈ 0,0078 |
| `…_m784_impulse` | 20/18/16, 784 | impulso de 0,9: nunca crece, sirve para cualquier máscara |
| `…_m784_sat` | 20/18/16, 784 | ruido 0,7: **satura a propósito** (pico ideal de 6,1) |
| `…_m0_impulse` | 20/18/16, máscara 0 | el extremo |
| `*_inv_*` (5 casos, 2026-10-18) | `INVERSE = 1`: N = 64 multitono y ruido, N = 256 chirp (Q1.15); 20/18/16 con 784, multitono e impulso | IDFT con el mismo núcleo |
| N impar (11 casos, 2026-10-18) | N = 8 (tono, ruido), 32 (multitono), 128 (ruido), 512 Q1.15 (multitono, chirp); 512 en 20/18/16: ruido, `m448` (bit 8 = etapa final encendido), `m192` (bit 8 apagado), `inv` y `nopipe` | la etapa radix-2 final con las dos opciones de escalado, `INVERSE` y `CMUL_PIPE = 0` |
| `*_nopipe_*` (5 casos; eran `*_pipe_*` hasta el 2026-10-18) | `CMUL_PIPE = 0` explícito en N = 64, 256 y 1024 (20/18/16, 784) | referencia sin registro, con los mismos estímulos que sus pares: la salida válida es idéntica, adelantada log₄N − 1 ciclos. El resto de los casos usa el default (`CMUL_PIPE = 1`) |

### 5.5 Estado de verificación

| Fecha | Qué | Resultado |
|---|---|---|
| 2026-10-11 | `cmul` con anchos separados (iverilog) | 127 776 + 200 000 vectores, 0 errores ✅ |
| 2026-10-11 | vm (iverilog): NB_TW separado / puertos separados | 45/45 y 33/33 bit a bit ✅ |
| 2026-10-17 | gtest, JSON único | **28/28** (14 casos × 2 modos) ✅ |
| 2026-10-17 | sim | 14/14 ✅ |
| 2026-10-17 | **vm en XSim** | **14/14, 19 360 vectores, 0 errores** ✅ |
| 2026-10-17 | Wrappers de síntesis e implementación con los formatos nuevos (iverilog): salida igual a la del modelo con 2 ciclos de retardo, en 4 configuraciones × 2 wrappers. El wrapper viejo da lo mismo con los defaults | 8/8 ✅ |
| 2026-10-17 | vm (iverilog) de 27/18/16 con 784 | igual bit a bit ✅ |
| 2026-10-17 | **Síntesis** (Vivado 2024.2, xck26-sfvc784-2LV-c, 200 MHz): 4 de 4 cierran timing, 16 DSP en todos (§5.6) | 4/4 ✅ |
| 2026-10-17 | **Implementación** de los mismos 4 casos (§5.6): la elegida (20/18/16) cierra con +0,050 ns; la de 27 bits no cierra por 17 ps | 3/4 ✅ |
| 2026-10-17 | `cmul_pipe` en el sandbox (iverilog, runners reales de gtest y sim; **no XSim ni Vivado**) | ver abajo ✅ |
| | · defaults (`CMUL_PIPE = 0`): los 14 casos de antes, `expected` + `actual` + estímulos | 238 `.dat` idénticos byte a byte a la versión anterior ✅ |
| | · vm emulado (modelo vs RTL en iverilog), 19 casos | 19/19 bit a bit ✅ |
| | · `CMUL_PIPE = 1` contra 0, mismo estímulo (5 pares, modelo y RTL) | muestras válidas idénticas; primer valid +2 / +3 / +4 en N = 64 / 256 / 1024 ✅ |
| | · 27/18/16 con 784, `CMUL_PIPE` 0 y 1 (ruido y multitono) | 4/4 bit a bit; mismo corrimiento de +4 ✅ |
| | · 3 tramas seguidas de ruido por el RTL con `CMUL_PIPE` 0 y 1 | 3072 muestras idénticas, `o_last` en las mismas posiciones ✅ |
| | · `cmul` solo, `PIPE = 1` contra 0 (18×20, 18×27, 16×16; ~12 % saturan) | 3 × 200 000 vectores, 0 distintos ✅ |
| | · gtest real, 19 casos × 2 modos (9 tests con `LatencyMatchesFormula`) | 38/38, 342 tests ✅ |
| | · sim real, 19 casos | 19/19 ✅ |
| | · wrappers de síntesis e implementación, `CMUL_PIPE` 0 y 1 | igual a `expected` corrido 2 ciclos ✅ |
| | · guarda de `PIPE` inválido | `$fatal` en tiempo 0 ✅ |
| 2026-10-17 | `cmul_pipe` en el repo: gtest y vm en XSim (19 casos) | ✅ (Julián) |
| 2026-10-17 | **Síntesis** de los casos `_pipe` (§5.6): WNS 1,936 ns (20/18) y 1,919 ns (27/18) a 200 MHz | 2/2 ✅ |
| 2026-10-17 | **Implementación** de los casos `_pipe` (§5.6): WNS +0,903 ns (20/18) y +0,785 ns (27/18) a 200 MHz | 2/2 ✅ |
| 2026-10-18 | `fft_inverse` en el sandbox (iverilog y runners reales de gtest, sim y plot; **no XSim ni Vivado**) | ver abajo ✅ |
| | · los 19 casos con `INVERSE = 0`: `expected`, `actual` y estímulos contra `cmul_pipe_default` | 323 `.dat` idénticos byte a byte ✅ |
| | · vm emulado, 24 casos | 24/24 bit a bit ✅ |
| | · salida con `INVERSE = 1` contra la IDFT de NumPy (N·ifft·2^−S) | 0,8 a 2,6 LSB de error máximo, dentro de la cota B (§4.7) ✅ |
| | · RTL: `INVERSE = 1` con x contra swap(`INVERSE = 0` con swap(x)), 4 tramas seguidas con extremos ±2^15, N = 1024 (784) y N = 64 | idéntico ciclo a ciclo ✅ |
| | · gtest real, 24 casos × 2 modos | 48/48, 432 tests ✅ |
| | · mutación: referencia con e^{−j} en un caso `INVERSE = 1` | 5 de 9 tests fallan (los que no son simétricos) ✅ |
| | · sim real 24/24; wrappers con `INVERSE = 1`; `run_plot_fft1d` con referencia IDFT | ✅ (SQNR 63,0 dB igual que la directa; picos en N − k) |
| 2026-10-18 | `fft_n512` en el sandbox (iverilog y runners reales de gtest, sim y plot; **no XSim ni Vivado**) | ver abajo ✅ |
| | · los 24 casos de N par contra `fft_inverse` | 408 `.dat` idénticos byte a byte ✅ |
| | · vm emulado, 35 casos (11 de N impar) | 35/35 bit a bit ✅ |
| | · N impar contra la DFT/IDFT de NumPy | 0,3 a 3,7 LSB en Q1.15 y < 0,8 LSB en 20/18/16 ✅ |
| | · latencia del RTL (primer valid) | 11 / 39 / 139 / 527 para N = 8 / 32 / 128 / 512; 523 sin registro: coincide con la fórmula ✅ |
| | · RTL: 3 tramas seguidas contra las mismas tramas sueltas (N = 512 directa; 512 inversa con 448; N = 8) | idénticas, `o_last` en la última de cada trama ✅ |
| | · gtest real 70/70 (630 tests), sim real 35/35, wrappers con N = 512, plot | ✅ |
| | · guarda: N = 2 y N = 12 cortan con `$fatal`; N = 4 elabora | ✅ |
| 2026-10-20 | `fft_n512` en el repo (PC nueva, Ubuntu nativo): gtest 66/70. Fallan los 4 casos de N = 8, 32 y 128 en modo `fixed` (los 9 tests, en 0,00 s). En `double` pasan. Causa: faltaban los twiddles `n8`, `n32` y `n128` en el repo (el modo `double` no los lee). Reproducido en el sandbox escondiendo `n8`. Copiados del tar: gtest 70/70 y `--validate` 35/35 | ✅ (Julián) |
| 2026-10-20 | **Implementación de N = 512** (20/18/16, máscara completa): WNS **+0,192 ns**, WHS +0,007, Fmax 208 MHz, 2408 LUT, 1053 FF, 266 CARRY8, 16 DSP, 0 BRAM | ✅ (Julián) |
| 2026-10-20 | **Refactor `Params`** (entrega `fft_params`, solo `cpp/fft1d_r22sdf.hpp`), en el sandbox: (1) sim 35/35 con los **vectores idénticos byte a byte** a los del modelo anterior (los reportes difieren solo en `elapsed`); (2) gtest 70/70 (630 tests); (3) **un binario por formato configurado en ejecución** con los `Params` de cada caso, contra los vectores del modelo viejo compilado con macros: 35/35 bit a bit, incluidos los contadores trazados; (4) dos instancias distintas en el mismo `Simulator` (1024 inversa con pipe y 512 sin pipe): 0 diferencias; (5) N = 2, 3, 12, 0 y −8 tiran `invalid_argument` | ✅ (sandbox) ⏳ repo |
| 2026-10-20 | **Paso 5b, `fft2d`** (entrega `fft2d_paso5b`, §5.7), en el sandbox: gtest 10/10 (60 tests) con los runners reales y el `fft2d_vars.sh` que genera `create_module`; mutaciones 8/8; tamaño completo 8 bobinas 801 / 384 bit a bit contra Python (0 de 4 194 304), latencia 525 857 = fórmula, 23,60 ms, SNR de imagen 51,03 dB | ✅ (sandbox) ⏳ repo |
| 2026-10-21 | **`fft2d_tb_plot`** (sandbox): tb real de sim, estímulo común, JSON único de 8 casos, generador del fantoma y `run_plot_fft2d.py`. gtest 16/16 (96 tests), sim 8/8 con autochequeo de tiempos, plot 8/8 y `--full-check`. `fft2d_check_full.py` en el repo de Julián: 0 diferencias en 4 194 304 muestras, 51,03 dB. **En el repo, 2026-10-21: gtest 16/16 (96 tests), sim 8/8, plot 8/8 con los mismos números que el sandbox (fantoma 71,33 dB)** | ✅ (repo) |
| 2026-10-20 | **`run_gtest.py` daba PASS con tests skipped**: si algo falla fuera de un test (una excepción en `SetUpTestSuite`), gtest sale con 1 pero el XML no tiene fallas. Encontrado con las mutaciones. Corregido: código de salida ≠ 0 sin fallas en el XML = FAIL. gtest del 1D sigue 70/70 | ✅ (sandbox) |
| 2026-10-20 | **`create_module` generaba mal `<PREFIX>_GTEST_CPP`** (`$<PREFIX>/…` en vez de `$<PREFIX>_TB_DIR/…`, en el `module_utils.sh` del 17). Corregido | ✅ (sandbox) |
| 2026-10-20 | Camino crítico de N = 512: BF2I escalada de la unidad 2 → dos carry chains (suma y redondeo) → saturación → A del DSP. Las máscaras 801 y 384 también escalan una BF2I con DSP, así que se espera un WNS del orden de +0,2 ns | 🟡 (falta implementar con 801 y 384) |

### 5.6 Resultados de síntesis e implementación (2026-10-17)

Vivado 2024.2, `xck26-sfvc784-2LV-c`, reloj de 5 ns (200 MHz) con 0,05 ns de incertidumbre, wrapper con registros de entrada y salida. Porcentajes sobre el XCK26 (DS987, tabla 5: 117 120 LUT, 234 240 FF, 1248 DSP).

| Caso | WNS (ns) | Fmax est. (MHz) | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|---|
| 16/16, escalado completo | 0,241 | 210,1 | 2615 (2,23 %) | 786 | 16 | 0 |
| 16/16, 784 | 0,241 | 210,1 | 2610 | 786 | 16 | 0 |
| **20/18/16, 784 (elegida)** | **0,313** | **213,4** | **2978 (2,54 %)** | **902** | **16** | 0 |
| 27/18/16, 784 (techo) | 0,289 | 212,3 | 3655 (3,12 %) | 1101 | 16 | 0 |

- **El ensanche cuesta solo lógica, no DSP ni timing.** 16 → 20 bits: +363 LUT (+13,9 %) y +116 FF (+14,8 %). 16 → 27: +1040 LUT (+39,8 %) y +315 FF (+40,1 %). Son unos 92 LUT y 29 FF por bit de datapath.
- **La máscara no cambia el área** (−5 LUT) ni el timing.
- **El WNS no empeora con el ancho**: el multiplicador del DSP es de 27 × 18 sin importar cuántos bits se usen. Las diferencias de ±0,07 ns están dentro de la variación de la herramienta.
- **BRAM = 0.** Las ROM de twiddles (8 de 1024 × 16) quedaron en LUTs porque Vivado absorbió su registro de salida dentro del DSP, y sin registro la ROM no puede ser BRAM. Las líneas de retardo quedaron en LUTRAM porque `delay_line` lee asíncrono. *Esto corrige la predicción anterior de "un BRAM18 por componente".*
- **Cuatro núcleos 20/18** (P = 4) serían ~11 900 LUT, el 10 % del dispositivo.
- **El margen es chico:** 0,24–0,31 ns es 5–6 % del período, y es una estimación sin ruteo.

**Implementación** (mismo flujo, `place_design` + `route_design` + `phys_opt` en el router). Estos son los números a citar en la tesis:

| Caso | WNS (ns) | WHS (ns) | Fmax (MHz) | LUT | FF | CARRY8 | DSP | BRAM | |
|---|---|---|---|---|---|---|---|---|---|
| 16/16, escalado completo | 0,066 | 0,019 | 202,7 | 2577 | 786 | 264 | 16 | 0 | ✅ |
| 16/16, 784 | 0,107 | 0,014 | 204,4 | 2576 | 786 | 180 | 16 | 0 | ✅ |
| **20/18/16, 784 (elegida)** | **0,050** | 0,016 | **202,0** | **2955** | **902** | 209 | **16** | 0 | ✅ |
| 27/18/16, 784 (techo) | −0,017 | 0,001 | 199,3 | 3637 | 1101 | 264 | 16 | 0 | ✗ |

- **El ruteo se comió 0,17–0,31 ns** del margen de síntesis. La configuración elegida cierra a 200 MHz, pero con **50 ps (1 % del período)**. La de 27 bits no cierra por 17 ps (después de la colocación estaba en −0,249 ns y `phys_opt` la acercó).
- **El hold cierra en todos** (WHS ≥ 0,001 ns).
- **Las LUT bajan un poco respecto de síntesis** (2978 → 2955): `opt_design` elimina lógica.
- **La máscara reduce los CARRY8** (264 → 180 en 16 bits): cada mariposa escalada tiene un sumador de redondeo, y con 784 escalan 3 etapas en vez de 10.
- **Camino crítico** (caso de 27 bits; los nets que `phys_opt` intentó mejorar): `r_bf2_do`/`r_tw` (AREG/BREG) → multiplicador → suma en cascada (PCIN) → salida P **sin registro** → `cast` (`u_round_im`: redondeo y saturación en fábrica) → mux del *bypass* W⁰ → `r_mu_do`. Es el candidato que esperábamos. ⏳ Confirmarlo en el 20/18 con el primer camino de `timing_summary_impl.rpt`.
- **Por qué es lento** (DS925, tabla 83, −2 a 0,72 V): el DSP48E2 llega a 644 MHz con todos los registros, 440 MHz sin MREG y 323 MHz sin MREG ni ADREG. Nosotros además sacamos P sin PREG hacia la fábrica, y ahí todavía falta el `cast`. Es la configuración más lenta posible del DSP.
- **Propuesta** (implementada como `CMUL_PIPE`, entrega `cmul_pipe`; ⏳ falta síntesis e implementación): registrar la suma completa antes del `cast`. Vivado lo absorbe como PREG del segundo DSP de la cascada, así que no gasta DSP ni LUT.
  - Parte el camino en dos: AREG/BREG → mult → cascada → PREG, todo adentro de los DSP; y PREG → `cast` → mux → `r_mu_do`.
  - Cuesta un ciclo por multiplicador: **latencia 1036 → 1040**. Es irrelevante contra los 50 ms por trama.
  - El *bypass* de W⁰ y los `valid` se atrasan un ciclo para seguir alineados: ~2·NB + 2 FF por unidad, ~170 FF en total con NB = 20.
  - El control de cada unidad arranca con el `valid` que le llega y `o_last` se cuenta sobre el `valid` de salida, así que el resto del RTL no cambia. El modelo C++ copia el registro. Numéricamente es bit-exacto igual; solo cambia la latencia.
  - Alternativa sin latencia: sumar la constante de redondeo adentro del ALU del DSP, para que en la fábrica quede solo la saturación. Gana menos y depende de que Vivado lo infiera bien.

**Síntesis con `CMUL_PIPE = 1`** (2026-10-17, mismo flujo):

| Caso | WNS (ns) | Fmax est. (MHz) | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|---|
| 20/18/16, 784, sin registro | 0,313 | 213,4 | 2978 | 902 | 16 | 0 |
| **20/18/16, 784, `CMUL_PIPE = 1`** | **1,936** | **326,4** | 2981 | 1070 | 16 | 0 |
| 27/18/16, 784, sin registro | 0,289 | 212,3 | 3655 | 1101 | 16 | 0 |
| 27/18/16, 784, `CMUL_PIPE = 1` | 1,919 | 324,6 | 3655 | 1325 | 16 | 0 |

- **El WNS sube 1,62 ns** (de 6 % a 39 % del período). Fmax estimada sin ruteo: ~325 MHz.
- **El registro quedó adentro del DSP**, deducido de la cuenta de FF: el aumento es exactamente 4·(2·NB + 2), es decir +168 con NB = 20 y +224 con NB = 27. Son solo el *bypass*, su selección y el `valid` atrasados. Si la suma registrada hubiera quedado en fábrica serían +312 y +368 más. ⏳ Confirmarlo directamente en el "DSP Final Report" (PREG = 1 en 8 DSP).
- **LUT y DSP no cambian.**
- **Con registro, 20 y 27 bits dan casi el mismo WNS** (17 ps de diferencia). Eso sugiere que el camino crítico nuevo no depende del ancho del dato; un candidato es la ROM de twiddles, que tiene 18 bits en los dos casos. Es una hipótesis: hay que verla en el reporte de timing de la implementación.

**Implementación con `CMUL_PIPE = 1`** (2026-10-17). **Estos son los números finales de la FFT 1D a citar en la tesis:**

| Caso | WNS (ns) | WHS (ns) | Fmax (MHz) | LUT | FF | CARRY8 | DSP | BRAM | BUFG |
|---|---|---|---|---|---|---|---|---|---|
| 20/18/16, 784, sin registro | 0,050 | 0,016 | 202,0 | 2955 | 902 | 209 | 16 | 0 | 1 |
| **20/18/16, 784, `CMUL_PIPE = 1`** | **0,903** | 0,009 | **244,1** | **2956** | **1070** | 209 | **16** | 0 | 1 |
| 27/18/16, 784, sin registro | −0,017 ✗ | 0,001 | 199,3 | 3637 | 1101 | 264 | 16 | 0 | 1 |
| 27/18/16, 784, `CMUL_PIPE = 1` | 0,785 | 0,009 | 237,2 | 3637 | 1326 | 264 | 16 | 0 | 2 |

- **La configuración elegida cierra a 200 MHz con 0,903 ns de margen (18 % del período)**, contra 0,050 ns sin el registro. Con registro, también cierra la de 27 bits.
- **El costo es solo de FF**: +168, y prácticamente 0 LUT. Ni DSP ni BRAM.
- **El ruteo se comió mucho más que antes**: 1,03 ns (20/18) y 1,13 ns (27/18), contra 0,26 ns sin registro. Indica que el camino crítico nuevo es de fábrica, con mucho ruteo. El de antes estaba dominado por el DSP y la cascada, que tienen ruteo dedicado y la síntesis los estima bien. ⏳ Identificarlo con el primer camino de `timing_summary_impl.rpt`.
- **250 MHz no está asegurado**: la Fmax después del ruteo es 244 MHz, medida con restricción de 5 ns. Con 4 ns la herramienta se esfuerza más, así que podría cerrar, pero hay que correrlo y probablemente atacar el camino nuevo.
- La de 27 bits con registro usó 2 BUFG (Vivado insertó uno, probablemente para una red de mucho *fan-out*) y 1 FF más que en síntesis (alguna replicación). No afecta.
- **El caso 27/18 sin registro falla siempre** y deja la regresión en rojo. Es solo una referencia: desde `cmul_pipe_default` está con `"enabled": false` en el JSON de implementación (en síntesis pasa y sigue activo).

**Camino crítico con `CMUL_PIPE = 1`** (2026-10-18, `timing_summary_impl.rpt` del 20/18/16):

```
u_bf2i/r_addr_reg[0]  (dirección de la línea de retardo de BF2I, unidad 0, 256 × 40 bits)
  → ruteo 1,043 ns, fan-out 167            ← lo más caro del camino
  → LUTRAM RAMS64E + MUXF7 + MUXF8 (lectura asíncrona)
  → LUT2 + 3 CARRY8 (suma de 21 bits de la mariposa)
  → ruteo 0,414 ns, fan-out 20 (el bit de signo va a la saturación de todos los bits)
  → LUT3 (saturación / mux de la mariposa)
  → ruteo 0,438 ns, fan-out 23 (el bit de signo se extiende a A[19..29] de 2 DSP + copia en fábrica)
  → entrada A del DSP (r_bf2_do absorbido como AREG; setup 0,522 ns)
```

- Datos: 3,407 ns, 32 % lógica y 68 % ruteo; 8 niveles. Margen 0,903 ns.
- **El cuello pasó del multiplicador a la mariposa BF2I de la unidad 0**, que tiene la línea de retardo más larga de las BF2I (256). Coincide con lo que se veía en síntesis: 20 y 27 bits daban casi el mismo WNS, porque lo que domina es el ruteo, no el ancho de la suma.
- **Confirmación de PREG:** la celda DSP destino se llama `g_pipe.r_ad_plus_bc_reg/DSP_A_B_DATA_INST`. Vivado le pone a la DSP48E2 el nombre del registro que absorbió.
- **Hold**: el peor (0,009 ns) es `r_bf2_do` → `r_bf2_do_d`, un FF a FF sin lógica que agregó `CMUL_PIPE`. Cumple; el router se encarga del hold.
- **Camino crítico de N = 512** (20/18/16, máscara completa, implementación, 2026-10-20). Margen +0,192 ns.

```
gen_unit[2].u_bf2i/r_addr_reg[1]     (fan-out 42, 0,501 ns)
  → LUTRAM RAMS32 de la línea (8 de profundidad), 0,199 ns
  → LUT2 + 2×CARRY8: la suma de la mariposa         (unos 0,49 ns con ruteo)
  → ruteo 0,401 ns + 2×CARRY8: el +BIAS del redondeo (u_rnd_sum_im, unos 0,82 ns)
  → LUT5 (saturación / mux), ruteo 0,866 ns
  → entrada A del DSP (AREG; setup 0,495 ns)
```

  - Datos: 3,974 ns, 31 % lógica y 69 % ruteo; 7 niveles (4 CARRY8).
  - La diferencia con 1024/784 está en la segunda cadena: en 1024 con 784 la BF2I que alimenta el DSP no escala, así que no hay `+BIAS`.
  - Hold: el peor (+0,007 ns) es otra vez `r_bf2_do` → `r_bf2_do_d`.
- **Opciones si hace falta 250 MHz** (a 200 MHz no hace falta nada):
  1. Barata: limitar el *fan-out* de la dirección de la línea (`max_fanout` o replicar el contador). Ataca el tramo de 1,043 ns. No cambia la función ni la latencia.
  2. Estructural: línea de retardo con **lectura sincrónica** (dirección calculada un ciclo antes y salida registrada), con el mismo retardo total D. Saca la LUTRAM y sus muxes del camino, permite mapear las líneas de 256 y 512 a BRAM y libera LUT. Si el retardo visible no cambia, el modelo no cambia.
  3. Registro en fábrica entre la mariposa y el DSP: evita el setup de 0,522 ns de la entrada A, pero suma un ciclo por unidad.
  4. (Idea, sin verificar en síntesis) Fusionar el `+BIAS` del redondeo en el sumador de la mariposa cuando escala (drop = 1): ⌊(a+b+1)/2⌋ usa el *carry-in* de la misma cadena y es bit-exacto. Para la resta no sale gratis: a − b + 1 = a + ~b + 2 necesita dos acarreos, así que haría falta un sumador ternario o reordenar. Si solo se arregla la suma, la resta sigue siendo crítica.

### 5.7 Modelo de la FFT 2D (`modules/fft2d/`, paso 5b, 2026-10-20) ✅ sandbox

- **`cpp/fft2d.hpp`**: `fft2d_model`, un `rtl::Module` con dos `fft1d_r22sdf_model` adentro (`Params` por instancia, §5.1), dos buffers alternados de N2 × N1 complejos (memoria ideal: un puerto de escritura, uno de lectura con salida registrada, *read-first*) y `o_last` por bobina. `fft2d::Params` = `pass1` + `pass2`; los defaults son los de producción (1024 × 512, 801 / 384, `INVERSE = 1`, `CMUL_PIPE = 1`, 20/18/16) y se cambian con `FFT2D_*`.
- **Orden:** la pasada 1 escribe en `fila·N1 + slot` tal como sale (bit-reversed). La pasada 2 lee por columnas (`slot t1`, `k'` natural). La salida `o = t1·N2 + t2` es A[p = br(t2), x = br(t1)], el orden en que se guarda U.
- **Protocolo:** el escritor alterna buffers por bobina y marca `full` al terminar. El lector toma un buffer lleno, baja `full` y lee T = N1·N2 posiciones seguidas, sin burbuja entre bobinas. El escritor puede empezar en un buffer vacío o en uno que se está leyendo, nunca en uno lleno sin leer. El modelo tira `std::logic_error` si pasa eso o si una escritura pisa una posición todavía no leída. Margen mínimo con entrada continua: N1 + N2 − 1 − d ciclos (1534 en 1024 × 512).
- **Latencia: L2D = L1 + T + 2 + L2** con la convención del 1D. El 2 es la escritura (sale registrada del núcleo 1 y se escribe en el flanco siguiente, junto con `full`) y la lectura registrada. 1024 × 512: **525 857 ciclos** (2,63 ms). 8 bobinas: última muestra en el ciclo 4 720 160, o sea **23,60 ms** a 200 MHz.
- **Un solo JSON** (decisión de Julián, 2026-10-21): `testbench/fft2d_tb_regression.json`, compartido por gtest y sim como en el 1D (`run_gtest` lo usa si no hay un `_gtest_regression.json`). 8 casos: los 5 de antes (ruido), `n64x32_impulse_m0`, `n64x32_tone` y `n64x32_phantom` (8 bobinas, máscaras 25 / 4 elegidas por barrido exhaustivo con la cadena de Python: 71,33 dB, sin saturación). `N_CYCLES` = 0 es automático.
- **Estímulo común** (`testbench/fft2d_stimulus.hpp`, entrega `fft2d_tb_plot`): ruido, impulso, tono o archivo (`STIM_TYPE` 0–3). Lo usan el tb y el gtest, así que con el mismo caso ven la misma entrada. El ruido sigue la misma secuencia que antes (mismas SQNR en el gtest).
- **`fft2d_tb.cpp`** (etapa sim, reemplaza la plantilla): vectores como el 1D (`i_*`, `o_valid`, `o_last`, `o_re`, `o_im` y los contadores `wr_count`, `rd_count`, `out_count`). Con `N_CYCLES` automático se chequea solo: todas las muestras, un `o_last` por bobina y primer `o_valid` igual a la fórmula; si no, falla.
- **`model/gen_fft2d_stim.py`**: escribe el k-space del fantoma (int32 crudo Q1.15) y un `.json` con cómo se generó, para que el plot rehaga fantoma y mapas.
- **`scripts/python/run_plot_fft2d.py`**: por caso y por bobina, |K| en dB, |A| en orden natural junto a la IDFT 2D ideal, error en LSB; con el fantoma, |m|, |m̂| desplegada con la salida del modelo y el error de imagen. `--full-check` plotea la salida de `fft2d_check_full.py` (que ahora deja `case.json`). Verificado en el sandbox: 8/8 casos; el fantoma 64 da 71,33 dB y el completo 51,03 dB, iguales a la cadena de Python; el estímulo coincide bit a bit con el fantoma regenerado. El tono cae donde tiene que caer (p0 = 7 + l, x0 = 20).
- **gtest** (`testbench/fft2d_gtest.cpp`, 5 casos × 2 modos, 6 tests): composición bit a bit con dos modelos 1D sueltos; IDFT 2D en double (306 dB en `double`; 62–70 dB en `fixed`, mínimo exigido 40); primer `o_valid` = fórmula, salida continua y un `o_last` por bobina; huecos de 0 a 3 ciclos entre filas sin cambiar ningún valor; guardas de N. Casos 64 × 32 (máscara completa, parcial 41/24, sin pipe), 32 × 8 (las dos pasadas con N impar) y 16 × 64 (más filas que columnas).
- **Mutaciones**: 8 de 8 detectadas (sin transponer, k' invertido, fórmula de latencia, `o_last` corrido, escritor o lector que no alternan, lector que arranca antes de `full`, burbuja de 1 ciclo entre bobinas).
- **Tamaño completo** (`model/fft2d_check_full.py` + `testbench/fft2d_dump.cpp`, compilado con `-O2`): fantoma, 8 bobinas, 801 / 384, normalización por bobina. **0 diferencias en 4 194 304 muestras** contra la cadena de Python; tiempos iguales a la fórmula; SNR de la imagen **51,03 dB**, la misma del estudio de §4.8. También 1023 / 511 con 2 bobinas: 0 diferencias. El C++ tarda 73 s con `-O2`.
- **Velocidad:** ~8 µs por ciclo y por núcleo de 1024 con `-O2`, unas 17 veces más con `-O0` (el Makefile). Con `-O2` los headers `ap_private` de Vitis HLS tiran falsos positivos de `-Wuninitialized`, así que el script los apaga.
- **Imágenes cuadradas con la FFT 2D directa** (2026-10-24, entrega `fft2d_imagenes`): `gen_fft2d_stim.py --from-json` genera todos los estímulos que pide el JSON (clave `stim_gen` de cada caso; Shepp-Logan y anillos de `gen/gen_phantom.py`, senos y *zone plate*). Casos `img*` con `INVERSE` = 0, máscara completa y 1 canal; gtest solo en 64 × 64 (en los grandes `"gtest": {"modes": []}`). `run_plot_fft2d` tiene un modo imagen: entrada en grises y espectros centrados (fftshift) en dB. Sim con `-O0`: ~35–40 s en 256², ~2,3 min en 512².

  | Caso | SQNR | Qué se ve |
  |---|---|---|
  | Shepp-Logan 64 / 256 / 512 | 47,6 / 36,5 / 30,7 dB | el espectro clásico de elipses |
  | seno (16, 0) y (12, 5) | 88,2 / 90,3 dB | dos deltas exactos (frecuencias enteras) |
  | seno rotado 25°, f = 20 | 49,0 dB | dos cruces de derrame: la frecuencia no cae en un bin |
  | anillos de período 16 | 53,5 dB | un anillo de radio 16 más una cruz: el patrón no es periódico en los bordes |
  | *zone plate* 256 / 512 | 48,7 / 42,6 dB | otra *zone plate*, con réplicas: pasa Nyquist en las esquinas |

  **Con escalado completo la SQNR cae ~6 dB cada vez que se duplica el lado** (Shepp-Logan: 47,6 → 36,5 → 30,7). Por Parseval, la energía de la salida dividida por N² no depende de N, pero el ruido de redondeo crece con la cantidad de muestras (N²). Es la misma razón por la que en la IFFT de SENSE se eligieron máscaras parciales (§4.8).
- ⏳ Próxima pasada: FFT 2D y después IFFT 2D (ida y vuelta). La salida de la FFT 2D sale bit-reversed en los dos ejes y por columnas, así que para la IFFT hay que reordenar (en la memoria ideal es una cuestión de direcciones).
- ⏳ Los huecos **entre frames** del núcleo 1D están probados solo en el modelo (el test de huecos de la 2D se apoya en eso): falta un caso de vm del 1D con huecos para confirmar que el RTL hace lo mismo.
- ⏳ Los dos buffers suman 4 MiB y no entran en las 64 URAM (2,25 MiB). Es del paso 6: el modelo deja la memoria aislada para reemplazarla.

---

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
- **Claude Code en VS Code (2026-10-25, migración desde claude.ai):** la bitácora pasa a vivir en el repo, en `docs/bitacora_tesis.md`, y esa copia es la fuente de verdad. `CLAUDE.md` en la raíz tiene las reglas de trabajo y el índice de la bitácora. `.claude/settings.json` bloquea `make qspi` y pide confirmación para `xsdb`, `xsct`, `git commit` y `git push`; es un resguardo, no una barrera: la protección real sigue siendo `CONFIRM_QSPI=NO`. Los manuales van en `docs/manuales/`. Cada llamada de Claude a la terminal es una shell nueva: el entorno se carga en el mismo comando.

### 6.2 Árbol del repositorio (`mri-fpga/`, 2026-10-17)

```
books/           libros en PDF (+ papers/fft, papers/mri, amd-docs/pg151-div-gen.pdf)
bringup/
  scripts/       Makefile, config.mk, boot/, jtag/, tools/, vitis/, vivado/   infraestructura de placa
  v0/            loopback DMA (Tcl de proyecto y block design, firmware)
  v1/, v1.1/     hardware/, firmware/, results/, Makefile                    (v1.1: 250 MHz; estado a confirmar)
doc/
  amd/           PDFs de AMD (+ txt/ y pdf/; pdf2text.sh para convertir)
  theory_r22sdf.md
model/           cfxp.py, fxp.py   (modelos de punto fijo en Python)
modules/         btfly_sdf, cast, cmul, cmul_karatsuba, delay_line,
                 fft1d_r2, fft1d_r2sdf, fft1d_r22sdf, tile_transpose
rtlsim/          rtlsim.hpp, rtlsim_core.hpp, rtlsim_trace.hpp, include/ (ap_fixed de Vitis HLS)
scripts/         Makefile, set_env.sh, lint_waivers.txt,
                 python/ (runners + venv), shell/ (utilidades), tcl/ (xsim, synth, impl)
equations_prompt.txt, prompt.txt
```

**Documentación disponible para consultar.** Claude solo puede leer lo que está subido al **proyecto de claude.ai** (o lo que se adjunta en el chat). Los PDFs del repo no los ve hasta que se suben, idealmente en `.txt` con `doc/amd/pdf2text.sh`.

| Dónde | Qué hay |
|---|---|
| Proyecto (claude.ai) | En txt: UG1085 (TRM), UG579 (DSP), UG573 (memorias), DS987 (K26), PG201, PG164, PG125, PG021, PG037, PG085. En PDF: DS925, UG1400, UG1137, UG1089, UG1209. Además: headers de rtlsim, scripts de bring-up, el RTL de la FFT y esta bitácora. **El RTL del proyecto es del 2026-10-04, anterior a los cambios de octubre**: conviene reemplazarlo por las versiones vigentes (tabla de arriba) |
| Repo `doc/amd/` | además: PG022 (DataMover), PG034 (CDMA), PG035 (AXIS interconnect), PG059 (AXI interconnect), PG065 (clk_wiz), PG078 (AXI BRAM ctrl), PG144 (GPIO), PG247 (SmartConnect), UG1037 (AXI), UG1118 (IP propio), UG572 (clocking), UG896 (IP), UG949 (metodología), UG984, UG994/995 (IP integrator), WP417 (interconnects) |
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

- `-std=c++17 -O0 -g -Wall -Wextra`. `HLS_AP_DIR = $RTLSIM_ROOT/include` (los `ap_fixed` están en el repo).
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

---

## 7. Bring-up en placa

| Etapa | Estado |
|---|---|
| 1 · DMA vivo | ✅ (v0) |
| 2 · Transferencia chica (4 KB) | ✅ |
| 3 · Transferencia grande | ✅ (barrido hasta 8 MiB) |
| 4 · Curva de ancho de banda | ✅ |
| 4b · Ancho de banda **agregado** con varios HP en paralelo | ⏳ planeado |
| 5 · FFT en el datapath (v1) | ⏳ |
| v1_1 a 250 MHz | ⏳ planeado, no corrido |

- **v0:** DDR → MM2S → AXIS FIFO → S2MM → DDR a 100 MHz; éxito = `memcmp(A, B) == 0` ✅.
- **Barrido de ancho de banda** (100 MHz, puerto de 128 bits, teórico 1600 MB/s) ✅:
  - pico de **1586 MB/s (99,1 %)**;
  - **N½ ≈ 3 KiB** (tamaño al que se alcanza la mitad del pico);
  - **T_fijo ≈ 4,5 µs** por transferencia;
  - **el flush de caché es el 19 % del tiempo de DMA** (989 µs para 8 MiB), así que hay que usar buffers no cacheables.
- **Consecuencia para el *corner-turn*:** con N½ ≈ 3 KiB, un tile de lecturas de 256 B rinde muy poco. El tile tiene que ser bastante más grande que 3 KiB por acceso.
- **Notas prácticas:**
  - En WSL la UART es `/dev/ttyUSB0` (`config.mk` decía `ttyUSB1`), con `usbipd attach` en cada reconexión.
  - En el Tcl del `clk_wiz`: sacar `MMCM_*` y `CLKOUT*_JITTER`/`PHASE_ERROR` (Vivado los ignora) y agregar `PRIM_SOURCE {No_buffer}`.

---

## 8. Trampas conocidas

| Trampa | Síntoma | Qué hacer |
|---|---|---|
| Twiddles de un ancho distinto a `NB_TW` | gtest con SQNR ~2 dB; **el vm pasa igual** | una carpeta por formato; ⏳ chequear `tw_re[0]` al cargar |
| `make` no seguía `DEFINES` | se probaba la configuración anterior sin aviso | corregido el 2026-10-17 (`makefile_fix`); con el Makefile viejo, `rm -rf build` |
| Binario `.exe` viejo en `build/` | gtest sin salida ni XML, 0,06 s | `rm -rf build`; ⏳ sacar el `.exe` de `run_gtest.resolve_executable` |
| Proceso que muere sin salida | "no se pudo parsear el XML" | ⏳ mostrar el código de salida o la señal |
| Excepción en `SetUpTestSuite` | gtest sale con 1, los tests quedan *skipped* y `run_gtest` decía PASS | corregido el 2026-10-20 (código ≠ 0 = FAIL). En los tests, simular adentro del test y no en `SetUpTestSuite` |
| `create_module` con `<PREFIX>_GTEST_CPP` mal armado | `run_gtest` busca `/<module>_gtest.cpp` | corregido el 2026-10-20 en `module_utils.sh`; `refresh_module_vars <module>` regenera el `_vars.sh` |
| Plantilla de tb de `create_module` | sim da PASS con 16 ciclos que no prueban nada (pasó con la `fft2d`, 2026-10-21) | reemplazar `<module>_tb.cpp` antes de confiar en sim (en la `fft2d`, resuelto con `fft2d_tb_plot`) |
| `-O2` con `ap_fixed` | decenas de `-Wuninitialized` dentro de `ap_private` (falsos positivos) | `-Wno-uninitialized -Wno-maybe-uninitialized` solo en los builds `-O2`; con `-O0` no aparecen |
| `cast` con NBF_OUT > NBF_IN | X | ⏳ `$fatal` |
| iverilog ignora `$fatal` en elaboración | el diseño inválido corre | doble `$fatal` (`generate` + `initial`) |
| N no potencia de 2 (o N < 4) | basura silenciosa | corta con `$fatal` / `static_assert` (desde `fft_n512`; antes exigía potencia de 4) |
| Máscara en hex en el JSON | Verilog no la entiende | decimal (784) |
| Texto después de `` `endif `` (p. ej. `` `endif __SYNTHESIS__ ``) | queda como código: Vivado da *syntax error near 'localparam'* en la línea siguiente | comentarlo: `` `endif // __SYNTHESIS__ `` (2026-10-17) |
| Default de `TW_DIR` **con comillas** en los wrappers | `STRINGIFY` produce `""twiddles/n1024""` y no elabora si el JSON no pasa `TW_DIR` | corregido el 2026-10-17 (default sin comillas) |
| Twiddles que no se cargan en síntesis | ROM en cero, multiplicadores optimizados y **menos de 16 DSP**: `MAX_DSP` pasa igual | revisar en el log de Vivado los avisos de `$readmem` y que el conteo sea **exactamente** 16 |
| Nombres dentro de la unidad | `BF2II` es la **primera** mariposa | bit 2u = primera, 2u + 1 = segunda |
| CRLF / LF mezclados | `patch` rechaza bloques | ⏳ `.gitattributes` con `eol=lf` |
| `run_plot_fft1d` con NBF del datapath | figuras 16 veces más chicas | corregido el 2026-10-17 (usa NBF_IN / NBF_OUT) |
| Modo `DOUBLE` con twiddles cuantizados | `double` fallaba 5/8 | corregido el 2026-10-12 (twiddles exactos) |
| Chequear el fin de línea con `file` | `file` no dice "CRLF" en los JSON; Python en modo texto convierte `\r\n` a `\n` sin avisar | contar con `grep -c $'\r$'` y abrir en binario (`'rb'`). Pasó con el JSON de regresión el 2026-10-17; se corrigió antes de entregar |

---

## 9. Pendientes y próximos pasos

**Plan acordado el 2026-10-25, antes del MAC y el CORDIC en C++:**
- **A. Ida y vuelta FFT 2D → IFFT 2D** (§4.12). Estudio en `model/fft2d_roundtrip_study.py`. ⏳ Julián tiene que decidir:
  - si se usa la regla "dividir lo justo y al principio de cada pasada";
  - cómo se arman los casos: el generador de Python calcula el espectro bit a bit y lo reordena como estímulo de un caso `INVERSE = 1` (recomendado), o el tb de C++ encadena dos `fft2d`.
  - Después: casos en el JSON y ploteo con original, espectro, reconstruida y error.
- **B. Redondeo convergente** (*round-half-even*, §4.6) en `cast.v` y en los modelos. La medida es el pico del píxel (0, 0) de la ida y vuelta y la fila p = 0 del fantoma.
- **C. Cerrar el núcleo 1D en hardware con la configuración final:**
  - implementación con 801 (N = 1024) y 384 (N = 512) **después de B**, porque B toca el camino crítico (BF2I escalada → suma → redondeo → saturación → DSP);
  - caso de vm con huecos entre tramas.
- **D. Infraestructura del código del PS:** dónde vive, gtest en la PC y compilación para el A53. Después, `unfold_ref.py` a C++, y modelo C++ del MAC y el CORDIC.

**Corto plazo (FFT 1D):**
1. ✅ Síntesis e implementación de los 4 casos (§5.6). La elegida cierra a 200 MHz con 50 ps de margen. ✅ Registro antes del `cast` del `cmul` (`CMUL_PIPE`) implementado y verificado en el sandbox. ✅ Aplicado; gtest y vm pasan en el repo. ✅ Síntesis e implementación: la elegida cierra con 0,903 ns (§5.6). ✅ Camino crítico nuevo identificado (§5.6): línea de retardo en LUTRAM → suma → saturación → entrada A del DSP. ✅ PREG absorbido (el DSP lleva el nombre de `r_ad_plus_bc_reg`). ✅ Default `CMUL_PIPE = 1` y caso 27/18 sin registro desactivado en implementación (entrega `cmul_pipe_default`). ⏳ Julián: aplicar `cmul_pipe_default` y correr gtest, sim y vm. Síntesis e implementación no hace falta repetirlas: los casos por defecto ahora son los que ya se midieron como `_pipe`.
   - Para la v1_1 a 250 MHz (no hace falta a 200): ver las opciones al final de §5.6.
   - Los otros usos de `cmul.v` (regresión propia de `modules/cmul`) siguen con `PIPE = 0`; no los pude correr porque no tengo esos archivos.
   - Próximos caminos críticos candidatos: la ROM de twiddles en LUT (contador → `tw_addr` → ROM de 1024 → BREG) y las mariposas escaladas (LUTRAM → suma → redondeo → mux → registro).
2. ⏳ Tolerancia del gtest = cota B (§4.7).
3. ⏳ Infraestructura:
   - mostrar el código de salida cuando un binario falla;
   - sacar el `.exe` de `run_gtest`;
   - twiddles generados por caso desde N/NB_TW;
   - `SCALE_STAGES` legible en el JSON;
   - repo en el filesystem de Linux con LF.
4. ⏳ Test con espectro aleatorio (tipo k-space) en el gtest.
5. ⏳ Experimento de redondeo convergente (§4.6).

**FFT 2D: plan propuesto (2026-10-18, a acordar con Julián).** Identidades verificadas en NumPy (`fft2d_theory.py`, entrega `fft2d_teoria`):
- **IFFT con el mismo núcleo:** swap(FFT(swap(X))) = N·IFFT(X), con swap(a + jb) = b + ja. Error 0 en N = 512 y 1024. El swap es cableado y no niega nada, así que no hay desborde y el error en punto fijo es el mismo que el de la FFT directa.
- **N = 2^impar con la misma estructura:** las unidades R2² de siempre (M = N, N/4, …, 8), todas con multiplicador por la regla `LOG_M != 2`, más **una etapa radix-2 final de span 1 sin twiddle**. Probado con el direccionamiento de twiddles del RTL (`tw_sel`, `tw_num`) para N = 8 … 2048, contra `np.fft.fft`, con error de 10⁻¹⁴. N = 512 usa 4 multiplicadores, igual que 1024.
- **fftshift / ifftshift:** son permutaciones (se hacen en el direccionamiento) o, con N múltiplo de 4, modulaciones (−1)^k a la entrada y (−1)^n a la salida. La modulación niega, y negar −2^15 desborda, así que se prefiere la permutación.
- **2D separable**, primero filas de 1024 y después columnas de 512. El bit-reversed de cada pasada es solo una permutación de índices: la pasada 2 trabaja columna por columna, así que el orden de las columnas no importa, y todo se puede reordenar al final. Como el unfolding es punto a punto por posición plegada, alcanza con guardar U en el mismo orden.
- Números: 524 288 muestras por pasada y por bobina (2,62 ms a 200 MHz). Con un núcleo, 8 bobinas × 2 pasadas = 41,9 ms. Con un carril (núcleo de 1024 + núcleo de 512, encadenados por bobina) son 9 turnos = 23,6 ms; con 2 carriles, 13,1 ms. El intermedio de una bobina en 16 bits ocupa 2 MiB, y las 64 URAM suman 2,25 MiB.

Pasos propuestos:
1. ✅ **Referencia 2D en NumPy** con las convenciones explícitas (§3.8, `model/fft2d_ref.py`). Plan acordado con Julián el 2026-10-18.
2. ✅ **`INVERSE` en `fft1d_r22sdf`** (swap a la entrada y a la salida), en el modelo y en el RTL (entrega `fft_inverse`, verificado en el sandbox; ⏳ correrlo en el repo).
3. ✅ **N = 2^impar** (512): etapa radix-2 final (`fft1d_r22sdf_tail`), bit 8 de `SCALE_MASK`, twiddles de 512 (entrega `fft_n512`, verificado en el sandbox; ⏳ correrlo en el repo, más síntesis e implementación del caso `n512_nb20_tw18_io16`).
4. 🟡 **Máscaras de las dos pasadas** (§4.8): con el fantoma, propuesta provisoria 801 / 384 con normalización por bobina. ⏳ Validar con datos reales (fastMRI; a conversar cómo se obtienen y preprocesan) y decidir margen y si la máscara pasa a ser configurable en tiempo de ejecución.
5. 🟡 **Modelo C++ de la `fft2d`** (dos núcleos y memoria ideal) contra NumPy, con su regresión.
   - 5a ✅ (sandbox, 2026-10-20): `Params` por instancia en el modelo 1D (§5.1), entrega `fft_params`. ⏳ Correr gtest y sim en el repo.
   - 5b ✅ (repo, 2026-10-21): modelo ciclo a ciclo `modules/fft2d/` (§5.7), entregas `fft2d_paso5b` y `fft2d_tb_plot`. gtest 16/16, sim 8/8, plot 8/8, `fft2d_check_full.py` OK. ⏳ Caso de vm del 1D con huecos entre frames.
6. **Memoria y *corner-turn*** (DDR por tiles o URAM; cantidad de carriles). Depende del ancho de banda real, que queda para cuando se retome el bring-up.

**MAC de desplegado** (empezado el 2026-10-21): puntos 1 a 5 ✅ con el fantoma (§4.9 a §4.11). ⏳ Siguen: confirmar la infraestructura del código del PS y pasar `unfold_ref.py` a C++ (bit a bit contra la referencia), modelo C++ del MAC + CORDIC contra `mac_cordic_fxp.py`, reordenamiento de la imagen de salida, y repetir λ / E / NB_U con mapas reales (fastMRI).

**Mediano plazo:**
- Curvas de SNR contra bits por eje (datos, twiddle, entrada) con varios estímulos y filas de fastMRI.
- Medir el SNR de fastMRI → ancho de entrada 16 o 18 bits.
- **IFFT**: signo, 1/N, `fftshift`.
- Etapa radix-2 extra para N = 512 (pasada por columnas).
- Máscara de la segunda pasada.
- *Corner-turn* (URAM o DDR con tiles grandes).
- Wrapper AXI-Stream de la FFT (v1 en placa) y v1_1 a 250 MHz.
- **RTL del unfolding** (todavía no empezó; cuesta lo mismo que la FFT y necesita M = 4).
- CORDIC de magnitud.

---

## 10. Riesgos y preguntas abiertas

- **El unfolding sin RTL** es la mitad del presupuesto de tiempo.
- **Los coeficientes de unfolding (32 MiB por trama)** pueden ser el mayor consumidor de ancho de banda.
- **El ancho de entrada** depende de un dato que todavía no tenemos (el SNR de fastMRI).
- **El sesgo del redondeo** en 16 bits: ¿cuánto pesa en la imagen final?
- **El ancho de banda real de DDR4 para la PL**: el pico teórico según DS987 es 19,2 GB/s (el 17,07 de antes parece ser DDR4-2133). Falta medir el agregado con varios puertos HP (§7); la fase de FFT con P = 4 necesita ~6,4 GB/s.

---

## 11. Bibliografía

| Referencia | Para qué |
|---|---|
| Pruessmann et al. 1999 | SENSE, matriz de unfolding |
| Roemer et al. 1990 | *phased array*, RSS |
| Uecker et al. 2014 | ESPIRiT (mapas de sensibilidad) |
| He & Torkelson 1996 (*A new approach to pipeline FFT processor*) | R2²SDF original. Ojo: los nombres BF2I/BF2II del RTL están invertidos respecto del paper |
| Garrido 2022 | survey de arquitecturas FFT *pipelined* |
| Oppenheim & Weinstein 1972 / Welch 1969 | análisis de ruido de cuantización en FFT (base para §4.7) |
| Brown, Cheng & Haacke, *MRI: Physical Principles*, cap. 28 | *parallel imaging* |
| Akçakaya et al., *MR Image Reconstruction*, caps. 2–3 y 6 | reconstrucción |
| Paul & Mathew, cap. 2 | regularización de Tikhonov |
| Khan, *Digital Design of Signal Processing Systems* | aritmética de punto fijo, arquitecturas |
| Smith & Smith 1995, *Handbook of Real-Time FFTs* | FFT en tiempo real |
| UG579 | DSP48E2: 27 × 18, *pattern detector*, redondeo convergente |
| UG573 | BRAM / URAM (URAM288 = 4K × 72, dos puertos) |
| UG1085 (TRM) | puertos HP y DDRC |
| DS987, DS925 | K26 SoM, Zynq UltraScale+ |
| PG021, PG085 | AXI DMA, AXIS FIFO |
| UG1089, UG1400, UG1137 | KV260, Vitis embebido, software MPSoC |

*Las de Oppenheim/Welch y He & Torkelson las propuso Claude; conviene conseguirlas y verificar la cita exacta.*

---

## 12. Registro de cambios

| Fecha | Cambio |
|---|---|
| antes del 2026-10-10 | Bring-up v0 y barrido de ancho de banda en placa. Diagnóstico del escalado (pico de 0,9 LSB). |
| 2026-10-10 | Decisión 1024 × 1024. `SCALE` por mariposa y `SCALE_MASK`; 784 por búsqueda exhaustiva. Estudio de precisión: mapa datos × twiddle, análisis de *full precision*, 2D y ruido térmico. |
| 2026-10-11 | Formatos independientes: `NB_TW` (`cmul` con dos anchos, ROM) y `NB_IN`/`NB_OUT` (ensanche por cableado, `cast` de salida). Guardas con doble `$fatal`. `gen_twiddles.py`. |
| 2026-10-12 | Modo `DOUBLE` con twiddles exactos; referencia con `(k·i) mod N`. Derivación de la cota B y Monte Carlo. |
| 2026-10-14 | gtest con máscara: referencia 2^−S, `INPUT_GAIN`, entrada cuantizada. |
| 2026-10-17 | `tb.sv` con formatos y máscara; JSON único con 14 casos. gtest 28/28, sim 14/14, **vm 14/14 en XSim**. Hallazgo del sesgo *half-up*. `run_plot_fft1d` corregido, con panel de error. Corrección del orden de etapas en la cota. Revisión de la infraestructura: bug de `make` con `DEFINES`, corregido y verificado con los runners reales. Árbol del repo y documentación disponible en §6.2. Definición de SQNR en §4.1. Wrappers de síntesis e implementación con los formatos nuevos y JSON de 4 casos. SNR contra SQNR y modelo de suma de ruidos (§4.1). **Síntesis: 4/4 a 200 MHz, 16 DSP en todos; 20/18 cuesta +14 % de LUT/FF** (§5.6). **Implementación: la elegida cierra con +0,050 ns; camino crítico DSP → `cast` en el `cmul`.** Cifra de DDR4 corregida a 19,2 GB/s (DS987). **`CMUL_PIPE`**: registro antes del `cast` (PREG), latencia 1040, verificado en el sandbox (defaults idénticos byte a byte; 19/19 vm; gtest 38/38 con `LatencyMatchesFormula`) y en el repo (gtest, vm en XSim). **Con `CMUL_PIPE = 1`, la elegida cierra después del ruteo con 0,903 ns (Fmax 244 MHz), +168 FF.** |
| 2026-10-24 | **Imágenes para la FFT 2D** (entrega `fft2d_imagenes`): Shepp-Logan 64 / 256 / 512, senos (exactos y rotados), anillos y *zone plate*, con la FFT directa; modo imagen en `run_plot_fft2d`; generador por JSON. Glosario de símbolos en §1. Hallazgo: con escalado completo, −6 dB de SQNR por cada duplicación del lado. Doc "SENSE en el FPGA: ecuaciones en limpio" (pasos 1–5 con fórmulas renderizadas, diagrama de la trama y tabla de valores); queda abierta la elección de λ. |
| 2026-10-25 | Prueba rápida de ida y vuelta FFT 2D → IFFT 2D (N = 256, modelo bit a bit): con escalado completo en las dos la imagen desaparece; con la inversa sin escalar vuelve, pero el sesgo *half-up* del espectro aparece como un pico de ≈ 4 000 LSB en el píxel (0, 0) (§4.6). Reparto de la escala (§4.12): lo que hay entre pasadas se conoce exacto; dividir lo justo y al principio de cada pasada da +18 dB en Shepp (54,3 contra 36,5); dividir al final satura. **Migración a Claude Code en VS Code** (entrega `migracion_vscode`): bitácora en `docs/`, `CLAUDE.md`, `.claude/settings.json`, `model/fft2d_roundtrip_study.py` (§4.12, §6.1). Plan A–D en §9. |
| 2026-10-21 | **MAC de desplegado, puntos 4 y 5** (§4.10, §4.11): `unfold_ref.py` (U por L D Lᴴ, re/im en el orden del C, Tikhonov, escala con 1/g_l y 1/K_n, E por `frexp`, buffer en orden hardware), `mac_cordic_fxp.py` (MAC exacto + salida + CORDIC bit-exacto), `mac_output_study.py`. U'' idéntico al de `solve`; interfaz en orden hardware idéntica al orden natural; con mapas enmascarados sin λ falla el 83 %; propuesta λ = 10⁻⁴, n = 12, G = 5, u16 UQ1.15 con D = 49 − S − E − F_OUT: 51,96 dB de punta a punta. |
| 2026-10-21 | **MAC de desplegado, puntos 1–3** (§4.9, `model/mac_unfold_study.py`): m̂ = 2^(S−19+E)·Σ U''·A verificado (5,5·10⁻¹⁶); E común por sesión alcanza; NB_U = 16 pierde 0,001 dB; acumulador de NB_U + 20 bits, suma independiente del orden; propuesta de infraestructura para el código del PS. Paso 5 de la FFT 2D cerrado en el repo (gtest 16/16, sim 8/8, plot 8/8). |
| 2026-10-21 | **FFT 2D: tb, plot y JSON único** (entrega `fft2d_tb_plot`): `fft2d_tb.cpp` de verdad, `fft2d_stimulus.hpp` común a tb y gtest, `fft2d_tb_regression.json` (8 casos, incluido el fantoma 64 × 64), `model/gen_fft2d_stim.py`, `run_plot_fft2d.py` con `--full-check`. `fft2d_check_full.py` corrido en el repo: OK. Hallazgo: el sesgo *half-up* se concentra en la fila p = 0 (+7,5 LSB) y se ve como una línea en la imagen desplegada (§4.6). |
| 2026-10-20 | **FFT 2D, paso 5b:** modelo ciclo a ciclo de la `fft2d` con memoria ideal (§5.7), entrega `fft2d_paso5b`: `fft2d.hpp`, gtest de 5 casos (10/10, mutaciones 8/8), `fft2d_dump.cpp` y `model/fft2d_check_full.py` (tamaño completo bit a bit contra Python, 51,03 dB), twiddles de 18 bits para N = 8 a 64 con `gen_twiddles_r22sdf.py`. Corregidos `run_gtest.py` (PASS con tests skipped) y `create_module` (`GTEST_CPP`). |
| 2026-10-20 | **FFT 2D, paso 5a:** `Params` por instancia en el modelo 1D (entrega `fft_params`): vectores idénticos byte a byte, 35/35 configurado en ejecución, dos instancias en un `Simulator`. Twiddles de N = 8, 32 y 128 que faltaban en el repo (gtest 70/70, `--validate` 35/35). Implementación de N = 512: +0,192 ns; camino crítico con el redondeo de una BF2I escalada, que también aparece con las máscaras 801 y 384 (§4.8, §5.6). |
| 2026-10-18 | **FFT 2D, paso 4 (fantoma):** modelo bit-exacto vectorizado en Python (35/35 + 16 de estrés, igual al RTL y al C++) y estudio de máscaras (§4.8). Lo que decide es S1, S2 y el margen; la ubicación da igual. Normalizar por bobina da +1,9 dB gratis. Propuesta provisoria 801 / 384: 51,0 dB, margen ≥ 6,5 dB. Entrega `fft2d_mascaras`. |
| 2026-10-18 | **FFT 2D, paso 3:** N = 2^impar con `fft1d_r22sdf_tail` (etapa radix-2 final), entrega `fft_n512`: 11 casos de N impar, twiddles nuevos, caso de síntesis/implementación de N = 512, `model/fft2d_theory.py`. Latencia 527 para N = 512. Verificado en el sandbox. |
| 2026-10-18 | **FFT 2D, paso 2:** parámetro `INVERSE` (swap real/imag; 0 LUT, 0 latencia), entrega `fft_inverse` con 5 casos `_inv_`. gtest referencia IDFT, plot con IFFT. Verificado en el sandbox; la relación swap es bit a bit también en el RTL. `model/fft2d_ref.py` con `--gen-dir` por defecto en `model/gen/`. |
| 2026-10-18 | **FFT 2D, paso 1:** convenciones y referencia (`model/fft2d_ref.py`, §3.8). Dimensiones confirmadas (512 × 1024 por bobina). Plegado, flujo del hardware, unfolding en orden hardware y k-space centrado verificados. Conclusión: el datapath no necesita hardware de reordenamiento ni shifts. |
| 2026-10-18 | **Default `CMUL_PIPE = 1`** (entrega `cmul_pipe_default`); los casos `_pipe_` pasan a ser `_nopipe_` con `CMUL_PIPE = 0`; caso 27/18 sin registro desactivado en implementación. Verificado en el sandbox: 19/19 vm, gtest 38/38, sim 19/19, wrappers. **Camino crítico nuevo**: línea de retardo LUTRAM de BF2I (unidad 0) → suma → saturación → entrada A del DSP; 68 % ruteo, *fan-out* 167 en la dirección. PREG confirmado por el nombre de la celda DSP. |

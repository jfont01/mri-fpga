# Modelo C++, verificación, síntesis e implementación

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

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
| `*_halfup_*` (7 casos; eran `*_conv_*` hasta el 2026-11-15, con los papeles al revés) | Desde la entrega 4 el default es convergente: los casos comunes son los convergentes y estos llevan `ROUND_CONV: 0` como referencia *half-up*. Antes, `ROUND_CONV = 1`: N = 64 y 1024 Q1.15 ruido; 20/18/16: 1024 ruido, 784 multitono y saturación, 512 ruido (tail) y 512 inversa | pares de casos existentes con el mismo estímulo: el sesgo baja a ~0 (§4.6) |
| `m801` y `m384`, convergente y `_halfup_` (4 casos, 2026-10-26; nombres dados vuelta el 2026-11-15) | 20/18/16, las máscaras de SENSE: 801 en N = 1024 (tonos 0,008 / 0,004 / 0,002) y 384 en N = 512 (0,004 / 0,002 / 0,001) | hasta ahora no estaban en la regresión 1D; sirven también para el paso C |
| `*_frames3_gap*` (5 casos, 2026-10-26) | 3 tramas con huecos de 0, 3 y 100 ciclos (N = 64, Q1.15, ruido), 1 ciclo (N = 512, 384, convergente) y 5 ciclos (N = 1024, 801, convergente). `TB_FRAMES` y `TB_GAP` son del tb de C++; la trama k es el estímulo rotado 7k muestras | huecos entre tramas en el RTL (el vm compara la salida y, ciclo a ciclo, `r_out_count` y los contadores de la primera unidad) |
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
| 2026-10-26 | `cast` con `ROUND_CONV` (copia del repo, XSim) | defaults 6/6 idénticos byte a byte; sim y vm 14/14; barrido exhaustivo de 11 formatos, 0 diferencias; guarda corta en C++ y XSim (iverilog no instalado: el `$fatal` del `initial` no se probó) ✅ |
| 2026-10-26 | `cast` en el repo (Julián): sim y vm 14/14; `fft1d_r22sdf` sin cambios: gtest, sim y vm 35/35 | ✅ |
| 2026-10-26 | `fft1d_round_conv` (entrega 1 del paso B) en la copia del repo (XSim) | ver abajo ✅ |
| | · los 35 casos de antes: estímulos y `expected` contra el baseline | idénticos byte a byte ✅ |
| | · gtest 46 casos × 2 modos | 92/92 ✅ |
| | · sim 46/46, **vm en XSim 46/46** | ✅ |
| | · modelo de Python (`--validate`) | 46/46 bit a bit ✅ |
| | · conversión del modelo C++ en las mariposas (`double` → `AP_RND_CONV`), exhaustiva en 2²¹ sumas | 0 diferencias ✅ |
| 2026-10-26 | `fft1d_round_conv` en el repo (Julián): gtest 92/92, sim 46/46, **vm en XSim 46/46** (los 11 casos nuevos con 0 errores), `--validate` 46/46; `fft2d` gtest y sim sin cambios | ✅ |
| 2026-10-26 | `fft1d_round_conv`, `fft2d` en la copia: los 23 casos idénticos byte a byte al baseline; gtest 28/28 | ✅ |
| 2026-10-26 | `fft2d_round_conv` (entrega 2) en la copia: gtest 34/34 (204 tests); sim 28/28, los 23 de antes idénticos byte a byte; los 32 estímulos de antes idénticos; plot 28/28 (salida del C++ = cadena de Python en todos los `rt*`); estudio de ida y vuelta idéntico | ✅ |
| 2026-10-26 | `fft2d_round_conv` en el repo (Julián): gtest 34/34, sim 28/28; vectores idénticos a los de la copia en los 28 casos | ✅ |
| 2026-10-26 | `synth_impl_huecos` (entrega 3) en la copia (XSim): sim 51/51, los 46 de antes idénticos byte a byte; gtest y **vm de los 5 casos con huecos 5/5** (3 × N muestras válidas en el modelo y en el RTL) | ✅ |
| 2026-10-26 | Wrappers de síntesis e implementación con `ROUND_CONV` (XSim, copia): 1024 / 801 y 512 / 384, *half-up* y convergente, salida = modelo; control negativo (caso convergente con `ROUND_CONV = 0`) distinto | ✅ |
| 2026-10-26 | `synth_impl_huecos` en el repo (Julián): gtest 102/102, sim 51/51, **vm de los 5 casos con huecos 5/5** (0 errores); síntesis e implementación de los 4 casos de SENSE (§5.6) | ✅ |
| 2026-11-15 | **Entrega 4: `ROUND_CONV = 1` por default** (copia del repo, XSim): estímulos de la ida y vuelta cruzados contra sus gemelos viejos 34/34 idénticos; vectores cruzados idénticos byte a byte en 20 casos de `fft1d_r22sdf` y 9 de `fft2d` (cada común nuevo = su `_conv` viejo; cada `_halfup` = su común viejo); los 50 casos sin par pasan a convergente. `fft1d_r22sdf`: gtest 102/102 (918 tests), sim 51/51, vm 51/51, `--validate` 51/51. `fft2d`: gtest 34/34 (204 tests), sim 28/28, plot 28/28. Los dos estudios de redondeo dan idéntico. En el repo (Julián): aplicada; tus estímulos iguales a los de la copia | ✅ |
| 2026-10-20 | `fft_n512` en el repo (PC nueva, Ubuntu nativo): gtest 66/70. Fallan los 4 casos de N = 8, 32 y 128 en modo `fixed` (los 9 tests, en 0,00 s). En `double` pasan. Causa: faltaban los twiddles `n8`, `n32` y `n128` en el repo (el modo `double` no los lee). Reproducido en el sandbox escondiendo `n8`. Copiados del tar: gtest 70/70 y `--validate` 35/35 | ✅ (Julián) |
| 2026-10-20 | **Implementación de N = 512** (20/18/16, máscara completa): WNS **+0,192 ns**, WHS +0,007, Fmax 208 MHz, 2408 LUT, 1053 FF, 266 CARRY8, 16 DSP, 0 BRAM | ✅ (Julián) |
| 2026-10-20 | **Refactor `Params`** (entrega `fft_params`, solo `cpp/fft1d_r22sdf.hpp`), en el sandbox: (1) sim 35/35 con los **vectores idénticos byte a byte** a los del modelo anterior (los reportes difieren solo en `elapsed`); (2) gtest 70/70 (630 tests); (3) **un binario por formato configurado en ejecución** con los `Params` de cada caso, contra los vectores del modelo viejo compilado con macros: 35/35 bit a bit, incluidos los contadores trazados; (4) dos instancias distintas en el mismo `Simulator` (1024 inversa con pipe y 512 sin pipe): 0 diferencias; (5) N = 2, 3, 12, 0 y −8 tiran `invalid_argument` | ✅ (sandbox) ⏳ repo |
| 2026-10-20 | **Paso 5b, `fft2d`** (entrega `fft2d_paso5b`, §5.7), en el sandbox: gtest 10/10 (60 tests) con los runners reales y el `fft2d_vars.sh` que genera `create_module`; mutaciones 8/8; tamaño completo 8 bobinas 801 / 384 bit a bit contra Python (0 de 4 194 304), latencia 525 857 = fórmula, 23,60 ms, SNR de imagen 51,03 dB | ✅ (sandbox) ⏳ repo |
| 2026-10-21 | **`fft2d_tb_plot`** (sandbox): tb real de sim, estímulo común, JSON único de 8 casos, generador del fantoma y `run_plot_fft2d.py`. gtest 16/16 (96 tests), sim 8/8 con autochequeo de tiempos, plot 8/8 y `--full-check`. `fft2d_check_full.py` en el repo de Julián: 0 diferencias en 4 194 304 muestras, 51,03 dB. **En el repo, 2026-10-21: gtest 16/16 (96 tests), sim 8/8, plot 8/8 con los mismos números que el sandbox (fantoma 71,33 dB)** | ✅ (repo) |
| 2026-10-20 | **`run_gtest.py` daba PASS con tests skipped**: si algo falla fuera de un test (una excepción en `SetUpTestSuite`), gtest sale con 1 pero el XML no tiene fallas. Encontrado con las mutaciones. Corregido: código de salida ≠ 0 sin fallas en el XML = FAIL. gtest del 1D sigue 70/70 | ✅ (sandbox) |
| 2026-10-20 | **`create_module` generaba mal `<PREFIX>_GTEST_CPP`** (`$<PREFIX>/…` en vez de `$<PREFIX>_TB_DIR/…`, en el `module_utils.sh` del 17). Corregido | ✅ (sandbox) |
| 2026-10-20 | Camino crítico de N = 512: BF2I escalada de la unidad 2 → dos carry chains (suma y redondeo) → saturación → A del DSP. Las máscaras 801 y 384 también escalan una BF2I con DSP, así que se espera un WNS del orden de +0,2 ns | 🟡 (falta implementar con 801 y 384) |

### 5.6 Resultados de síntesis e implementación (2026-10-17)

**2026-10-26 (paso C): las máscaras de SENSE y el redondeo convergente** ✅ (Vivado 2024.2, repo de Julián). 20/18/16, `CMUL_PIPE = 1`, 200 MHz, post-route:

| Caso | WNS (ns) | Fmax est. (MHz) | LUT | FF | CARRY8 | DSP | BRAM |
|---|---|---|---|---|---|---|---|
| N = 1024, 801, *half-up* | +0,565 | 225,5 | 2965 | 1070 | 221 | 16 | 0 |
| N = 1024, 801, convergente | **+0,468** | 220,7 | 2985 | 1074 | 221 | 16 | 0 |
| N = 512, 384, *half-up* | +0,687 | 231,9 | 2405 | 993 | 182 | 16 | 0 |
| N = 512, 384, convergente | **+0,540** | 224,2 | 2415 | 997 | 182 | 16 | 0 |
| (N = 512, máscara completa, *half-up*, 2026-10-20) | +0,192 | 208,0 | 2408 | 1053 | 266 | 16 | 0 |

- **Los 4 cierran a 200 MHz** con ≥ 0,47 ns de margen (WHS positivo en todos, 0,004 a 0,011 ns).
- **Costo del convergente:** −0,10 ns (801) y −0,15 ns (384) de margen, +10 a +20 LUT y +4 FF; mismos DSP y CARRY8. La síntesis no lo distingue (WNS 1,715 / 1,717 en los dos modos): aparece recién con el ruteo.
- **Camino crítico, el mismo en los 4:** dirección de la línea de retardo (LUTRAM, `RAMS32`) de la BF2I escalada → suma y redondeo (4 `CARRY8`) → saturación (LUT) → entrada A del DSP del `cmul` de la misma unidad (unidad 2 con 801, M = 64; unidad 3 con 384, M = 8). 6 a 7 niveles; 62 a 68 % ruteo. Con el convergente crece sobre todo el ruteo (2,35 → 2,48 / 2,54 ns).
- Las máscaras de SENSE tienen **más** margen que N = 512 con máscara completa (+0,192): escalan menos etapas, y menos de ellas alimentan un DSP.
- ✅ **Decisión de Julián (2026-11-15):** `ROUND_CONV = 1` pasa a default, como `CMUL_PIPE` el 2026-10-18 (entrega 4). En los JSON de síntesis e implementación, `…_m801` y `…_m384` ahora son los convergentes y los *half-up* se llaman `…_halfup`; los números de esta tabla no cambian (son las mismas configuraciones).

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
- **Imágenes cuadradas con la FFT 2D directa** (2026-10-24, entrega `fft2d_imagenes`): `gen_fft2d_stim.py --from-json` genera todos los estímulos que pide el JSON (clave `stim_gen` de cada caso; Shepp-Logan y anillos de `gen/gen_phantom.py`, senos y *zone plate*). Casos `img*` con `INVERSE` = 0, máscara completa y 1 canal; gtest solo en 64 × 64 (en los grandes `"gtest": {"modes": []}`). `run_plot_fft2d` tiene un modo imagen: entrada en grises y espectros centrados (fftshift) en dB. Sim con `-O0`: ~35–40 s en 256², ~2,3 min en 512². En el repo (2026-10-25): gtest 18/18, sim 17/17, plot 17/17.

  | Caso | SQNR | Qué se ve |
  |---|---|---|
  | Shepp-Logan 64 / 256 / 512 | 47,6 / 36,5 / 30,7 dB | el espectro clásico de elipses |
  | seno (16, 0) y (12, 5) | 88,2 / 90,3 dB | dos deltas exactos (frecuencias enteras) |
  | seno rotado 25°, f = 20 | 49,0 dB | dos cruces de derrame: la frecuencia no cae en un bin |
  | anillos de período 16 | 53,5 dB | un anillo de radio 16 más una cruz: el patrón no es periódico en los bordes |
  | *zone plate* 256 / 512 | 48,7 / 42,6 dB | otra *zone plate*, con réplicas: pasa Nyquist en las esquinas |

  **Con escalado completo la SQNR cae ~6 dB cada vez que se duplica el lado** (Shepp-Logan: 47,6 → 36,5 → 30,7). Por Parseval, la energía de la salida dividida por N² no depende de N, pero el ruido de redondeo crece con la cantidad de muestras (N²). Es la misma razón por la que en la IFFT de SENSE se eligieron máscaras parciales (§4.8).
- ✅ Ida y vuelta FFT 2D → IFFT 2D (2026-10-25): casos `rt*`, con la ida calculada en Python y la vuelta en la `fft2d` (§4.12). Encadenar dos `fft2d` queda para cuando esté la memoria (paso 6).
- ✅ Huecos **entre frames** del núcleo 1D en el RTL (2026-10-26, entrega 3): 5 casos de vm con 3 tramas y huecos de 0, 1, 3, 5 y 100 ciclos, incluidos N = 512 (etapa final) y la configuración de producción; 5/5 en XSim, en la copia y en el repo. El test de huecos de la 2D ya no se apoya solo en el modelo.
- **`FFT2D_ROUND_CONV`** (2026-10-26, entrega 2 del paso B): prende el redondeo convergente en los dos núcleos (default 1 desde el 2026-11-15; los casos `n64x32_phantom_halfup` y `rt*_halfup` son la referencia *half-up*; `rt256_sine_rot25` solo existe en convergente porque en *half-up* satura). El gtest de composición lo hereda por `Params`. El plot muestra, en los casos inversos, el sesgo de la fila p = 0 contra el resto.
- ⏳ Los dos buffers suman 4 MiB y no entran en las 64 URAM (2,25 MiB). Es del paso 6: el modelo deja la memoria aislada para reemplazarla.

### 5.8 Código del PS: cálculo de U'' (`modules/ps_sense/`, paso D1, 2026-11-15) ✅ PC y A53 (qemu)

**Qué es.** `cpp/ps_sense.hpp` / `ps_sense.cpp`: la versión en C++ de `model/unfold_ref.py`, operación por operación, para correr en el A53 una vez por sesión. C++17 portable: sin excepciones, sin memoria dinámica, sin `iostream`, sin `ap_fixed` ni `rtlsim`. Compila igual con el `g++` de la PC, con `aarch64-none-elf-g++` (bare-metal, Vitis 2024.2, GCC 13.3) y con `aarch64-linux-gnu-g++`.

**Partes.**
- Núcleo por posición (x, p), con los mismos nombres que la referencia: `gram` (A = SᴴS + λI, suma secuencial en l), `ldlh` (por columnas, 1/d una vez), `ldlh_ok` (pivotes finitos y > 0), `solve_u` (por bobina: adelante, D⁻¹, atrás) y `unfold_position` (las cuatro juntas). Arreglos de trabajo en la pila, con límites `MAX_L = 16` y `MAX_AF = 4` (unos 3 KiB; se cambian con `-DPS_SENSE_MAX_L` / `-DPS_SENSE_MAX_AF`).
- Entrada: `gather_s` arma S[l][a] = s_l(x, p + a·n2) de mapas complejos `double` intercalados con índice [l][x][y] (el layout en memoria de un `complex128` de NumPy `smaps[l, x, y]`).
- Escala y cuantización: `cordic_gain`, `coil_scales` (c_l = (1/g_l)·(1/K)), `quantize` (`rint(ldexp(v, shift))` con saturación a nb bits), `pack_word` ({im, re}).
- Índices: `bitrev`, `log2_exact`, `word_index` (`pos_coil_a` y `coil_pos_a`).
- Sesión: `build_buffer(dims, smaps, g, config, out, report)`. Pasada 1: U' en todas las posiciones válidas → máximo → E por `frexp`. Pasada 2, en el orden del stream (t1, t2) → (x = br(t1), p = br(t2)): recalcula U' (da los mismos bits), cuantiza y escribe directo en el buffer de 32 bits. Así no hace falta un temporal de 128 MiB para U' (el precio es calcular U dos veces). Las posiciones con LDLᴴ inválida (solo con λ = 0) quedan en 0 y no cuentan para E. `report`: E, máximo, pivotes mínimo y máximo, cantidad de inválidas y de saturadas.
- Chequeos: dimensiones (potencias de 2 en n_ro y n2), parámetros (g_l > 0 y finitos, λ ≥ 0, nb en [2, 16]) y `fp_contract_off()` (abajo). Devuelve un `status`, no excepciones.

**Primera verificación** (2026-11-15, scratchpad, `check_ps_sense.py` + un arnés en C++, antes de que existiera la regresión del módulo): 19 casos generados con las funciones de `unfold_ref.py` (`gram`, `ldlh`, `solve_U`, `scale_quantize`, `pack_hw`) y comparados **bit a bit**: U (los `double`, en las posiciones válidas), d, la máscara de válidas, E, el máximo, d_min / d_max, la cantidad de inválidas y de saturadas, K_12 y el buffer completo.
- Casos: Af = 1, 2, 3 y 4; L = 1, 6, 8 y 16; λ = 0, 0,3 y 10⁻⁴·mediana; con y sin 1/K; nb = 16 y 12; los dos layouts; imágenes rectangulares (64 × 32, 16 × 128, 32 × 48 con Af = 3); mapas al azar, al azar con un disco (1687 de 2048 posiciones inválidas con λ = 0), mapas del fantoma 256 y **1024 × 1024 (tamaño real: 8 388 608 palabras)**, con y sin máscara; y dos casos con g elegido para que el máximo quede justo debajo de una potencia de 2 (rint da 2^(nb−1) y satura solo si esa componente es positiva: con los datos al azar pasó en una de las corridas, así que la saturación quedó probada una vez; ⏳ en la regresión, un caso armado a mano que sature siempre).
- En la referencia, U se pone en 0 en las posiciones inválidas antes de `scale_quantize` (si no, `unfold_ref.py` mete NaN en el máximo). Es la misma regla que el C.
- **Resultado: 19/19 idénticos**, en la PC con `-O2 -ffp-contract=off` y con `-O0 -g` (las flags del Makefile de gtest), y **en el A53**: binario de `aarch64-linux-gnu-g++ -O2 -ffp-contract=off -mcpu=cortex-a53 -static` corrido con `qemu-aarch64 -cpu cortex-a53`.
- Tiempo de la sesión completa (1024 × 1024, dos pasadas): 0,7 s en la PC con `-O2`. En qemu tarda 7 s, pero eso no dice nada del A53 real: ⏳ medir en la placa.

**FMA (H29).** GCC 13 para aarch64 junta `a*b + c` en un `fmadd` con `-O2` aunque se compile con `-std=c++17`: `ps_sense.o` tiene 12 instrucciones FMA sin `-ffp-contract=off` y 0 con la flag. Un FMA redondea una vez en vez de dos, así que cambia el último bit y se pierde el bit a bit. Por eso `build_buffer` empieza con `fp_contract_off()`: calcula a·a − (1 + 2⁻²⁹) con a = 1 + 2⁻³⁰, que da 0 sin FMA y 2⁻⁶⁰ con FMA. Si da distinto de 0 devuelve `status::fp_contract_on`. Probado: x86 con `-march=native -ffp-contract=fast` y aarch64 con `-O2` sin la flag, los dos lo rechazan.

**Bare-metal:** `aarch64-none-elf-g++ -O2 -ffp-contract=off -mcpu=cortex-a53 -c` compila sin warnings; el texto ocupa 6,7 KiB y tiene 0 FMA. Compila también con `-Wpedantic -Wconversion -Wshadow -Wdouble-promotion` sin warnings.

**Dos conceptos que usa esta verificación** (explicados a Julián el 2026-11-16):
- **FMA** (*fused multiply-add*): una instrucción que calcula a·b + c con **un solo redondeo**, al final. Sin FMA son dos operaciones y dos redondeos: primero round(a·b), después round(… + c). El FMA es igual o más preciso, pero da **otro resultado** en el último bit, y la referencia (NumPy) hace cada operación por separado. Ejemplo verificado con fracciones exactas: con a = 1 + 2⁻³⁰, a·a = 1 + 2⁻²⁹ + 2⁻⁶⁰. Con dos redondeos, a·a − (1 + 2⁻²⁹) = 0 (el 2⁻⁶⁰ se pierde al redondear el producto); con FMA da 2⁻⁶⁰. El A53 tiene FMA (`fmadd`, ARMv8); el x86-64 base no (llegó con FMA3; hace falta `-mfma` o `-march`). La "contracción" es que el compilador junte por su cuenta a*b + c en un FMA; `-ffp-contract=off` la prohíbe.
- **qemu**: emulador de código abierto. En modo **usuario** (`qemu-aarch64`, el que usamos) corre **un programa** de Linux para ARM en la PC: traduce las instrucciones ARM a x86 mientras corre (traducción binaria dinámica) y le pasa las llamadas al sistema (abrir archivos, imprimir) al Linux de la PC. Por eso el gtest se compila para "ARM con Linux" y estático (en la PC no hay bibliotecas ARM). El punto flotante de cada instrucción ARM se emula exacto (incluido el redondeo único de `fmadd`): por eso detecta lo del H29. No emula tiempos, cachés ni el entorno bare-metal. En modo **sistema** (`qemu-system-aarch64`; AMD mantiene uno para Zynq UltraScale+) emula una máquina entera y podría arrancar un firmware bare-metal; no lo usamos.

**Regresión del módulo** (entrega D1-infra, 2026-11-15; el parche `ps_sense_infra.patch` está aplicado y commiteado en esta PC: `2a0d6d2`, y el renombre de los objetivos en `7a95825`. En la PC con WSL2, en `1600a0d`, todavía no estaba: hace falta `git pull`; decisiones de Julián: gtest con el patrón de la `fft2d`, `Makefile` propio para el A53 con qemu, borrar la plantilla de `create_module`):
- `model/gen_ps_sense_vectors.py --from-json` escribe, por cada caso del JSON (`VEC_DIR` + `vec_gen`), `case.txt` (dimensiones, config y resultados esperados; los `double` en hex, exactos), los mapas, g y las salidas de referencia (U crudo, d, válidas, buffer). Usa `gram`, `ldlh`, `solve_U`, `scale_quantize` y `pack_hw` de `unfold_ref.py`, con mapas rectangulares. Determinista: regenerado da los mismos bytes.
- Casos `g = sat`: el generador asegura que la componente de módulo máximo sea **positiva** (si no, niega los mapas: S → −S da U → −U exacto) y elige g para que el máximo quede en 1 − 2⁻¹⁷. Así rint da 2^(nb−1) y satura siempre (n_sat = 1 en `sat16` y `sat12_cpa`); el generador corta si no satura.
- `testbench/ps_sense_gtest.cpp`: 9 tests por caso. Sin contracción a FMA; consistencia de `case.txt`; núcleo por posición bit a bit (validez, d, U); sesión completa bit a bit (E, máximo, pivotes, contadores, buffer); `cordic_gain`; `word_index` es una permutación en los dos layouts; `bitrev`; `quantize` (empates al par, saturación por arriba y no por abajo, `pack_word`); entradas inválidas (códigos de `status`, mapas en cero → todo inválido, buffer en 0, E = 0). Los vectores se cargan una vez adentro de los tests (no en `SetUpTestSuite`, §8). La carpeta sale de `PS_SENSE_VEC` (variable de entorno) o del define `PS_SENSE_VEC_DIR`.
- JSON: 18 casos en `testbench/vec/` (11 MB, versionados) y 2 de 1024 × 1024 en `build/vec_full/` (unos 600 MB, sin versionar, con `"gtest": {"modes": []}` para que `run_gtest` los saltee). Un solo modo (`fixed`): el módulo es todo `double` y `run_gtest` solo acepta `fixed` y `double`; la etiqueta `[fixed]` no significa nada acá.
- `modules/ps_sense/Makefile`, con los objetivos separados en **compilar** y **correr** (nombres desde el 2026-11-16, a pedido de Julián; antes `bm`, `qemu`, `host` y `full`):
  - `compile-baremetal`: `build/baremetal/libps_sense.a` con `aarch64-none-elf` (lo que va a Vitis) e inspección del código con `objdump`: 0 FMA. No ejecuta nada (el bare-metal solo corre en la placa).
  - `compile-a53-linux`: el mismo gtest para aarch64 con Linux, estático, contra googletest compilado desde `/usr/src/googletest`. `compile-host`: el gtest para la PC con `-O2`. `compile`: los tres.
  - `run-qemu`: el binario de `compile-a53-linux` corrido con `qemu-aarch64 -cpu cortex-a53` sobre cada carpeta. `run-host`: lo mismo en la PC. `run-full`: genera los de 1024 y corre PC + qemu. `run`: `run-host` + `run-qemu`.
  - `clean`: solo lo suyo.
- `scripts/Makefile`: `-ffp-contract=off` en `CXXFLAGS` para todos los módulos. **No cambia nada en los otros módulos**: x86-64 base no tiene FMA, y además se comprobó compilando 28 builds existentes (4 por grupo: `cast` tb, `fft1d_r22sdf` gtest `fixed` / `double` y tb, `fft2d` gtest `fixed` / `double` y tb) con y sin la flag: los 56 objetos dan **código de máquina idéntico** (`objdump -d`).

**Resultados (copia del repo en el scratchpad):**

| Qué | Resultado |
|---|---|
| `run_gtest --keep-going` | 18/18 casos, 162 tests |
| `make run-host` (`-O2`) | 18/18 |
| `make run-qemu` (A53) | 18/18 |
| `make run-full` (1024 × 1024, PC y qemu) | 2/2 y 2/2 |
| `case.txt` con CRLF (clon con `core.autocrlf = true`, como en Windows) | 18/18: `fscanf` toma el `\r` como espacio; los `.bin` quedan idénticos |
| `make compile-baremetal` | OK: 0 FMA, 6721 bytes de código |
| En el repo de Julián (2026-11-16, con los nombres viejos) | `bm` OK (0 FMA, 6721 bytes), `qemu` 18/18; con el parche del generador aplicado, `run_gtest` 18/18 y `full` 2/2 + 2/2 |
| **Re-verificación en un clon de `7a95825`** (2026-11-16, al traer la revisión de la PC con WSL2) | `run_gtest` 18/18 (162 tests, 0 skipped), `make compile` OK (0 FMA, 6721 bytes), `make run` 18/18 + 18/18, `make run-full` 2/2 + 2/2; el generador regenera los 18 casos de `testbench/vec/` idénticos byte a byte; `-ffp-contract=off`: 28 builds de los otros módulos, 56 objetos con código de máquina idéntico (las fuentes de esos módulos no cambiaron desde `874a7d4`) |
| Mutaciones (9: reasociar una suma en `gram`, sacar el bit-reversal, saturación con un LSB de más, layouts cruzados, inválidas en el máximo, conjugado en la sustitución hacia atrás, λ a la mitad, 1/g·1/K reordenado, orden del producto en `ldlh`) | **9/9 detectadas** (de 2 a 18 casos fallan en cada una) |

**Pendiente:**
- Tiempo real en el A53.
- Regla de λ (mediana sobre qué posiciones) y formato de los mapas cuando se calculen en el PS.

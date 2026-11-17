# Diseño del MAC de desplegado y del CORDIC (paso D2)

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Teoría: [teoria_sense.md](teoria_sense.md) §5.2 y [teoria_cordic.md](teoria_cordic.md). Números y verificaciones: [04_precision.md](04_precision.md) §4.9–4.11 y **§4.13**. Referencia bit a bit: `model/mac_cordic_fxp.py`.*

*Creado el 2026-11-16, en la PC con Ubuntu, al terminar la teoría de D2. El trabajo sigue en la PC con WSL2: ver "Cómo retomar" al final.*

**Estado (2026-11-17):** teoría hecha y verificada en NumPy (§4.13). Once decisiones tomadas: las 4 del 2026-11-16 y, el 2026-11-17, P1–P7 de la sección 3 (P4, D por puerto, después de explicar qué es D). `modules/cordic_mag` creado con `create_module` (2026-11-17). **Propuesta del modelo C++ de `cordic_mag` en §4.5, ⏳ a confirmar (C1–C8).** **Paso 1 del plan (parche de `mac_cordic_fxp.py`) hecho, verificado y aplicado** (sección 5; commit `5c614a0`). No hay código C++ ni RTL todavía.

---

## 1. Qué es D2

Los modelos C++ bit a bit de lo que va en la PL después de la `fft2d`: el MAC de desplegado (U'' · A) y la magnitud por CORDIC. Con esto, toda la cadena queda modelada bit a bit en C++, y estos modelos son la referencia del RTL (vm).

```
            ┌──────────────── unfold (un carril) ────────────────┐   ┌──────────── cordic_mag ─────────────┐
A_l, U''_l ─► 1. MAC: P = Σ_l U''_l · A_l ─► 2. M = sat(round(P / 2^s)) ─►─► 3. CORDIC: x ≈ K·|M| ─► 4. out = sat(round(x / 2^G))
 (16 bits)       exacto, 36 bits              22 bits con signo             23 bits                     16 bits sin signo
```

con s = D − G (D lo fija el PS por sesión; G = 5 bits de guarda) y la salida ≈ |m̂|·2^F_OUT (§4.11).

## 2. Decisiones tomadas (Julián, 2026-11-16)

| # | Decisión | Por qué |
|---|---|---|
| 1 | **Dos módulos:** `unfold` (pasos 1 y 2: MAC + ajuste a M) y `cordic_mag` (pasos 3 y 4: CORDIC + salida) | el CORDIC es una pieza independiente y reutilizable; se verifica y se sintetiza por separado |
| 2 | **Modelo rtlsim ciclo a ciclo, de un solo carril** | es el patrón del proyecto (como la FFT) y lo que necesita el vm del RTL; M = 4 depende de la medición 4b y se arma después con copias |
| 3 | **Redondeo *half-up* por defecto, con un parámetro `ROUND_CONV`** | el modo mueve la media solo 0,016 LSB (§4.13): acá los empates son raros, a diferencia de la FFT |
| 4 | **Corregir `model/mac_cordic_fxp.py` antes de usarlo como referencia** (por parche) | sus defaults (`OutCfg`: G = 3, W_M = 20) no son los elegidos (G = 5, W_M = 22) y su docstring tiene la escala sin K |
| 5 (P1, 2026-11-17) | **P de 36 bits** | H31; no tocar `ps_sense` (verificado con −2^15); en el DSP48E2 no cuesta nada |
| 6 (P2, 2026-11-17) | **Interfaz de `unfold`: una muestra de bobina por ciclo, AF productos en paralelo, huecos permitidos, los AF resultados juntos** | la propuesta de la sección 3 |
| 7 (P3, 2026-11-17) | **Latencias provisorias: `unfold` 4 ciclos después de la última bobina, `cordic_mag` N_IT + 2 = 14** | se revisan si el RTL no cierra a 200 MHz |
| 8 (P5, 2026-11-17) | **Modelo C++ con enteros de 64 bits y chequeo de rango contra los anchos del RTL; `run_gtest` solo en modo `fixed`** | `ap_int` envolvería sin avisar; la referencia es el Python bit a bit |
| 9 (P6, 2026-11-17) | **Julián crea `cordic_mag` con `create_module`**; `unfold` ya existe | — |
| 10 (P7, 2026-11-17) | **El código se muestra completo en el chat, archivo por archivo, para que Julián lo lea antes**; para aplicarlo, el parche como archivo (regla de `CLAUDE.md`: copiar del chat rompe los CRLF), verificado en el scratchpad | Julián quiere leerlo antes de que entre al repo |
| 11 (P4, 2026-11-17) | **D por puerto: `i_d`, 6 bits, constante durante la sesión** (s = D − G de −5 a 58) | E sale de los mapas de cada sesión, F_OUT es la calibración de brillo y S puede pasar a ser de ejecución; un parámetro obligaría a resintetizar |

## 3. Propuesta pendiente de confirmar

Lo que propuso Claude el 2026-11-16, con la recomendación. **⏳ Julián tiene que confirmar o cambiar cada punto antes de escribir código.**

| # | Punto | Propuesta (recomendada) | Alternativa |
|---|---|---|---|
| P1 | Ancho de P | **36 bits** = NB_A + NB_U + 1 + ⌈log₂L⌉ (H31) | que el PS no use −2^15 en U'' para quedarse en 35; descartada porque `ps_sense` ya está verificado con −2^15 |
| P2 | Interfaz del carril de `unfold` | **una muestra de bobina por ciclo, AF MAC en paralelo** (el mismo A_l por U''[l][0] y U''[l][1]); una posición = L muestras válidas seguidas en orden de bobina; **huecos permitidos** en `i_valid`; salida de los AF valores juntos | salida serializada (un M por ciclo) |
| P3 | Latencias (provisorias) | `unfold`: `o_valid` **4 ciclos** después de la última bobina; `cordic_mag`: **N_IT + 2 = 14** | partir el redondeo de 36 bits en dos etapas si no cierra a 200 MHz (se ve con el RTL) |
| P4 | D | **por puerto** (`i_d`, 6 bits, de sesión); s = D − G va de −5 a 58; con s ≤ 0 se corre a la izquierda sin redondear y después se satura (como el Python) | D como parámetro fijo |

**Respuestas (2026-11-17):** P1, P2, P3, P5 y P6 como se propusieron; P7, en el chat para leerlo antes (decisión 10); P4, por puerto (decisión 11), después de esta explicación de qué es D: D es el corrimiento total de P a la salida, D = 30 − (S − 19 + E) − F_OUT = 49 − S − E − F_OUT (los 30 son los dos Q1.15 de A y U''; S − 19, la escala de la `fft2d`; E, el exponente de U''; F_OUT, los bits fraccionales de la salida). Con el fantoma, S = 6, E = 15, F_OUT = 15 → D = 13 (lo imprime `mac_output_study.py`). Se reparte: `unfold` corre s = D − G y `cordic_mag` corre G. Va por puerto porque E sale de los mapas de cada sesión, F_OUT es la calibración de brillo y S cambia si las máscaras pasan a ser de ejecución; no se puede meter 2^−D en U'' porque U'' ya usa todo su rango (eso es E): con 2^−13 le quedarían 3 bits útiles. Abierto: si con datos reales puede hacer falta D < 0 (con S = 6 y F_OUT = 15 haría falta E > 28).
| P5 | Tipos del modelo | **enteros de 64 bits + chequeo de rango** contra el ancho del RTL (si los anchos están bien, el modelo es exacto; si no, el test lo detecta); en `run_gtest`, **solo el modo `fixed`** | `ap_int` con los anchos del RTL (envuelve sin avisar) |
| P6 | Cómo se crean los módulos | Julián crea `cordic_mag` con `create_module` (como `ps_sense`); `unfold` ya existe (plantilla vacía) | — |
| P7 | Cómo se entrega el código | ⏳ sin decidir: directo en `modules/unfold/` y `modules/cordic_mag/`, o todo por parche | — |

*Nota (2026-11-17):* `CLAUDE.md` ya dice "las entregas y git los manejo yo; pasame el parche como archivo", lo que apunta a **todo por parche** en P7, salvo que Julián autorice escribir directo (como en D1 con `ps_sense`). Falta que lo confirme.

## 4. Especificación (según la propuesta)

### 4.1 Números (verificados en NumPy, §4.13)

| Señal | Ancho | Por qué |
|---|---|---|
| A_l, U'' | 16 + 16 bits con signo (Q1.15) | salida de la `fft2d`; buffer del PS |
| Un producto complejo | re: 32 bits; **im: 33** | ur·ai + ui·ar llega a +2^31 con las cuatro en −2^15 |
| P (8 bobinas) | **36 bits** con signo | P_im llega a 2^34 (H31); P_re cabe en 35 |
| M | 17 + G = **22 bits** con signo | suficiente: solo recorta lo que igual satura la salida (umbral \|M\| ≥ 0,607·2^(16+G)); mínimo: con 21 se recortarían valores que no saturan |
| x del CORDIC | **23 bits** sin signo (NB_M + 1) | llega a K·√2·2^21 = 4 883 994 |
| y del CORDIC | **23 bits** con signo (NB_M + 1) | queda en [−2^21, 2^21] |
| Salida | 16 bits sin signo | ≈ \|m̂\|·2^F_OUT |

Error de la cadena (n = 12, G = 5, *half-up*): media +0,025 LSB, rms 0,295, máximo 0,79 (ideal: rms 0,289, máximo 0,5).

### 4.2 `unfold` (un carril)

- **Parámetros:** `L` = 8, `AF` = 2, `NB_A` = 16, `NB_U` = 16, `G` = 5, `NB_M` = 17 + G, `ROUND_CONV` = 0. Derivado: `NB_P` = NB_A + NB_U + 1 + ⌈log₂L⌉ = 36. Parámetros nuevos siempre al final (regla del proyecto).
- **Puertos:** `i_valid`, `i_a` (re, im), `i_u[AF]` (re, im; mismo formato {im, re} que el buffer de U''), `i_d` (6 bits), `o_valid`, `o_m[AF]` (re, im de NB_M bits).
- **Comportamiento:** un contador de muestras válidas sabe en qué bobina va (0 … L − 1). La primera bobina de una posición carga el acumulador y las siguientes suman. Con la L-ésima: M_a = sat_{NB_M}(round(P_a / 2^s)) y `o_valid`.
- **Pipeline:** registro de entrada → productos → acumulador → redondeo y saturación (4 ciclos después de la última bobina).
- **Recursos (estimado, para el RTL):** 4 DSP por producto complejo y por a → 8 DSP por carril con AF = 2; el acumulador entra en los 48 bits del DSP48E2 (UG579).

### 4.3 `cordic_mag`

- **Parámetros:** `NB_M` = 22, `G` = 5, `N_IT` = 12, `NB_OUT` = 16, `ROUND_CONV` = 0. Derivado: ancho interno NB_M + 1 = 23.
- **Puertos:** `i_valid`, `i_m` (re, im de NB_M bits), `o_valid`, `o_mag` (NB_OUT bits sin signo).
- **Comportamiento:** x₀ = |Re M|, y₀ = |Im M|; en cada iteración i: si y ≥ 0, x += y >> i e y −= x >> i; si no, al revés (los dos lados con los valores de antes; `>>` aritmético). Salida: sat_u(round(x_n / 2^G)).
- **Pipeline:** valor absoluto → N_IT iteraciones → salida: latencia N_IT + 2. Un dato por ciclo, huecos permitidos.

### 4.5 Modelo C++ de `cordic_mag`: propuesta (2026-11-17) ⏳ a confirmar

La plantilla de `create_module` no tiene archivos de más: los runners exigen `cpp/<mod>.cpp` aunque el modelo sea *header-only* (`run_gtest` y `run_regression_sim` lo pasan en `EXTRA_SRCS`; queda como en la FFT, con un comentario), `tb.{cpp,hpp}` son de `run_regression_sim`, `tb.sv` y `rtl/` del vm, los *wrappers* y JSON de síntesis e implementación se usan después, y `flist/` está en el `.gitignore` (lo genera `update_flist`). Lo que sobra es el contenido de ejemplo de cada archivo, que se reemplaza (el `#include <ap_fixed.h>` del modelo también: P5). El `cordic_mag_gtest.cpp` no lo crea la plantilla (la variable sí).

| # | Punto | Propuesta |
|---|---|---|
| C1 | Flujo de vectores | `model/gen_cordic_mag_vectors.py --from-json` escribe por caso `testbench/vec/<caso>/` (versionado, como `ps_sense`): `case.txt` (configuración con la que se generó), `m.bin` (int32 [n][re, im]), `xn_ref.bin` y `out_ref.bin` (int32 [n]), con `mac_cordic_fxp.m_to_out`. Un header común, `testbench/cordic_mag_stimulus.hpp`, lee el caso y arma el estímulo ciclo a ciclo con huecos; lo usan el tb (sim → vectores del vm) y el gtest. |
| C2 | Parámetros (macros `CORDIC_MAG_*`) | `NB_OUT` = 16, `G` = 5, `N_IT` = 12, `NB_M` = NB_OUT + 1 + G (= 22, misma regla que el Python), `ROUND_CONV` = 0. Locales: `NB_W` = NB_M + 1 = 23 (x e y), `LATENCY` = N_IT + 2. En el RTL, en ese orden, para que `NB_M` se pueda derivar de los anteriores. |
| C3 | Puertos | `i_valid`, `i_m_re`, `i_m_im` (NB_M con signo), `o_valid`, `o_mag` (NB_OUT sin signo); `int64_t` (P5). Sin reset, como la FFT: registros con valor inicial 0. |
| C4 | Pipeline | 14 registros: \|·\| → N_IT iteraciones → redondeo y saturación. Un `valid` por etapa; los datos se cargan solo con su `valid` (*clock enable*): `o_mag` mantiene el último valor entre salidas. Importa porque el vm compara todos los ciclos, válidos o no. |
| C5 | Chequeos de rango (P5) | `i_m` en NB_M bits con signo; x en [0, 2^NB_W) e y en NB_W bits con signo en cada iteración; si no, excepción con ciclo, etapa y valor. `static_assert` de los parámetros y de que `>>` sea aritmético (en C++17 es de la implementación). |
| C6 | Qué chequea el gtest por caso | que `case.txt` coincida con los parámetros compilados (si no: "regenerá los vectores"); ningún chequeo de rango disparado; tantas salidas como entradas; cada salida exactamente LATENCY ciclos después de su entrada; `o_mag` = Python bit a bit, y también x_n (por un accesor de depuración, no un puerto); `o_mag` quieto entre salidas. |
| C7 | Casos | al azar en todo el rango de M (incluida la zona que satura); esquinas (±2^21 en las cuatro combinaciones, −2^21 en un solo eje), cero, ejes, valores chicos (\|M\| < 2^G); el borde de saturación (\|M\| ≈ 0,607·2^(16+G)); empates del redondeo de salida (x_n con los G bits bajos = 2^(G−1)), en *half-up* y convergente; huecos al azar (con datos basura en `i_m` mientras `i_valid` = 0); n/G = 8/4 y 16/0 (NB_M = 21 y 17); NB_OUT = 12. |
| C8 | `N_CYCLES` | 0 = automático en el tb (como la `fft2d`): entradas + huecos + LATENCY + cola. Para el vm se resuelve en el paso del RTL. |

Orden del código en el chat (decisión 10): `cpp/cordic_mag.hpp` → generador → `cordic_mag_stimulus.hpp` y gtest → tb y JSON.

### 4.4 Consecuencias para el sistema (van a [07 §5](07_integracion.md) cuando se confirme)

- **El MAC necesita las L bobinas de una misma posición seguidas.** La IFFT deja una imagen por bobina en la DDR; leerlas intercaladas posición por posición rompe las ráfagas (N½ ≈ 3 KiB, §7). Hace falta un reordenamiento por bloques antes del MAC (p. ej. 1024 posiciones × 8 bobinas en un ping-pong de BRAM, unos 64 KiB). Es del paso 6.
- **Alcanza un solo CORDIC para todo el MAC:** dos carriles dan 4 valores cada 8 ciclos (0,5 por ciclo); con un serializador adelante, un CORDIC sobra.

## 5. Plan de verificación

1. ✅ **Parche de `mac_cordic_fxp.py`** (2026-11-17, `d2_mac_cordic_fxp.patch`, verificado en el scratchpad; aplicado por Julián en `5c614a0`, self-test 24/24 en su repo): defaults G = 5 y W_M = 17 + G; opción `round_conv`; docstring con la escala m̂ / K_n; el self-test prueba la esquina de 36 bits (P_im = 2^34). Verificar que el self-test siga pasando y que `mac_output_study.py` dé los mismos números de §4.11 (pasando G y W_M explícitos).
   - **Qué hace el parche.** `OutCfg`: G = 5, N_IT = 12, W_OUT = 16 y **W_M = None → W_OUT + 1 + G** (= 17 + G = 22; es la cuenta de §4.13 escrita en general: la salida satura con |M| ≥ 0,607·2^(W_OUT+G)); explícito se respeta. Redondeo por paso, con dos campos nuevos al final: **`ROUND_CONV_M`** (P → M, el `ROUND_CONV` de `unfold`) y **`ROUND_CONV_OUT`** (x_n → salida, el de `cordic_mag`), los dos en 0 (*half-up*). `round_shift(x, s, conv)` con empates al par igual que `cast.v` / `fft1d_r22sdf_fxp._cast`: m + (r > h) + (r == h)·lsb(m). Dos funciones que espejan los módulos: `p_to_m` (paso 2) y `m_to_out` (pasos 3 y 4); `output_stage` las compone. `round_shift` da `OverflowError` si el corrimiento a la izquierda (s ≤ 0) o el + 2^(s−1) no entran en int64 (numpy envolvería sin avisar). Docstring: rangos exactos de P (H31), NB_P = NB_A + NB_U + 1 + ⌈log₂L⌉, por qué W_M es suficiente y mínimo, y la escala m̂ / K_n = 2^(S−19+E)·P·2^−30 → salida ≈ |m̂|·2^F_OUT. En `mac_output_study.py`, solo la etiqueta "cota exacta 16 + 16 + 3 = 35" pasa a 36 (H31).
   - **Self-test (24 chequeos, todos OK):** defaults; MAC exacto; rango del producto por los 16 vértices (es multilineal, así que los extremos están en los vértices: es una prueba, no una muestra); esquina H31 con `mac()` (P_re = 0, P_im = 2^34, 36 bits); `round_shift` contra `Fraction` exacto en los dos modos, s = 1 … 58, con empates forzados; s ≤ 0 y el desborde; CORDIC en todo el rango de M (±2^21) con n = 8, 12, 16; saturación: con W_M = 22 recorta 156 424 valores de Re M y la salida es idéntica a la de W_M = 40, con W_M = 21 cambian 130 957 salidas (22 es el mínimo); escala con U' al azar (S = 6, E = 12, D = 16): 2^(S−19+E)·P·2^−30 difiere de m̂ / K_n en 5·10⁻⁵ y de m̂ en 0,393 = 1 − 1/K_n, salida − |m̂|·2^F_OUT con media +0,032 LSB; cadena contra K_n·|P|·2^−D: media +0,033, rms 0,295, máx 0,69 LSB; los 4 modos de redondeo van a su paso, y convergente/convergente baja la media 0,0157 LSB (§4.13: 0,016).
   - **Mutaciones 9/9 detectadas** (banderas cruzadas o ignoradas, W_M − 1, empates al impar, `>=` en el convergente, corrimiento a la izquierda perdido, saturación a W_M − 1, signo de P_im, *half-up* → *floor*).
   - **`mac_output_study.py`, antes y después del parche: salida idéntica** (salvo el tiempo y la etiqueta). **Ojo (H32):** desde el 2026-11-15 el estudio hereda `ROUND_CONV = 1` de la `fft2d` y ya no da la tabla de §4.11 (n = 12, G = 5: 0,80 / 0,309 / +0,093 / 52,01 dB, en vez de 0,81 / 0,31 / +0,09 / 51,96). Con la `fft2d` en *half-up* (`CORE` con `ROUND_CONV = 0`, inyectado desde un script del scratchpad) **reproduce la tabla de §4.11 exacta, antes y después del parche**.
2. **Generador de vectores** desde el JSON de cada módulo (como `gen_ps_sense_vectors.py`), con `mac_cordic_fxp.py` como referencia.
3. **Casos de `unfold`:** al azar; la esquina de 2^34; s de −5 a 58 (incluidos 0 y los extremos); saturación positiva y negativa de M; huecos en `i_valid`; varias posiciones seguidas; `ROUND_CONV` 0 y 1; otros L y AF.
4. **Casos de `cordic_mag`:** todo el rango; las esquinas ±2^21; cero y los ejes; saturación de la salida; `ROUND_CONV`; n/G = 8/4 y 16/0 para reproducir la tabla de §4.11.
5. **Punta a punta con el fantoma:** `unfold` → `cordic_mag` contra `output_stage` de Python, bit a bit.
6. Después, con el RTL: sim y vm (el modelo genera los vectores), síntesis e implementación.

## 6. Cómo retomar (en la PC con WSL2)

1. En la PC con Ubuntu, antes de irse: commitear `doc/`, aplicar el parche de `CLAUDE.md` y hacer `git push`.
2. En la PC con WSL2: `git stash push -- doc/` (la revisión de allá ya está rehecha acá), `git pull`, y mover los manuales: `mkdir -p books/amd-docs && mv doc/amd/*.pdf books/amd-docs/`.
3. Leer `doc/estado.md` y este archivo.
4. Confirmar o cambiar los puntos P1–P7 de la sección 3, y anotarlo acá (sección 2) y en `estado.md`. *(2026-11-17: confirmados todos salvo P4.)*
5. Seguir el plan de la sección 5. *(2026-11-17: paso 1 hecho y commiteado (`5c614a0`). Orden acordado para seguir: primero `cordic_mag`, que no depende de P4 (Julián lo crea con `create_module`); antes del código, el diseño del modelo en el chat (clase, pipeline, JSON, formato de vectores, casos); después, archivo por archivo en el chat: modelo `cpp/cordic_mag.hpp`, generador de vectores, gtest; verificado en el scratchpad y entregado como parche. Después `unfold`, con P4 cerrado, y la prueba punta a punta.)*

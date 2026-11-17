# Diseño del MAC de desplegado y del CORDIC (paso D2)

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Teoría: [teoria_sense.md](teoria_sense.md) §5.2 y [teoria_cordic.md](teoria_cordic.md). Números y verificaciones: [04_precision.md](04_precision.md) §4.9–4.11 y **§4.13**. Referencia bit a bit: `model/mac_cordic_fxp.py`.*

*Creado el 2026-11-16, en la PC con Ubuntu, al terminar la teoría de D2. El trabajo sigue en la PC con WSL2: ver "Cómo retomar" al final.*

**Estado:** teoría hecha y verificada en NumPy (§4.13). Cuatro decisiones tomadas; la interfaz, el pipeline y la forma del modelo están **propuestos y falta que Julián los confirme** (sección 3). No hay código todavía.

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

## 3. Propuesta pendiente de confirmar

Lo que propuso Claude el 2026-11-16, con la recomendación. **⏳ Julián tiene que confirmar o cambiar cada punto antes de escribir código.**

| # | Punto | Propuesta (recomendada) | Alternativa |
|---|---|---|---|
| P1 | Ancho de P | **36 bits** = NB_A + NB_U + 1 + ⌈log₂L⌉ (H31) | que el PS no use −2^15 en U'' para quedarse en 35; descartada porque `ps_sense` ya está verificado con −2^15 |
| P2 | Interfaz del carril de `unfold` | **una muestra de bobina por ciclo, AF MAC en paralelo** (el mismo A_l por U''[l][0] y U''[l][1]); una posición = L muestras válidas seguidas en orden de bobina; **huecos permitidos** en `i_valid`; salida de los AF valores juntos | salida serializada (un M por ciclo) |
| P3 | Latencias (provisorias) | `unfold`: `o_valid` **4 ciclos** después de la última bobina; `cordic_mag`: **N_IT + 2 = 14** | partir el redondeo de 36 bits en dos etapas si no cierra a 200 MHz (se ve con el RTL) |
| P4 | D | **por puerto** (`i_d`, 6 bits, de sesión); s = D − G va de −5 a 58; con s ≤ 0 se corre a la izquierda sin redondear y después se satura (como el Python) | D como parámetro fijo |
| P5 | Tipos del modelo | **enteros de 64 bits + chequeo de rango** contra el ancho del RTL (si los anchos están bien, el modelo es exacto; si no, el test lo detecta); en `run_gtest`, **solo el modo `fixed`** | `ap_int` con los anchos del RTL (envuelve sin avisar) |
| P6 | Cómo se crean los módulos | Julián crea `cordic_mag` con `create_module` (como `ps_sense`); `unfold` ya existe (plantilla vacía) | — |
| P7 | Cómo se entrega el código | ⏳ sin decidir: directo en `modules/unfold/` y `modules/cordic_mag/`, o todo por parche | — |

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

### 4.4 Consecuencias para el sistema (van a [07 §5](07_integracion.md) cuando se confirme)

- **El MAC necesita las L bobinas de una misma posición seguidas.** La IFFT deja una imagen por bobina en la DDR; leerlas intercaladas posición por posición rompe las ráfagas (N½ ≈ 3 KiB, §7). Hace falta un reordenamiento por bloques antes del MAC (p. ej. 1024 posiciones × 8 bobinas en un ping-pong de BRAM, unos 64 KiB). Es del paso 6.
- **Alcanza un solo CORDIC para todo el MAC:** dos carriles dan 4 valores cada 8 ciclos (0,5 por ciclo); con un serializador adelante, un CORDIC sobra.

## 5. Plan de verificación

1. **Parche de `mac_cordic_fxp.py`:** defaults G = 5 y W_M = 17 + G; opción `round_conv`; docstring con la escala m̂ / K_n; el self-test prueba la esquina de 36 bits (P_im = 2^34). Verificar que el self-test siga pasando y que `mac_output_study.py` dé los mismos números de §4.11 (pasando G y W_M explícitos).
2. **Generador de vectores** desde el JSON de cada módulo (como `gen_ps_sense_vectors.py`), con `mac_cordic_fxp.py` como referencia.
3. **Casos de `unfold`:** al azar; la esquina de 2^34; s de −5 a 58 (incluidos 0 y los extremos); saturación positiva y negativa de M; huecos en `i_valid`; varias posiciones seguidas; `ROUND_CONV` 0 y 1; otros L y AF.
4. **Casos de `cordic_mag`:** todo el rango; las esquinas ±2^21; cero y los ejes; saturación de la salida; `ROUND_CONV`; n/G = 8/4 y 16/0 para reproducir la tabla de §4.11.
5. **Punta a punta con el fantoma:** `unfold` → `cordic_mag` contra `output_stage` de Python, bit a bit.
6. Después, con el RTL: sim y vm (el modelo genera los vectores), síntesis e implementación.

## 6. Cómo retomar (en la PC con WSL2)

1. En la PC con Ubuntu, antes de irse: commitear `doc/`, aplicar el parche de `CLAUDE.md` y hacer `git push`.
2. En la PC con WSL2: `git stash push -- doc/` (la revisión de allá ya está rehecha acá), `git pull`, y mover los manuales: `mkdir -p books/amd-docs && mv doc/amd/*.pdf books/amd-docs/`.
3. Leer `doc/estado.md` y este archivo.
4. Confirmar o cambiar los puntos P1–P7 de la sección 3, y anotarlo acá (sección 2) y en `estado.md`.
5. Seguir el plan de la sección 5, empezando por el parche de `mac_cordic_fxp.py`.

# Estudio de precisión

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).* Resumen de hallazgos: [hallazgos.md](hallazgos.md).

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
- **Se corrige con redondeo convergente** (*round-half-even*). Al descartar un bit: `resultado = m + (bit_descartado AND lsb(m))`, una compuerta por mariposa. UG579 (v1.11) describe cómo hacerlo en el DSP48E2 con el *pattern detector* (Cap. 2, "PATTERNDETECT and PATTERNBDETECT Logic", p. 36; Cap. 3, Tabla 3-3, `SEL_MASK = ROUNDING_MODE1/2`, p. 57), pero las mariposas están en fabric, así que acá serían LUTs.
- **Medido en el modelo bit a bit (2026-10-26, `model/fft_rounding_study.py`)** ✅ modelo Python. Regla general, al descartar d bits de x con m = ⌊x / 2^d⌋ y r = x − m·2^d:

  ```
  half-up (hoy):  m + (r ≥ 2^(d−1))
  convergente:    m + (r > 2^(d−1)) + (r = 2^(d−1))·lsb(m)
  d = 1 (mariposa):  m + (b0 AND lsb(m))
  d = 4 (salida):    m + (b3 AND (b2 OR b1 OR b0 OR lsb(m)))
  ```

  | Medida | *half-up* | conv. mariposas | conv. mariposas + salida | conv. en todo |
  |---|---|---|---|---|
  | 1D 20/18/16, N = 1024, máscara completa, ruido: sesgo Re / Im | +0,062 / +0,061 LSB | +0,044 / +0,042 | 0,000 / −0,001 | igual |
  | 1D: rms; sesgo² / MSE | 0,419; 4,4 % | 0,414; 2,2 % | 0,414; 0,0 % | igual |
  | Fantoma 1024 × 512, 801 / 384: fila p = 0 | +7,54 / +7,51 | +5,62 / +5,58 | +0,017 / +0,008 | igual |
  | Fantoma: resto de las filas | +0,055 | +0,046 | +0,001 | igual |
  | Fantoma: SNR de la imagen desplegada | 51,03 dB | 51,06 | 51,12 | igual |
  | Ida y vuelta Shepp 256: SQNR; error (0, 0) | 54,31 dB; 729 LSB | 54,45; 474 | 54,46; 3 | igual |
  | Ida y vuelta Shepp 64: error (0, 0) | 47 | 31 | 5 | igual |
  | Ida y vuelta seno 256: error (0, 0); saturaciones | 3 448; 2 | 3 347; 2 | 2 853; 0 | igual |
  | Ida y vuelta anillos 256: error (0, 0) | 490 | 854 | 1 350 | igual |

  - **El `cast` de salida (20 → 16, descarta 4 bits) pone 2/3 del sesgo**; las mariposas, 1/3. Convergente solo en las mariposas no alcanza. El `cmul` (descarta 17 bits) no aporta: con convergente ahí los números son idénticos (un empate exacto en 17 bits casi nunca pasa).
  - **El error en (0, 0) del seno y los anillos no es sesgo.** Con convergente, sale exacto de la media del error del espectro después de la ida: +0,087 LSB en el seno y −0,041 en los anillos (por 2^16 / 2^(S3+S4)), contra +0,0004 en Shepp. Con twiddles de 24 bits da lo mismo (no son los twiddles). Con redondeo estocástico (sin sesgo e independiente de la señal) baja a 48 y 47 LSB, a costa de 3 a 7 dB de SQNR. Es **error de redondeo correlacionado con la señal**, propio de imágenes sintéticas con espectro determinístico; en los anillos el sesgo *half-up* lo compensaba en parte. Con datos reales el ruido térmico hace de *dither*: se espera que se comporten como Shepp. Por eso la medida de B es Shepp (0, 0) y la fila p = 0, no el seno ni los anillos.
  - **`ap_fixed` con `AP_RND_CONV` y `AP_SAT` = esta regla**, verificado exhaustivo (21 → 20 bits, 2²¹ valores; 20 → 16, 2²⁰; 17 → 16, 2¹⁷): 0 diferencias. El modelo C++ lo puede usar tal cual.
  - Costo en hardware: el convergente necesita los bits bajos de la suma para decidir el incremento, así que no se puede absorber como una constante; cae en el camino de la BF2I escalada (crítico con 801 / 384 y N = 512, §5.6). Se mide en C.
- **Implementado en el núcleo 1D (entrega 1 del paso B, 2026-10-26)** ✅ XSim. Error contra la FFT ideal en LSB de salida, una trama por caso, medido sobre los vectores de sim de cada par de casos (el *half-up* y su `_conv_`):

  | Par | Sesgo Re / Im *half-up* | Sesgo Re / Im convergente | rms *half-up* → convergente |
  |---|---|---|---|
  | N = 64, Q1.15, ruido | +0,594 / +0,500 | +0,031 / −0,016 | 1,031 → 0,654 |
  | N = 1024, Q1.15, ruido | +0,511 / +0,484 | +0,020 / −0,001 | 1,008 → 0,669 |
  | N = 1024, 20/18/16, ruido | +0,059 / +0,062 | −0,010 / −0,000 | 0,422 → 0,417 |
  | N = 1024, 20/18/16, 784, multitono | +0,053 / +0,052 | −0,004 / +0,004 | 0,425 → 0,421 |
  | N = 1024, 20/18/16, **801**, multitono | +0,048 / +0,048 | +0,004 / −0,007 | 0,409 → 0,405 |
  | N = 512, 20/18/16, ruido | +0,074 / +0,039 | −0,002 / −0,025 | 0,417 → 0,412 |
  | N = 512, 20/18/16, **384**, multitono | +0,045 / +0,062 | +0,004 / −0,002 | 0,423 → 0,423 |

  - El sesgo desaparece en todos, incluidas las máscaras de SENSE (lo que queda, ±0,02, es la incertidumbre de una trama).
  - **Con 16 bits de datapath, el convergente baja el error rms de ~1,0 a ~0,66 LSB (−3,7 dB)**: ahí casi la mitad del error era sesgo (tabla de arriba). Con 20 bits el rms casi no cambia (−0,1 dB): el beneficio es sacar el error sistemático que en 2D se concentra. Conclusión para la tesis: el redondeo convergente vale más cuanto más angosto es el datapath.
- **En la `fft2d` y la ida y vuelta (entrega 2 del paso B, 2026-10-26)** ✅ modelo C++ (= cadena de Python bit a bit). Es la medida que se eligió para B:

  | Caso | *Half-up* | Convergente |
  |---|---|---|
  | Fantoma 64 × 32 (8 bobinas, 25 / 4): sesgo de la fila p = 0, Re / Im | +1,15 / +1,05 LSB | +0,06 / 0,00 LSB |
  | Fantoma 64 × 32: error máximo; SNR de la imagen desplegada | 4,99 LSB; 71,33 dB | 3,82 LSB; 71,37 dB |
  | Ida y vuelta Shepp 256: error en (0, 0); error máximo; SQNR | 729 LSB; 22,9 LSB; 54,31 dB | 3 LSB; 6,4 LSB; 54,46 dB |
  | Ida y vuelta Shepp 64: error en (0, 0) | 47 LSB | 5 LSB |
  | Shepp 64 con el peor reparto (ida completa, vuelta sin escalar): error en (0, 0) | 375 LSB | 24 LSB |
  | Seno rotado 256 con la regla | 2 saturaciones (no podía ser caso) | 0 saturaciones; (0, 0) sigue en 2 853 LSB (correlación con la señal, no sesgo) |

  La línea de la fila p = 0 de SENSE y el pico de (0, 0) de la ida y vuelta desaparecen; en Shepp 256 el error máximo de toda la imagen baja a un cuarto. La SNR global casi no cambia: lo que se gana es sacar un error con estructura. ⏳ Falta el fantoma de tamaño completo (1024 × 512, 801 / 384) con convergente en el modelo C++ (`fft2d_check_full.py` todavía no pasa `ROUND_CONV`); el modelo de Python ya dio +7,5 → +0,02 LSB (tabla de arriba).

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

### 4.12 Ida y vuelta FFT 2D → IFFT 2D: reparto de la escala (2026-10-25) ✅ modelo C++

Estudio con `fft1d_r22sdf_fxp.py` (20/18/16, N = 256, imágenes de `gen_fft2d_stim.py`, reordenamiento ideal entre las dos transformadas). Script: `model/fft2d_roundtrip_study.py` (`-N`, `--images`; reproduce la tabla de abajo, también en el repo). Casos de regresión: al final de esta sección.

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

**Por qué al principio: cota de los picos internos** ✅ (2026-10-25, `model/fft1d_scaling_bound.py`). Dentro de una pasada, con las S divisiones en las primeras S etapas:

```
pico de cualquier etapa interna  ≤  máx( pico de la entrada , pico de la salida de la pasada )
(en módulo, salvo el redondeo)
```

- Etapas que dividen: (a ± b)·w / 2 con |w| = 1 es un promedio; no supera al mayor de los dos.
- Etapas que no dividen: después de k etapas de una DIF quedan 2^k DFT independientes de N/2^k puntos. Cada valor intermedio es la IDFT (con su 1/M) de un bloque de la salida, así que no supera el pico de la salida.
- Consecuencia: **alcanza con chequear las fronteras** (lo que hace el estudio). Dividiendo al final no hay garantía: las primeras etapas suman sin dividir.
- Verificado: DIF en float con N = 16 a 1024, 8 señales y todos los S, cociente máximo 1,000. Bit a bit (`fft1d_r22sdf_fxp.py`, 20/18/16, N = 64, 512 con tail y 1024; Shepp, ruido y chirp, S mínimo para que la salida quede < 1): al principio 0 saturaciones; al final, con el mismo S, satura siempre (59 a 68 796 componentes) aunque la salida tenga pico 0,9. Chirp en float, N = 256, S = 4: pico interno 14,4 al final contra 0,9 al principio.

**Regla adoptada (Julián, 2026-10-25) para los casos de ida y vuelta:** S mínimo acumulado en cada frontera, divisiones al principio de cada pasada, y el generador exige 0 saturaciones en las 4 pasadas (bit a bit) y S1 + S2 + S3 + S4 = 2·log2 N. "Lo justo" no deja margen: el criterio "< 1" sale de la FFT en float y el redondeo suma algunos LSB, por eso el chequeo es bit a bit.
- **No es la regla de la máscara de SENSE.** 801 (etapas 0, 5, 8 y 9) y 384 (7 y 8) dividen casi todo al final y la cota no las cubre. Con el fantoma tienen margen ≥ 6,5 dB (§4.8); con fastMRI (§9, paso 4) conviene medir la saturación interna y compararlas con sus equivalentes al principio (15 y 3).
- **Seno rotado en N = 256 con la regla: satura 2 componentes** en la pasada 4 (una en la etapa 7, otra en el `cast` de salida), en el píxel (0, 0). La imagen vale 0,9 ahí y el sesgo *half-up* le suma +3 276 LSB, así que llega a 1. Con máscaras no se arregla (la ganancia total es 1): es la medida para el redondeo convergente (§9, B). En N = 64 no pasa.

**Casos de regresión `rt*`** (entrega `fft2d_idavuelta`). `gen_fft2d_stim.py` (familia `roundtrip`) arma la imagen, le hace las pasadas 1 y 2 bit a bit con `fwd_masks` y escribe el espectro en orden natural como estímulo (`STIM_TYPE = 3`) de un caso de la `fft2d` con `INVERSE = 1`, `SCALE_MASK1` = m3 y `SCALE_MASK2` = m4. Con esas máscaras corre las 4 pasadas y aplica el chequeo de la regla (si falla, `--from-json` sale con error); avisa si las máscaras no son las de la regla. `run_plot_fft2d` agrega una fila: original, reconstruida, error contra la original, y dos chequeos (el estímulo es el espectro regenerado; la salida del C++ es, bit a bit, la de la cadena de Python). Parámetros de las imágenes: los del estudio.

| Caso | m1, m2 / m3, m4 | Ida y vuelta | \|error(0, 0)\| | Vuelta sola contra la IDFT 2D |
|---|---|---|---|---|
| `rt64_shepp` | 15, 31 / 1, 3 | 65,12 dB | 47 LSB | 76,6 dB |
| `rt64_sine_rot25` | 31, 63 / 0, 1 | 63,75 | 169 | 81,9 |
| `rt64_rings` | 63, 31 / 1, 0 | 65,92 | 233 | 76,4 |
| `rt64_zoneplate` | 7, 7 / 7, 7 | 93,56 | 2 | 93,8 |
| `rt64_shepp_full` (contraste: ida completa, vuelta sin escalar) | 63, 63 / 0, 0 | 47,52 | 375 | 66,0 |
| `rt256_shepp` (sin gtest, como los `img256`) | 63, 127 / 1, 3 | 54,31 | 729 | 71,6 |

- Todos: 0 saturaciones, estímulo = espectro regenerado y **salida del modelo C++ = cadena de Python bit a bit**. Los números de ida y vuelta son los del estudio.
- **Con redondeo convergente** (2026-10-26, entrega 2 del paso B): `rt64_shepp_conv`, `rt64_shepp_full_conv`, `rt256_shepp_conv` y `rt256_sine_rot25_conv`. Con `ROUND_CONV = 1` en el caso, el generador hace también la ida con convergente (si no, el sesgo de la ida seguiría en (0, 0)). Resultados en §4.6.
- El error de la ida y vuelta viene casi todo de la ida: la vuelta sola da entre 10,5 y 18,5 dB más (salvo la *zone plate*, donde las dos son iguales).
- Verificado en el repo (2026-10-25): gtest 28/28 (168 tests), sim de los 6 casos `rt*` 6/6, plot 6/6. Los casos de antes no cambian: estímulos idénticos byte a byte y plots idénticos (texto y PNG). Falta el vm: el RTL de la `fft2d` todavía no existe.

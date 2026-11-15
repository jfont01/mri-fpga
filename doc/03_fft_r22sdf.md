# FFT 1D R2²SDF: diseño

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).* Teoría: [teoria_fft.md](teoria_fft.md).

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
- **`ROUND_CONV`** (2026-10-26, parámetro al final, default 0 en `cast.v`; el núcleo lo usa en 1 por default desde el 2026-11-15): con 1, los empates van al par (redondeo convergente) en vez de siempre para arriba. Lógica: `sube = round_bit AND (sticky OR lsb)`, con round_bit el bit más alto de los que se descartan, sticky el OR del resto y lsb el bit más bajo de lo que queda. Equivale a `ap_fixed<…, AP_RND_CONV, AP_SAT>`. Guarda: `ROUND_CONV = 1` exige `ROUND_MODE = 1` (doble `$fatal` y `static_assert`). Verificado: barrido exhaustivo del RTL en XSim en 11 formatos (incluidos 21 → 20 y 20 → 16, los de la FFT, con 2²¹ y 2²⁰ valores), 0 diferencias contra la definición; regresión de `cast` con 14 casos (8 nuevos), sim y vm 14/14 (copia y repo); los 6 casos viejos, idénticos byte a byte.
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

Con `ROUND_CONV = 1` (2026-10-26) las mariposas que escalan y la salida redondean al par en los empates, sin sesgo; el `cmul` sigue *half-up* (descarta NBF_TW bits y un empate exacto casi no ocurre). **Default 1 desde el 2026-11-15** (entrega 4), después de medir que cierra a 200 MHz con las máscaras de SENSE (§5.6).

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

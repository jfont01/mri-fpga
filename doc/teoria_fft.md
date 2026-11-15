# Teoría de la FFT: R2²SDF, IFFT, FFT 2D y arquitecturas comparadas

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Diseño concreto del RTL: [03_fft_r22sdf.md](03_fft_r22sdf.md); precisión: [04_precision.md](04_precision.md). Las ecuaciones están en LaTeX: se ven con la vista previa de Markdown (`Ctrl+Shift+V`).*

**Trazabilidad.** Cada identidad de este documento está verificada en NumPy (se indica el error) o leída del paper original: S. He y M. Torkelson, *A New Approach to Pipeline FFT Processor*, 1996 (`books/papers/fft/A New Approach to Pipeline FFT Processor.pdf`). Scripts: `model/fft2d_theory.py` (identidades y flujo de datos del RTL) y `model/fft1d_r22sdf_fxp.py` (aritmética bit a bit). Reemplaza a `theory_r22sdf.md` (2026-11-15), que era de cuando la imagen era de 512 × 256.

---

## 1. La DFT y la FFT radix-2

La DFT de $N$ puntos:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, W_N^{nk},
\qquad W_N = e^{-j 2\pi / N},
\qquad 0 \le k < N
$$

Calculada directo cuesta $N^2$ productos complejos. La FFT la parte en mitades. En la versión **DIF** (*decimation in frequency*), que es la que usamos, se separan las salidas pares e impares:

$$
\begin{aligned}
X[2k]   &= \mathrm{DFT}_{N/2}\Big\{\, x[n] + x[n + N/2] \,\Big\} \\
X[2k+1] &= \mathrm{DFT}_{N/2}\Big\{\, \big(x[n] - x[n + N/2]\big)\, W_N^{\,n} \,\Big\}
\end{aligned}
\qquad n = 0, \dots, N/2 - 1
$$

La operación de adentro, $a + b$ y $(a - b)\,W$, es la **mariposa** radix-2. Repitiendo la partición $\log_2 N$ veces quedan $\log_2 N$ **etapas** de $N/2$ mariposas cada una: $\tfrac{N}{2}\log_2 N$ productos en vez de $N^2$. Como cada etapa separa pares de impares, **la salida sale en orden bit-reversed** (el índice $k$ con sus bits dados vuelta).

## 2. Single-path delay feedback (SDF): una muestra por ciclo

Una arquitectura *pipelined* SDF hace cada etapa con **una mariposa y una línea de retardo** de $L = N/2, N/4, \dots, 1$ muestras, y procesa una muestra por ciclo sin parar:

- **Fase de carga** (las primeras $L$ muestras del bloque): la muestra que entra se guarda en la línea; por la salida sale lo que la línea tenía guardado del bloque anterior.
- **Fase de mariposa** (las siguientes $L$): llega $b = x[n + L]$ mientras de la línea sale $a = x[n]$. La mariposa calcula $a + b$, que sale ya, y $a - b$, que vuelve a la línea (*feedback*) para salir durante la próxima fase de carga.

La memoria total es $N/2 + N/4 + \dots + 1 = N - 1$ muestras, el mínimo posible para una FFT de streaming. El control es un contador por etapa: su bit alto dice en qué fase está.

## 3. Radix-2²: la mitad de los multiplicadores

He y Torkelson aplican **dos** pasos de la partición DIF juntos, con un mapeo de índices de tres niveles (paper, ec. 2):

$$
n = \tfrac{N}{2}\, n_1 + \tfrac{N}{4}\, n_2 + n_3,
\qquad
k = k_1 + 2 k_2 + 4 k_3,
\qquad
n_1, n_2, k_1, k_2 \in \{0, 1\},\;\; n_3, k_3 \in \big[0, \tfrac{N}{4}\big)
$$

El twiddle compuesto se descompone así (paper, ec. 4):

$$
W_N^{\left(\frac{N}{4} n_2 + n_3\right)\left(k_1 + 2k_2 + 4k_3\right)}
= (-j)^{\,n_2 (k_1 + 2k_2)} \cdot W_N^{\,n_3 (k_1 + 2k_2)} \cdot W_{N/4}^{\,n_3 k_3}
$$

La clave es el primer factor:

$$
W_N^{N/4} = e^{-j\pi/2} = -j
$$

Multiplicar por $-j$ **no necesita multiplicador**: es cruzar parte real e imaginaria y cambiar un signo,

$$
-j\,(a + jb) = b - ja
\qquad\Longrightarrow\qquad
(\mathrm{re},\, \mathrm{im}) \;\to\; (\mathrm{im},\, -\mathrm{re})
$$

Sustituyendo, la DFT queda como **4 DFT de $N/4$ puntos** (paper, ec. 5 y 6). Cada una toma la salida de una mariposa de dos niveles:

$$
H(k_1, k_2, n_3) =
\Big[\, x(n_3) + (-1)^{k_1}\, x\big(n_3 + \tfrac{N}{2}\big) \Big]
+ (-j)^{\,k_1 + 2k_2}\,
\Big[\, x\big(n_3 + \tfrac{N}{4}\big) + (-1)^{k_1}\, x\big(n_3 + \tfrac{3N}{4}\big) \Big]
$$

$$
X(k_1 + 2k_2 + 4k_3) = \mathrm{DFT}_{N/4}\Big\{\, H(k_1, k_2, n_3)\; W_N^{\,n_3 (k_1 + 2k_2)} \,\Big\}
$$

En palabras: **dos etapas radix-2 seguidas, con un $-j$ trivial entre ellas, y recién después un multiplicador "de verdad"**. Por eso el R2²SDF tiene un multiplicador cada dos etapas, la mitad que un R2SDF, con la misma mariposa simple y la misma memoria.

**Verificado (2026-11-15, NumPy):** $W_N^{N/4} = -j$ con error $6\cdot10^{-17}$; la ec. 4 con índices al azar, error $\le 3\cdot10^{-13}$; las 4 DFT de $N/4$ de la ec. 5–6 reproducen la DFT con error $\le 6\cdot10^{-14}$, para $N$ = 16, 64, 256 y 1024. Con el direccionamiento de twiddles del RTL, `fft2d_theory.py` da error de $10^{-14}$ para $N$ = 8 a 2048.

## 4. Cómo queda en nuestro RTL

```
x[n] ─► [unidad M = N] ─► [unidad M = N/4] ─► … ─► [unidad M = 4] ─► X[k] (bit-reversed)

unidad:  ─► mariposa con −j (línea M/2) ─► reg ─► mariposa (línea M/4) ─► reg ─► multiplicador W_M ─► reg ─►
```

- **Unidades:** $\lfloor \log_2 N / 2 \rfloor$. La última ($M = 4$) no multiplica: sus twiddles son todos triviales. Para $N = 1024$: 5 unidades, 10 etapas, **4 multiplicadores complejos (16 DSP)**.
- **Nombres al revés que el paper.** En el paper el $-j$ se aplica a la entrada de la segunda mariposa (BF2II). En la realización SDF, ese dato es justo el que sale de la línea de la **primera** mariposa durante su fase de carga, así que el conmutador vive en el módulo de la primera etapa. Por eso en el RTL la **primera** mariposa se llama `btfly_bf2ii` (la del $-j$) y la segunda `btfly_bf2i`. Es equivalente; está en las trampas.
- **Dirección del twiddle:** `tw_addr = tw_num · tw_sel`, con `tw_sel` $\in \{0, 1, 2, 3\}$ sacado de dos bits del contador (las 4 ramas: $W^0$, $W^n$, $W^{2n}$, $W^{3n}$) y `tw_num` el resto del contador corrido. Con `tw_sel = 0` ($W^0 = 1$) **se saltea el multiplicador**. Trampa: `tw_num` se trunca a $\log_2 N - 2$ bits; dejar los bits altos corre la dirección $N/2$, y como $W^{k + N/2} = -W^k$, invierte el signo del twiddle.
- **Todas las unidades leen la misma ROM** de $N$ twiddles ($W_N^k$, $k = 0, \dots, N - 1$), cada una con su paso.
- **Control por etapa:** cada etapa tiene su contador, que arranca cuando le llegan los datos (un contador global le daría a las etapas profundas un índice que no corresponde a la muestra que tienen).
- **$N = 2^{\text{impar}}$** (512): las unidades llegan hasta $M = 8$, todas con multiplicador, y se agrega una **etapa radix-2 final** de span 1 sin twiddle (`fft1d_r22sdf_tail`). $N = 512$ también usa 4 multiplicadores.
- **Escalado selectivo:** cada etapa puede dividir por 2 o no (`SCALE_MASK`, bit $i$ = etapa $i$ en orden de tiempo). La salida es la DFT por $2^{-S}$, con $S$ las etapas que dividen.
- **Redondeo:** al dividir se descarta un bit; con `ROUND_CONV = 1` (default desde el 2026-11-15) los empates van al par (§4.6).
- **Latencia** (primer `o_valid`), con $U = \lfloor \log_2 N / 2 \rfloor$:

$$
\text{latencia} =
\begin{cases}
N + (3 + \texttt{CMUL\_PIPE})\,(U - 1) & \log_2 N \text{ par} \quad (1040 \text{ para } N = 1024) \\[4pt]
N + (3 + \texttt{CMUL\_PIPE})\,U - 1 & \log_2 N \text{ impar} \quad (527 \text{ para } N = 512)
\end{cases}
$$

  El paper da $N - 1 + 3(\log_4 N - 1)$; la diferencia son nuestros registros de pipeline.

## 5. La IFFT con el mismo núcleo

Sea el *swap* que cruza parte real e imaginaria:

$$
\operatorname{swap}(a + jb) = b + ja = j\,\overline{(a + jb)}
$$

Entonces:

$$
\operatorname{swap}\Big(\mathrm{FFT}\big(\operatorname{swap}(X)\big)\Big)
= N \cdot \mathrm{IFFT}(X)
= \sum_{k} X[k]\, e^{+j 2\pi nk / N}
$$

Prueba: con $Y = \sum_k X[k]\, W_N^{-nk}$,

$$
\mathrm{FFT}\big(j\,\overline{X}\big)[n] = j\,\overline{Y}
\qquad\Longrightarrow\qquad
\operatorname{swap}\big(j\,\overline{Y}\big) = j\,\overline{\big(j\,\overline{Y}\big)} = j\,(-j)\,Y = Y
$$

En hardware es **cruzar cables** a la entrada y a la salida (`INVERSE = 1`): cero lógica, cero latencia, y el error en punto fijo es exactamente el de la FFT directa sobre la entrada cruzada (el swap no niega nada, así que no hay desborde). **Verificado:** error 0 en $N$ = 512 y 1024 (NumPy) y bit a bit en el RTL.

## 6. La FFT 2D

- **Separable:** la DFT 2D es la DFT 1D de todas las filas y después la de todas las columnas. Entre las dos pasadas hace falta un ***corner-turn*** (escribir por filas y leer por columnas).
- **Nuestra convención** (§3.8): pasada 1 sobre las 1024 muestras de readout de cada fila, pasada 2 sobre las 512 filas, las dos con `INVERSE = 1`. Cada pasada calcula $2^{-S}\sum$ (sin $1/N$), así que la salida es

$$
A = 2^{\,19 - S_1 - S_2} \cdot \mathrm{ifft2}(K),
\qquad 19 = \log_2(1024 \cdot 512)
$$

- **El orden bit-reversed no molesta:** la pasada 2 procesa columna por columna, así que el orden de las columnas da igual, y como el desplegado de SENSE es punto a punto, alcanza con guardar U en el mismo orden. **El datapath no necesita hardware de reordenamiento** (verificado, §3.8).
- **No hacen falta *shifts*:** con k-space centrado, la imagen sale multiplicada por $(-1)^{x + p}$, y la magnitud final es la misma.
- **Reparto de la escala entre pasadas:** lo que hay en cada frontera se conoce exacto, y si las etapas que dividen van **al principio** de cada pasada, el pico de cualquier etapa interna está acotado por el pico de la entrada o el de la salida de la pasada; alcanza con chequear las fronteras (§4.12, verificado en float y bit a bit).

## 7. Arquitecturas comparadas

### 7.1 La tabla del paper

Leída de la Tabla 1 de He y Torkelson:

| Arquitectura | Multiplicadores | Sumadores | Memoria | Control |
|---|---|---|---|---|
| R2MDC | $2(\log_4 N - 1)$ | $4 \log_4 N$ | $3N/2 - 2$ | simple |
| R2SDF | $\log_2 N - 1$ | $2 \log_2 N$ | $N - 1$ | simple |
| R4SDF | $\log_4 N - 1$ | $8 \log_4 N$ | $N - 1$ | medio |
| R4MDC | $3(\log_4 N - 1)$ | $8 \log_4 N$ | $5N/2 - 4$ | simple |
| R4SDC | $\log_4 N - 1$ | $3 \log_4 N$ | $2N - 2$ | complejo |
| **R2²SDF** | $\boldsymbol{\log_4 N - 1}$ | $4 \log_4 N$ | $\boldsymbol{N - 1}$ | **simple** |

El R2²SDF tiene el mínimo de multiplicadores y de memoria, con control simple; solo el R4SDC usa menos sumadores, a cambio de control complejo y el doble de memoria.

### 7.2 Las que implementamos

| | `fft1d_r2` (iterativa) | `fft1d_r2sdf` | **`fft1d_r22sdf` (elegida)** |
|---|---|---|---|
| Idea | una mariposa reusada, memoria *in-place* | streaming radix-2, una mariposa por etapa | streaming radix-2² |
| Ciclos por trama de $N = 1024$ | $N + \tfrac{N}{2}\log_2 N$ = **6144** (carga + 10 pasadas de 512) | **1024** (una muestra por ciclo) | **1024** |
| Multiplicadores complejos ($N = 1024$) | 1 | 8 ($\log_2 N - 2$: en nuestro RTL las dos últimas etapas son triviales) | **4** ($\log_4 N - 1$) |
| Memoria | $N$ muestras, con 2 lecturas + 2 escrituras por ciclo (no entra en una BRAM de dos puertos: termina en LUTRAM) | $N - 1$ | $N - 1$ |
| Control | FSM (carga / cálculo) y generador de direcciones | un contador global | un contador por etapa |
| Latencia | después de cargar la trama entera | $N - 1$ | $N + 16 = 1040$ (con registros) |

- La iterativa no llega: para 8 bobinas × 2 pasadas por trama necesitaría 6 veces los ciclos de una SDF.
- Entre las dos SDF, la memoria y el throughput son iguales: **el R2²SDF gana en multiplicadores (DSP), en ROM de twiddles y en puntos de redondeo** (cada multiplicador redondea; con la mitad, hay menos error). Los DSP no son el recurso crítico en la KV260 (16 de 1248), así que el argumento para la tesis es menos redondeo y menos ROM, no la velocidad.
- **Equivalencia:** el R2²SDF es radix-2 DIF por dentro con los twiddles reorganizados; da la misma salida que el R2SDF (en float, error 0), lo que sirvió como referencia al construirlo.

### 7.3 Multiplicador complejo: directo contra Karatsuba

$$
\begin{aligned}
\text{directo:}\quad   & (a + jb)(c + jd) = (ac - bd) + j\,(ad + bc)
  && \text{4 productos reales, 2 sumas} \\[2pt]
\text{Karatsuba:}\quad & \mathrm{Re} = ac - bd, \qquad \mathrm{Im} = (a + b)(c + d) - ac - bd
  && \text{3 productos reales, 5 sumas}
\end{aligned}
$$

- **Karatsuba ❌:** WNS −0,918 ns a 200 MHz. Ahorra un DSP pero agrega sumas antes y después del producto, en fábrica, sobre el camino crítico.
- **Directo ✅:** 4 DSP por multiplicador, con la resta y la suma **adentro** de los DSP48E2 (encadenados por `PCIN`). Fmax 211 MHz sin registro; con `CMUL_PIPE = 1` (registro antes del redondeo, absorbido como PREG del DSP), el núcleo cierra con +0,903 ns (§3.2, §5.6).
- El DSP48E2 multiplica 27 × 18 (UG579): con dato ≤ 27 bits y twiddle ≤ 18, cada producto real es un DSP. Por eso los twiddles son de 18 bits.

## Referencias

- S. He, M. Torkelson, *A New Approach to Pipeline FFT Processor*, Proc. IPPS, 1996: ecuaciones 1–6, figuras 3–5, tabla 1.
- M. Garrido, *A Survey on Pipelined FFT Hardware Architectures*, 2022: familias SDF, SDC, MDC y MDF (`books/papers/fft/`).
- Proakis y Manolakis, *Tratamiento digital de señales*: DFT, FFT radix-2, DIT y DIF (`books/`).
- AMD UG579 (DSP48E2): multiplicador 27 × 18, cascada `PCIN`, registros AREG/BREG/MREG/PREG.

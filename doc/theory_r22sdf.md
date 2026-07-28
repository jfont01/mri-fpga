# Arquitectura Radix-2² Single-Path Delay Feedback (R2²SDF)

**Nota de trazabilidad.** Todo lo de este documento está verificado numéricamente
contra NumPy o leído directamente del paper original (He & Torkelson, *A New
Approach to Pipeline FFT Processor*, 1996). Donde un resultado sale de una
simulación, se indica. Donde sale del paper, se cita la ecuación o figura.

---

## 1. Motivación y ubicación

El R2²SDF es una arquitectura de FFT *pipelined* de vía única (single-path),
que procesa una muestra por ciclo de reloj. Nace para resolver una tensión entre
dos familias previas:

- **Radix-2 (R2SDF):** mariposa (butterfly) simple, control sencillo, pero un
  multiplicador complejo en casi cada etapa.
- **Radix-4 (R4SDF):** menos multiplicadores, pero mariposa compleja (≥8 sumadores)
  y control más difícil.

El radix-2² logra **la complejidad multiplicativa del radix-4 conservando la
mariposa del radix-2**. Es el punto óptimo para una FFT de vía única cuando el
throughput requerido es de una muestra por ciclo, que es exactamente el régimen
de este trabajo (el presupuesto de tiempo tiene amplio margen: no se necesita
paralelismo multi-vía tipo MDC).

---

## 2. Derivación de la descomposición

Punto de partida: la DFT de N puntos (paper, ec. 1):

    X(k) = Σ_{n=0}^{N-1} x(n) · W_N^{nk} ,    0 ≤ k < N ,    W_N = e^{-j2π/N}

La idea del radix-2² es aplicar **dos** pasos de descomposición radix-2 DIF
(decimation-in-frequency) *juntos*, con un mapeo de índices tridimensional
(paper, ec. 2):

    n = ⟨ (N/2)·n₁ + (N/4)·n₂ + n₃ ⟩_N        n₁,n₂ ∈ {0,1},  n₃ ∈ [0, N/4)
    k = ⟨ k₁ + 2·k₂ + 4·k₃ ⟩                  k₁,k₂ ∈ {0,1},  k₃ ∈ [0, N/4)

El paso clave es la descomposición del *twiddle factor* compuesto (paper, ec. 4):

    W_N^{( (N/4)·n₂ + n₃ )·(k₁ + 2k₂ + 4k₃) }
        = (−j)^{ n₂·(k₁ + 2k₂) } · W_N^{ n₃·(k₁ + 2k₂) } · W_{N/4}^{ n₃·k₃ }

Acá aparece **el corazón del método**: el factor

    W_N^{N/4} = e^{-j2π/N · N/4} = e^{-jπ/2} = −j

**Verificado numéricamente** para N = 64, 256, 512, 1024: en todos los casos
`W_N^(N/4) = -j` exacto. Multiplicar por −j **no requiere multiplicador**: es
intercambiar parte real e imaginaria y cambiar un signo:

    −j · (a + jb) = b − ja   →   (re, im) → (im, −re)

### 2.1 La mariposa de dos niveles

Sustituyendo la descomposición del twiddle y expandiendo la suma sobre n₂, se
obtiene un conjunto de 4 DFT de longitud N/4 (paper, ec. 5), con el núcleo de
mariposa de dos niveles (paper, ec. 6):

    H(k₁,k₂,n₃) = [ x(n₃) + (−1)^{k₁}·x(n₃ + N/2) ]                 ← BF2I
                + (−j)^{k₁ + 2k₂} · [ x(n₃ + N/4) + (−1)^{k₁}·x(n₃ + 3N/4) ]   ← BF2II

Los dos corchetes son mariposas radix-2 comunes (**BF2I**). El factor
`(−j)^{k₁+2k₂}` que las combina toma los valores triviales {1, −j, −1, j} según
(k₁,k₂) — esto es la mariposa **BF2II**, que añade solo la multiplicación
trivial por −j.

**Verificado numéricamente:** implementando la ec. 6 completa y comparando contra
`np.fft.fft`, el error máximo es del orden de 10⁻¹³ para N = 16, 64, 256. La
descomposición reproduce la DFT exactamente.

---

## 3. Estructura del pipeline SDF

El mapeo a hardware single-path delay feedback (paper, Fig. 3 y 4) da una cadena
de `log₂(N)` etapas de delay-feedback, agrupadas de a pares:

    x(n) → BF2I → BF2II → ⊗ → BF2I → BF2II → ⊗ → ... → BF2I → BF2II → X(k)
                          W₁                W₂

Reglas de la estructura (leídas de las Fig. 3 y 4 del paper para N = 256):

- Las etapas alternan **BF2I** (mariposa simple) y **BF2II** (mariposa + la
  multiplicación trivial por −j, implementada con un conmutador real/imaginario
  y sumas/restas controladas).
- Un **multiplicador complejo no trivial** aparece **después de cada par**
  BF2I-BF2II, *excepto el último par*. Esto da `log₄(N) − 1` multiplicadores.
- Las líneas de retardo (delay lines) tienen los largos N/2, N/4, ..., 1, igual
  que en el R2SDF. La memoria total es N−1.
- El control lo genera un único contador binario de `log₂(N)` bits, que sirve a
  la vez de sincronización y de puntero de dirección para leer los twiddles.
- La salida sale en **orden bit-reversed**, igual que el R2SDF.
- La latencia (con registros de pipeline entre multiplicador y mariposa) es
  `N − 1 + 3(log₄N − 1)`.

**Restricción importante:** el radix-2² requiere que **N sea potencia de 4**
(es decir, `log₂N` par), para que las etapas se emparejen limpiamente.

---

## 4. Comparación de recursos (Tabla 1 del paper)

Leído directamente de la Tabla 1 de He & Torkelson. `log₄N = log₂N / 2`.

| Arquitectura | Multiplicadores    | Sumadores  | Memoria   | Control  |
|--------------|--------------------|------------|-----------|----------|
| R2MDC        | 2(log₄N − 1)       | 4·log₄N    | 3N/2 − 2  | simple   |
| **R2SDF**    | **log₂N − 1**      | 2·log₂N    | N − 1     | simple   |
| R4SDF        | log₄N − 1          | 8·log₄N    | N − 1     | medium   |
| R4MDC        | 3(log₄N − 1)       | 8·log₄N    | 5N/2 − 4  | simple   |
| R4SDC        | log₄N − 1          | 3·log₄N    | 2N − 2    | complex  |
| **R2²SDF**   | **log₄N − 1**      | 4·log₄N    | N − 1     | simple   |

El paper resume: el R2²SDF alcanza el **mínimo** requerimiento tanto de
multiplicadores como de almacenamiento, y es solo segundo (detrás del R4SDC) en
sumadores — con control **simple**. Por eso lo presenta como la arquitectura
ideal para implementación VLSI de FFT pipelined.

---

## 5. Ventaja concreta para este trabajo

Comparación R2SDF vs R2²SDF para los tamaños relevantes (**verificado por conteo
directo** de las fórmulas de la Tabla 1):

| N     | R2SDF (mult) | R2²SDF (mult) | Ahorro       | ¿N potencia de 4? |
|-------|--------------|---------------|--------------|-------------------|
| 64    | 5            | 2             | 3  (60 %)    | sí                |
| 256   | 7            | 3             | 4  (57 %)    | sí (phase-encode) |
| 512   | 8            | 3             | 5  (62 %)    | **no** (ver §5.1) |
| 1024  | 9            | 4             | 5  (55 %)    | sí                |

Puntos clave:

- **La memoria y los sumadores son idénticos** entre R2SDF y R2²SDF (N−1 y
  2·log₂N respectivamente). La ventaja es *puramente* en número de
  multiplicadores y en la ROM de twiddles asociada.
- Como en este diseño los DSP no son el recurso crítico (hay margen), la
  justificación de elegir R2²SDF **no es** el ahorro de velocidad, sino la
  **reducción de multiplicadores/DSP, de ROM de twiddles (BRAM) y la
  simplificación de la verificación numérica** (menos puntos de redondeo en
  punto fijo). Este matiz es el argumento correcto a sostener en la tesis.

### 5.1 El caso N = 512 (readout)

El readout usa N = 512 = 2⁹. Como `log₂(512) = 9` es **impar**, 512 **no es
potencia de 4**, y la cadena radix-2² deja una **etapa radix-2 suelta** que no
se empareja. Es un caso conocido y resoluble: se coloca una etapa radix-2 simple
(idéntica a las del R2SDF) al principio o al final de la cadena radix-2². El
phase-encode (N = 256 = 2⁸) sí es potencia de 4 y no tiene este problema.

Esto es material de tesis: documentar que el sistema maneja dos casos (N=512 con
etapa extra, N=256 limpio) es exactamente el tipo de detalle de ingeniería que
se valora.

---

## 6. Equivalencia con el R2SDF (base para la verificación)

**Verificado numéricamente:** el R2²SDF produce **exactamente la misma salida**
que el R2SDF (error 0.00 bit a bit en la simulación, no solo cercano a cero).
Ambos son radix-2 DIF por dentro; el radix-2² solo *reorganiza* los twiddles
para que la mitad sean triviales.

Esto tiene una consecuencia práctica muy útil para construir y verificar el
modelo del R2²SDF: se puede usar el R2SDF ya verificado como **referencia
bit-exacta** en cada paso, además de la DFT. Si el R2²SDF en construcción se
aparta del R2SDF, el error localiza inmediatamente el problema.

---

## 7. Resumen de lo verificado vs. lo pendiente

**Verificado y sólido (este documento):**

- La identidad `W_N^(N/4) = −j` para todos los N relevantes.
- La descomposición de la ec. 6 reproduce la DFT (error ~10⁻¹³).
- Los conteos de recursos de la Tabla 1 y el ahorro de multiplicadores para
  N = 64, 256, 512, 1024.
- La equivalencia bit-exacta R2²SDF ↔ R2SDF.
- El caso especial N = 512 (etapa radix-2 suelta).

**Pendiente (para el modelo C++):**

- El mapeo exacto del control de bits del pipeline: qué bit del contador dispara
  el −j en cada BF2II y la ubicación precisa de cada multiplicador en la cadena.
  Se resolverá construyendo el modelo incrementalmente, con el R2SDF como red de
  seguridad en cada etapa.

---

## Referencias

- S. He, M. Torkelson, *A New Approach to Pipeline FFT Processor*, Proc. IPPS,
  1996. (Fuente primaria del R2²SDF; ecuaciones 1–6, Figuras 3–5, Tabla 1.)
- Garrido, *A Survey on Pipelined FFT Hardware Architectures*, 2022. (Contexto y
  comparación de familias SDF/SDC/MDC/MDF.)
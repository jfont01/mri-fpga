# Teoría del CORDIC de magnitud

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Estudio de parámetros y números: [04_precision.md](04_precision.md) §4.11. Modelo bit a bit: `model/mac_cordic_fxp.py`. Las ecuaciones están en LaTeX: se ven con la vista previa de Markdown (`Ctrl+Shift+V`).*

**Trazabilidad.** El algoritmo es el de J. E. Volder (*The CORDIC Trigonometric Computing Technique*, 1959), en modo *vectoring*. La ganancia, la cota de error y el comportamiento están verificados en NumPy (se indica dónde).

---

## 1. Para qué está

Después del desplegado, cada píxel de la imagen es un complejo $\hat{m} = \mathrm{re} + j\,\mathrm{im}$. Lo que se muestra es la **magnitud**:

$$
|\hat{m}| = \sqrt{\mathrm{re}^2 + \mathrm{im}^2}
$$

Un píxel por ciclo (8,39 M por trama a 200 MHz) con multiplicadores y raíz cuadrada es caro. El CORDIC la calcula **solo con sumas y corrimientos**, y la latencia es un ciclo por iteración en un pipeline desenrollado.

## 2. La idea: girar el vector hasta el eje x

Girar el vector $(x, y)$ un ángulo $\theta$ conserva su largo. Si se lo gira hasta que quede sobre el eje $x$ ($y = 0$), la coordenada $x$ **es** la magnitud. El problema es que una rotación general necesita multiplicar por $\cos\theta$ y $\sin\theta$.

El truco de Volder es usar solo los ángulos

$$
\theta_i = \arctan 2^{-i}, \qquad \tan\theta_i = 2^{-i}
$$

porque entonces la rotación (sin el factor $\cos\theta_i$) es un **corrimiento y una suma**:

$$
\begin{aligned}
x_{i+1} &= x_i + \sigma_i\, y_i\, 2^{-i} \\
y_{i+1} &= y_i - \sigma_i\, x_i\, 2^{-i}
\end{aligned}
\qquad
\sigma_i =
\begin{cases}
+1 & \text{si } y_i \ge 0 \;\; \text{(girar hacia abajo)} \\
-1 & \text{si } y_i < 0
\end{cases}
$$

En cada paso se gira $\pm\theta_i$ para acercar $y$ a cero (eso es el modo *vectoring*). Los ángulos van bajando: 45°, 26,57°, 14,04°, 7,13°, … y sumados alcanzan para llevar cualquier vector del primer cuadrante al eje.

**Primer cuadrante:** nuestro modelo arranca con $(|\mathrm{re}|, |\mathrm{im}|)$, que tiene la misma magnitud y siempre queda entre 0° y 90°.

## 3. La ganancia $K_n$

Cada micro-rotación sin el factor $\cos\theta_i$ estira el vector en $1/\cos\theta_i = \sqrt{1 + 2^{-2i}}$. Después de $n$ pasos:

$$
x_n \approx K_n\, |\hat{m}|,
\qquad
K_n = \prod_{i=0}^{n-1} \sqrt{1 + 2^{-2i}}
\;\xrightarrow{\;n\,=\,12\;}\; 1{,}646760
$$

$K_n$ es una constante (no depende del vector). **En nuestro diseño no se corrige en la PL:** el PS ya multiplica $U''$ por $1/K_n$ (§4.11, [teoria_sense.md §5.2](teoria_sense.md)), así que la salida del CORDIC ya está en escala. **Verificado (NumPy):** $K_8 = 1{,}646744$, $K_{12} = 1{,}646760$, $K_{16} = 1{,}646760$.

## 4. Cuánto se equivoca

Hay dos errores:

- **Residuo angular** (el algoritmo en sí): después de $n$ pasos el vector queda a lo sumo a $\arctan 2^{-(n-1)}$ del eje, así que

$$
\frac{\Delta|\hat{m}|}{|\hat{m}|} \;\le\; 1 - \cos\!\left(\arctan 2^{-(n-1)}\right) \;\approx\; 2^{-(2n-1)}
$$

  Con $n = 12$ es $\sim 10^{-7}$: 0,008 LSB a fondo de escala. **Verificado (NumPy, en float):** el error relativo máximo sobre 1000 vectores al azar coincide con la cota ($3{,}0\cdot10^{-5}$ con $n = 8$; $1{,}2\cdot10^{-7}$ con $n = 12$; $4{,}7\cdot10^{-10}$ con $n = 16$).
- **Truncamiento** (la implementación en enteros): cada corrimiento `>> i` tira bits, y esos errores se acumulan con $n$. Se compensa con **$G$ bits de guarda**: el CORDIC trabaja con $G$ bits fraccionales extra y recién al final redondea a 16 bits.

Por eso más iteraciones no siempre es mejor: con $n = 16$ y sin bits de guarda el error llega a 15 LSB (tabla de §4.11).

## 5. Lo que elegimos

**$n = 12$ iteraciones y $G = 5$ bits de guarda** (§4.11, con el fantoma):

| $n$ | $G$ | Error máx (LSB) | rms | SNR de la imagen |
|---|---|---|---|---|
| 8 | 0 | 7,17 | 2,38 | 51,47 dB |
| 8 | 4 | 1,37 | 0,34 | 51,97 dB |
| **12** | **5** | **0,81** | **0,31** | **51,96 dB** |
| 16 | 0 | 15,17 | 7,35 | 50,34 dB |

- Error menor que 1 LSB, y el residuo angular es despreciable aun a fondo de escala.
- Referencia: el redondeo ideal a 16 bits tiene rms 0,289 LSB; la magnitud exacta da 51,99 dB.
- La cadena completa es: suma del MAC $P$ (exacta) → corrimiento por $D$ bits con redondeo y saturación a $17 + G$ bits → CORDIC → redondeo a u16.

**La magnitud sube la SNR** de 51,03 dB ($\hat{m}$ complejo) a 51,99 dB: descarta la parte del error que está en cuadratura con un píxel de valor grande. En el fondo ($m = 0$) queda el módulo del ruido (distribución de Rayleigh), de unos 30 LSB, que los 16 bits de salida resuelven.

## 6. En hardware

- **Pipeline desenrollado:** $n = 12$ etapas, cada una con 2 sumadores/restadores de ~23 bits ($17 + G$ más el crecimiento) y corrimientos fijos (cableado). Unos **24 sumadores**, **ningún DSP**, latencia de 12 ciclos, un píxel por ciclo.
- El signo de $y$ decide si cada etapa suma o resta (un mux por bit, o el sumador/restador con el signo como control).
- Alternativas descartadas o no estudiadas: multiplicadores + raíz (2 DSP y una raíz iterativa), aproximaciones tipo *alpha max plus beta min* (más baratas pero con error de algunos %).

## Referencias

- J. E. Volder, *The CORDIC Trigonometric Computing Technique*, IRE Trans. Electronic Computers, 1959.
- B. Parhami, *Computer Arithmetic: Algorithms and Hardware Designs*, cap. 22, *The CORDIC Algorithms* (`books/`). Cita a R. Andraka, *A survey of CORDIC algorithms for FPGA based computers*, 1998: conviene conseguirlo.
- Xilinx DSP Workbook, *The CORDIC Algorithm* (`books/Xilinix DSP Workbook training material/notes/TheCORDICAlgorithmXilinx.pdf`). Está en la PC con Ubuntu, **sin versionar** (`*.pdf` está en `.gitignore`; de esa carpeta git solo tiene `Xilinx_DSP_Primer/`: modelos `.mdl` y sus datos, entre ellos `CORDIC_Mag/`). En la PC con WSL2 falta: copiarlo (2026-11-16).

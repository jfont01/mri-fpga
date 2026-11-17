# Teoría de SENSE y del cálculo de U

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Números y verificaciones: [04_precision.md](04_precision.md) §4.9–4.11; glosario de símbolos: [01_contexto.md](01_contexto.md). Las ecuaciones están en LaTeX: se ven con la vista previa de Markdown (`Ctrl+Shift+V`).*

**Trazabilidad.** Formulación de K. P. Pruessmann et al., *SENSE: Sensitivity Encoding for Fast MRI*, 1999 (⏳ no está en `books/`: en `books/papers/mri/` hay otro paper, *Cardiac Real-Time Imaging Using SENSE*), escrita con **nuestros** símbolos (glosario). Cuando haga falta la notación exacta del paper, se toma de una captura del paper, no de memoria. Las identidades están verificadas en NumPy (se indica dónde).

---

## 1. De dónde sale la imagen

En MRI cartesiano, lo que mide cada bobina no es la imagen sino su **transformada de Fourier 2D**, el *k-space*. Una fila del k-space se mide de una vez (dirección de *readout*, 1024 muestras); para medir otra fila hay que repetir la excitación con otro gradiente de fase (*phase-encode*). Por eso **el tiempo de adquisición es proporcional a la cantidad de filas de phase-encode**.

La bobina $l$ no ve la imagen $m$ tal cual, sino pesada por su **mapa de sensibilidad** $s_l$ (cuánto "ve" esa bobina de cada píxel; es suave, y alto cerca de la bobina):

$$
K_l = \mathcal{F}_{2D}\{\, s_l \cdot m \,\} + \text{ruido},
\qquad l = 0, \dots, L - 1, \quad L = 8
$$

y con todas las filas se recupera $s_l \cdot m = \mathcal{F}_{2D}^{-1}\{K_l\}$.

## 2. Acelerar: medir menos filas y aparece el plegado

Para ir más rápido se mide **una de cada $A_f$ filas** de phase-encode ($A_f = 2$: la mitad del tiempo). Medir en el k-space con paso $A_f$ equivale a reducir el campo de visión (FOV) por $A_f$ en esa dirección, y la IFFT de lo medido da una imagen **plegada**: cada píxel es la suma de $A_f$ píxeles verdaderos separados $N / A_f$ filas.

$$
y_l(x, p) = \sum_{a=0}^{A_f - 1} s_l\!\left(x,\; p + a\,\tfrac{N}{A_f}\right)\, m\!\left(x,\; p + a\,\tfrac{N}{A_f}\right),
\qquad p = 0, \dots, \tfrac{N}{A_f} - 1
$$

Con $A_f = 2$ y $N = 1024$: el píxel plegado $(x, p)$ mezcla las filas $p$ y $p + 512$. La imagen plegada de cada bobina es de 512 × 1024; esto es lo que calcula la IFFT 2D del FPGA.

**Verificado (NumPy):** la IFFT 2D de un k-space al que se le saca una fila de cada dos da exactamente la suma de las dos mitades de la imagen, sin factor $1/A_f$ con la normalización de `ifft2` sobre el tamaño submuestreado (§3.8, check A).

## 3. SENSE: separar lo superpuesto con las bobinas

En un mismo píxel plegado, cada bobina ve **la misma mezcla con distintos pesos**, porque cada una tiene su mapa. Juntando las $L$ bobinas, para cada $(x, p)$:

$$
\mathbf{y} = S\,\mathbf{m},
\qquad
\mathbf{y} \in \mathbb{C}^{L},\;\;
\mathbf{m} \in \mathbb{C}^{A_f},\;\;
S \in \mathbb{C}^{L \times A_f},
\qquad
S[l, a] = s_l\!\left(x,\; p + a\,\tfrac{N}{A_f}\right)
$$

- $\mathbf{y}$: las 8 imágenes plegadas en ese píxel.
- $\mathbf{m}$: los 2 píxeles verdaderos que se superponen ahí.

Son 8 ecuaciones con 2 incógnitas: sobra información, y se resuelve por **mínimos cuadrados**:

$$
\hat{\mathbf{m}} = \left(S^{\mathsf{H}} S\right)^{-1} S^{\mathsf{H}}\, \mathbf{y} = U\,\mathbf{y},
\qquad U \in \mathbb{C}^{A_f \times L} \quad \text{(la "matriz de desplegado")}
$$

$U S = I$: si no hubiera ruido, $\hat{\mathbf{m}} = \mathbf{m}$ exacto. **Verificado:** $\lVert U S - I \rVert \approx 4\cdot10^{-16}$ y $\lVert U\mathbf{y} - \mathbf{m} \rVert \approx 5\cdot10^{-16}$ con $S$ al azar (NumPy).

Pruessmann incluye además la covarianza del ruido entre bobinas, $\Psi$:

$$
U = \left(S^{\mathsf{H}} \Psi^{-1} S\right)^{-1} S^{\mathsf{H}} \Psi^{-1}
$$

Nosotros usamos $\Psi = I$ (bobinas con ruido independiente e igual), que es el caso más simple; ⏳ revisarlo con datos reales.

**El costo de acelerar:** cuanto más parecidas son las filas de $S$ (bobinas que ven parecido a los dos píxeles superpuestos), peor condicionada está $S^{\mathsf{H}}S$ y más se amplifica el ruido. Pruessmann lo mide con el **factor g** (*g-factor*) por píxel. La SNR de la imagen SENSE, comparada con la de una adquisición completa, pierde dos cosas: $\sqrt{A_f}$ porque se midieron $A_f$ veces menos filas, y $g$ por la geometría de las bobinas:

$$
\mathrm{SNR}_{\text{SENSE}} = \frac{\mathrm{SNR}_{\text{full}}}{g\,\sqrt{A_f}},
\qquad
g_\rho = \sqrt{\left[\left(S^{\mathsf{H}} \Psi^{-1} S\right)^{-1}\right]_{\rho\rho}\,\left[S^{\mathsf{H}} \Psi^{-1} S\right]_{\rho\rho}} \;\ge\; 1
$$

donde $\rho$ es cada uno de los $A_f$ píxeles verdaderos de la posición plegada. Fuentes: Brown, Cheng y Haacke, ec. 28.46–28.47 (usan $B$ por $S$ y $R$ por $A_f$), y Weiger, Pruessmann y Boesiger 2000 (`books/papers/mri/`).

- Con $\Psi = I$, $g$ sale de lo que **ya calcula el PS**: $A = S^{\mathsf{H}}S$ y su inversa (la diagonal de $A^{-1}$ es la varianza del ruido de $\hat{\mathbf{m}}$, porque $U U^{\mathsf{H}} = A^{-1}$). **Verificado (NumPy, 2026-11-16):** $g \ge 1$ y $\operatorname{diag}(U U^{\mathsf{H}}) = \operatorname{diag}\big((S^{\mathsf{H}}S)^{-1}\big)$ en 2000 casos al azar ($L = 8$, $A_f = 2$).
- ⏳ Con $\lambda > 0$ la covarianza de $\hat{\mathbf{m}}$ pasa a ser $U_\lambda U_\lambda^{\mathsf{H}}$, que ya no es $(S^{\mathsf{H}}S + \lambda I)^{-1}$: hay que verificar en NumPy qué $g$ usar antes de usarlo para elegir $\lambda$.

## 4. Regularización (Tikhonov)

Donde los mapas se anulan (afuera del objeto, o mapas tipo ESPIRiT enmascarados), $S^{\mathsf{H}}S$ es singular y no se puede invertir. Se regulariza:

$$
U_\lambda = \left(S^{\mathsf{H}} S + \lambda I\right)^{-1} S^{\mathsf{H}},
\qquad\text{y entonces}\qquad
U_\lambda S = I - \lambda \left(S^{\mathsf{H}} S + \lambda I\right)^{-1}
$$

El sesgo relativo es del orden de $\lambda / d_{\min}$ (el pivote más chico). **Verificado** a $10^{-15}$ (§4.10). Con mapas enmascarados y sin regularizar, el cálculo falla en el 83 % de las posiciones; con

$$
\lambda = 10^{-4} \cdot \operatorname{mediana}\!\left(\operatorname{diag} S^{\mathsf{H}}S\right)
$$

no hay divisiones por cero en ningún caso y la imagen pierde 0,02 dB respecto de $\lambda = 10^{-6}$ (§4.10). Con datos reales $\lambda$ también va a regular el ruido amplificado (factor g): ⏳ elegirlo con fastMRI.

## 5. Por qué U se calcula en el PS y una sola vez

**Los mapas no cambian entre tramas** (la bobina y el paciente están quietos), así que $U$ tampoco. Toda la parte cara y delicada (formar $S^{\mathsf{H}}S$, invertir, dividir) se hace **una vez por sesión en el PS** (el ARM A53, en punto flotante), y en cada trama la PL solo hace la IFFT 2D y un **producto matriz-vector** de 2 × 8 por píxel plegado. No hay divisiones ni raíces en tiempo real.

### 5.1 El algoritmo en el PS: LDLᴴ sin raíces

Por cada posición plegada $(x, p)$, con $S$ de 8 × 2:

1. Se arma la matriz de 2 × 2, hermitiana y definida positiva:

$$
A = S^{\mathsf{H}} S + \lambda I
$$

2. Se factoriza $A = \mathcal{L}\, D\, \mathcal{L}^{\mathsf{H}}$ ($\mathcal{L}$ triangular inferior con unos en la diagonal, $D$ diagonal), columna por columna:

$$
d_j = A_{jj} - \sum_{k<j} \left|\mathcal{L}_{jk}\right|^2 d_k,
\qquad
\mathcal{L}_{ij} = \frac{A_{ij} - \sum_{k<j} \mathcal{L}_{ik}\, d_k\, \overline{\mathcal{L}_{jk}}}{d_j}
\quad (i > j)
$$

   $A$ es definida positiva **si y solo si todos los $d_j > 0$**: es el chequeo de validez de cada posición.

3. $U = A^{-1} S^{\mathsf{H}}$, resolviendo para cada bobina $l$:

$$
\mathcal{L}\,\mathbf{z} = \overline{S[l, :]}^{\,\mathsf{T}},
\qquad
\mathbf{w} = D^{-1} \mathbf{z},
\qquad
\mathcal{L}^{\mathsf{H}}\, \mathbf{u}_l = \mathbf{w}
$$

Por qué $\mathcal{L} D \mathcal{L}^{\mathsf{H}}$ y no Cholesky: **no necesita raíces cuadradas**, y para $A_f = 2$ cuesta 328 flops, 2 divisiones y 32 redondeos por posición (**172 Mflop por sesión**; ⏳ medir en el A53, se estima menos de un segundo). Sirve igual para $A_f$ = 3 y 4. (En la bitácora el factor triangular se escribe $L$; acá $\mathcal{L}$ para no confundirlo con la cantidad de bobinas.)

**Verificado (§4.10):** $\mathcal{L} D \mathcal{L}^{\mathsf{H}} = A$ y $U$ = `np.linalg.solve` con error relativo $10^{-15}$ a $10^{-16}$, para $A_f$ = 2, 3 y 4, con y sin $\lambda$. Referencia del código del PS: `model/unfold_ref.py` (re e im separados y en el orden de operaciones del futuro C, para poder exigirle al C bit a bit).

### 5.2 De U a lo que lee la PL (U'')

La PL no recibe $U$ tal cual. El PS le mete adentro tres constantes, para que la PL solo multiplique y sume:

$$
U' = \frac{U}{g_l},
\qquad
U'' = U' \cdot \frac{2^{-E}}{K_c}
$$

- $g_l$: la ganancia con la que se normalizó el k-space de la bobina $l$ (su pico a 0,9 del fondo de escala de 16 bits).
- $E$: un exponente común por sesión, para que $U''$ entre en 16 bits.
- $K_c$: la ganancia del CORDIC ([teoria_cordic.md](teoria_cordic.md)).

En cada píxel plegado y para cada uno de los 2 píxeles verdaderos, la imagen es:

$$
\hat{m}\big(x,\; p + 512\,a\big) = 2^{\,S_1 + S_2 - 19} \sum_{l} U'(x, p, a, l)\; A_l(x, p)
$$

donde $A_l$ es la salida de la IFFT 2D (en Q1.15) y la potencia de 2 deshace el escalado de la FFT (§4.9). Lo que **calcula la PL** es la misma suma, pero con $U''$, que trae adentro $2^{-E}$ y $1/K_c$:

$$
\frac{\hat{m}\big(x,\; p + 512\,a\big)}{K_c} = 2^{\,S_1 + S_2 - 19 + E} \sum_{l} U''(x, p, a, l)\; A_l(x, p)
$$

El $1/K_c$ no se corrige: el CORDIC calcula la magnitud multiplicando por su ganancia $K_c$, así que a la salida queda $|\hat{m}|$ en la escala correcta ($\approx |\hat{m}| \cdot 2^{F_{\text{OUT}}}$, con el corrimiento $D = 49 - S - E - F_{\text{OUT}}$ de §4.11). **Verificado (NumPy, `mac_cordic_fxp.py`, 2026-11-16):** el lado derecho difiere de $\hat{m}$ en un 39 % ($1 - 1/K_{12}$) y de $\hat{m}/K_{12}$ en $3\cdot10^{-5}$ (la cuantización de $U''$; $5\cdot10^{-5}$ con otra semilla en la PC con WSL2); a la salida del CORDIC, el error contra $|\hat{m}|\cdot 2^{F_{\text{OUT}}}$ tiene media $+0{,}01$ LSB (la escala es la correcta). *(Corrección del 2026-11-16: antes este párrafo decía que la PL calcula $\hat{m}$, sin el $1/K_c$.)* $U''$ son 2 × 8 coeficientes complejos de 16 + 16 bits por posición: **32 MiB** en DDR, guardados en el mismo orden bit-reversed en que sale la IFFT 2D (así el MAC no reordena nada).

## 6. Los mapas de sensibilidad (pendiente)

Hasta ahora usamos mapas **sintéticos** (`model/gen/gen_smaps.py`). Con datos reales hay que estimarlos, una vez por sesión, en el PS:

- Se necesita el **centro del k-space completo** (líneas ACS, que fastMRI trae) para tener imágenes de baja resolución $I_l$ de cada bobina sin plegado.
- **RSS** (*root sum of squares*):

$$
s_l = \frac{I_l}{\sqrt{\sum_{l'} \left|I_{l'}\right|^2}}
$$

  Simple, pero la división se va al infinito afuera del objeto (hay que enmascarar o regularizar) y el cociente crudo es ruidoso (hay que suavizar).
- **ESPIRiT:** método por autovectores de una matriz armada con el k-space de calibración; da mapas más limpios y enmascarados.
- ⏳ Elegir el método y medir su costo en el A53 (estimación inicial: ~79 Mflop, decenas de ms).

## 7. Nuestros números

| | Valor |
|---|---|
| Imagen final | 1024 × 1024 |
| Bobinas, aceleración | $L = 8$, $A_f = 2$ (en phase-encode) |
| k-space por bobina | 512 filas de phase-encode × 1024 de readout |
| Imagen plegada por bobina | 512 × 1024 |
| Posiciones plegadas | 524 288 (cada una con un $U$ de 2 × 8) |
| MAC por trama | 1024 × 512 × 16 = 8,39 M productos complejos |
| $U''$ | 32 MiB por sesión, leído entero en cada trama (32 MiB × 20 fps = 640 MiB/s = 671 MB/s) |
| $\lambda$ | $10^{-4}$ × mediana de $\operatorname{diag}(S^{\mathsf{H}}S)$ (propuesta, a confirmar con fastMRI) |

## Referencias

- K. P. Pruessmann, M. Weiger, M. B. Scheidegger, P. Boesiger, *SENSE: Sensitivity Encoding for Fast MRI*, MRM 1999 (⏳ no está en `books/`).
- M. Weiger, K. P. Pruessmann, P. Boesiger, *Cardiac Real-Time Imaging Using SENSE*, MRM 43:177–184, 2000 (`books/papers/mri/`): la fórmula de SNR con $g$ y $\sqrt{R}$.
- R. W. Brown, Y.-C. N. Cheng, E. M. Haacke et al., *Magnetic Resonance Imaging: Physical Principles and Sequence Design*, 2.ª ed., cap. 28, §28.4, ec. 28.42–28.47 (`books/`).
- Akçakaya et al., *Magnetic Resonance Image Reconstruction: Theory, Methods and Applications*; *Parallel Imaging in Clinical MR Applications*; Paul y Mathew, *Regularized Image Reconstruction in Parallel MRI with MATLAB*; Liang y Lauterbur, *Principles of MRI: A Signal Processing Perspective* (`books/`).
- Watkins, *Fundamentals of Matrix Computations*; Golub y Van Loan, *Matrix Computations*: factorización LDLᴴ (`books/`).

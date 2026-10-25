# SENSE en el FPGA: ecuaciones en limpio

Oct 24, 2026 · @jfont

Cada ecuación de la reconstrucción, en el orden en que viaja una trama, con lo que significa cada símbolo y los números de nuestro sistema.

## El camino de una trama

&#91;embedded content: camino de una trama · 5 pasos en el FPGA, U″ desde el PS\]

Los pasos 2 a 5 corren en el FPGA en cada trama, con las 8 bobinas en serie; U″ lo calcula el PS una sola vez por sesión y el FPGA solo lo lee de la DDR.

## 1. Entrada al FPGA: el k-space a 16 bits

Cada bobina mide su k-space; antes de entrar al FPGA se lo multiplica por una ganancia y se lo redondea a 16 bits.

```latex
K_{q,l} \;=\; \text{redondeo a 16 bits}\Big(\, g_l \cdot K_l \,\Big)
\qquad\qquad
g_l \;=\; \frac{0{,}9}{\text{pico de } |K_l|}
```

- **K\_l**: el k-space que mide la bobina l (l va de 1 a 8). Es una tabla de 512 filas × 1024 columnas de números complejos.
- **g\_l**: la ganancia de la bobina l. Lleva el valor más grande de K\_l a 0,9 del máximo que entra en 16 bits, para aprovechar el rango sin saturar. Se elige una vez por sesión.
- **K\_q,l**: el resultado, lo que realmente entra al FPGA. Está en Q1.15: números entre −1 y 1 con 15 bits de fracción. La "q" es de cuantizado.

## 2. IFFT 2D: la imagen plegada de cada bobina

El FPGA le hace la IFFT 2D al k-space de cada bobina. Como se midió una línea de cada dos, sale una imagen de la mitad de alto, con dos partes del cuerpo superpuestas.

Lo que calcula el hardware:

```latex
A_l(x,\,p) \;=\; \frac{1}{2^{\,S_1+S_2}} \sum_{k'=0}^{511} \sum_{r=0}^{1023} K_{q,l}(k',\,r)\; e^{\,j 2\pi \left( \frac{r\,x}{1024} \,+\, \frac{k'\,p}{512} \right)}
```

Qué significa ese resultado (el plegado): cada píxel mezcla dos píxeles verdaderos separados por media imagen, cada uno pesado por cuánto lo ve esa bobina.

```latex
y_l(x,\,p) \;=\; s_l(x,\,p)\; m(x,\,p) \;+\; s_l(x,\,p+512)\; m(x,\,p+512)
\qquad\text{con}\qquad
y_l \;=\; A_l \cdot \frac{2^{\,S_1+S_2-19}}{g_l}
```

- **A\_l**: la salida de la IFFT 2D para la bobina l, 512 × 1024, en 16 bits. Es la imagen plegada tal como sale del FPGA.
- **x, p**: x es la columna (0 a 1023); p es la fila de la imagen plegada (0 a 511).
- **S1, S2**: cuántas etapas de la primera y de la segunda pasada dividen por 2, para que los números no crezcan y saturen. Con las máscaras 801 y 384, S1 = 4 y S2 = 2.
- **y\_l**: la misma imagen plegada en unidades físicas, deshaciendo g\_l y las divisiones por 2. El 19 sale de 1024 × 512 = 2^19: la IFFT de libro divide por la cantidad de muestras y el hardware no.
- **m**: la imagen verdadera, la que queremos recuperar (1024 × 1024).
- **s\_l**: el mapa de sensibilidad de la bobina l: cuánto ve esa bobina de cada píxel.

## 3. Desplegado SENSE: separar los dos píxeles

Para cada píxel plegado hay 8 ecuaciones, una por bobina, y solo 2 incógnitas: los dos píxeles superpuestos. La matriz U resuelve ese sistema.

Las 8 bobinas en un mismo píxel plegado, escritas como matrices:

```latex
\underbrace{\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_8 \end{bmatrix}}_{\mathbf{y}\ (8 \times 1)}
\;=\;
\underbrace{\begin{bmatrix} s_1(x,p) & s_1(x,p{+}512) \\ s_2(x,p) & s_2(x,p{+}512) \\ \vdots & \vdots \\ s_8(x,p) & s_8(x,p{+}512) \end{bmatrix}}_{S\ (8 \times 2)}
\underbrace{\begin{bmatrix} m(x,p) \\ m(x,p{+}512) \end{bmatrix}}_{\mathbf{m}\ (2 \times 1)}
```

La solución:

```latex
\hat{\mathbf{m}} \;=\; U\,\mathbf{y}
\qquad\qquad
U \;=\; \left( S^{H} S + \lambda I \right)^{-1} S^{H} \qquad (2 \times 8)
```

- **S**: la tabla de sensibilidades de ese píxel plegado: 8 filas (bobinas) × 2 columnas (los dos píxeles). S^H es la transpuesta conjugada.
- **U**: la matriz de desplegado de Pruessmann. Dice cómo combinar las 8 bobinas para obtener cada uno de los 2 píxeles. Funciona porque cada bobina ve con distinta intensidad la parte de arriba y la de abajo. Hay una U por píxel plegado: 512 × 1024 matrices de 2 × 8.
- **λ**: un número chico (10⁻⁴) que evita dividir por cero donde los mapas valen cero, fuera del cuerpo. Con λ = 0 es la fórmula original de Pruessmann.
- **m̂** ("m sombrero"): la imagen reconstruida, 1024 × 1024 complejos. El sombrero indica que es una estimación de m.
- U la calcula el ARM (PS) una vez por sesión, con una factorización L·D·L^H que no necesita raíces cuadradas.

## 4. Lo que hace el hardware: el MAC con U″

El FPGA no trabaja con y\_l ni con U: usa A\_l, tal como sale de la IFFT, y una versión de U que prepara el PS, llamada U″. Al final corre la coma con una potencia de 2.

```latex
\hat m(x,\; p + 512\,a) \;=\; 2^{\,S_1+S_2-19+E} \sum_{l=1}^{8} U''(x,p,a,l)\; A_l(x,p)
```

```latex
U' \;=\; \frac{U}{g_l}
\qquad\qquad
U'' \;=\; U' \cdot \frac{1}{K_c} \cdot 2^{-E}
```

En palabras: en cada píxel, la imagen final es la suma sobre las 8 bobinas de (coeficiente) × (imagen plegada de esa bobina), multiplicada por una potencia de 2 fija.

- **a**: cuál de los dos píxeles superpuestos se calcula: a = 0 es la fila p, a = 1 es la fila p + 512.
- **U′**: U con la ganancia de cada bobina adentro. Así el FPGA usa A\_l directamente, sin deshacer g\_l.
- **E**: un exponente común para que U′ entre en 16 bits. Es como el exponente de un número en punto flotante, pero uno solo para toda la tabla. Con el fantoma, E = 15.
- **K\_c**: la ganancia del CORDIC (paso 5). También va dentro de U″, para que el FPGA no tenga que multiplicar por ella.
- **U″**: lo que el PS le entrega al FPGA: 2 × 8 números de 16 bits por píxel plegado, 32 MiB en total.

## 5. La magnitud con CORDIC

La imagen que se muestra es la magnitud de m̂. El CORDIC la calcula solo con sumas y corrimientos de bits, sin multiplicadores ni raíz cuadrada.

Lo que se quiere calcular:

```latex
|\hat m| \;=\; \sqrt{\operatorname{Re}(\hat m)^2 + \operatorname{Im}(\hat m)^2}
```

Cómo lo hace: arranca con el vector (x, y) = (|parte real|, |parte imaginaria|) y lo gira hacia el eje horizontal en 12 pasos.

```latex
\text{para } i = 0, 1, \dots, 11: \qquad
\begin{aligned}
x_{i+1} &= x_i + \sigma_i \, y_i \, 2^{-i} \\
y_{i+1} &= y_i - \sigma_i \, x_i \, 2^{-i}
\end{aligned}
\qquad
\sigma_i = \begin{cases} +1 & \text{si } y_i \ge 0 \\ -1 & \text{si } y_i < 0 \end{cases}
```

```latex
x_{12} \;\approx\; K_c \, |\hat m|
\qquad\qquad
K_c \;=\; \prod_{i=0}^{11} \sqrt{1 + 2^{-2i}} \;\approx\; 1{,}6468
```

- Cada paso gira el vector un ángulo de arctan(2^−i) hacia el eje horizontal: 45°, 26,6°, 14°, 7,1° y así. Multiplicar por 2^−i es correr bits, así que cada paso cuesta dos sumas.
- Al terminar, y queda casi en 0 y toda la longitud del vector quedó en x, agrandada por la ganancia fija K\_c. Como 1/K\_c ya viene dentro de U″, x\_12 es directamente |m̂|.
- Con 12 pasos y 5 bits de guarda, el error es menor que 1 LSB de la salida de 16 bits.
- La salida es |m̂| × 2^15, en 16 bits sin signo.

## Valores de nuestro sistema

Con la configuración actual y el fantoma de 1024 × 1024, la cadena completa da 51,96 dB de SNR contra la imagen verdadera.

| Símbolo | Qué es | Valor |
| --- | --- | --- |
| L | cantidad de bobinas | 8 |
| Af | aceleración: se mide 1 línea de cada Af | 2 |
| N | lado de la imagen final | 1024 |
| K\_l, A\_l | tamaño por bobina | 512 × 1024 complejos |
| g\_l | ganancia por bobina (fantoma) | entre 2,7·10⁻⁵ y 3,8·10⁻⁵ |
| S1, S2 | etapas que dividen por 2 (máscaras 801 y 384) | 4 y 2 |
| λ | regularización | 10⁻⁴ |
| E | exponente común de U″ (fantoma) | 15 |
| K\_c | ganancia del CORDIC con 12 pasos | 1,6468 |
| U″ | tamaño en memoria | 32 MiB |
| salida |  | m̂ |

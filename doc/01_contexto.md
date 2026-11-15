# Contexto de la tesis

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

## 1. Contexto de la tesis

### Objetivo

Reconstruir imágenes de resonancia magnética con **SENSE** (Pruessmann et al. 1999) a **20 fps** (50 ms por trama) en una **Kria KV260**: SoM K26, chip XCK26, Zynq UltraScale+ MPSoC.

### Parámetros del sistema

| Parámetro | Valor | Nota |
|---|---|---|
| Imagen final | **1024 × 1024** | Decidido el 2026-10-10. La descripción del proyecto todavía dice 512×512; hay que actualizarla |
| Bobinas | L = 8 | |
| Factor de aceleración | Af = 2, submuestreo en *phase-encode* | |
| k-space por bobina | 1024 (readout, completo) × 512 (PE, submuestreado) | La **FFT2D es de 1024 × 512** por bobina; el unfolding lleva la imagen plegada de 1024 × 512 a la final de 1024 × 1024 |
| Formato base | Q1.15 (16 bits) en los puertos y en DDR | El datapath interno de la FFT es más ancho (§3.4) |
| Datos | Dataset offline o simulación | No hay escáner en vivo por ahora |

**Por qué 1024 y no 512.** A 512, el acumulador del unfolding (1 MiB) y el intermedio de una bobina (512 KiB) entran en chip; a 1024 el acumulador son 4 MiB, contra 2,88 MiB de BRAM + URAM. La memoria externa pasa a ser **estructural**, no una elección, y eso hace al problema más representativo. 🟡

### SENSE en una línea

Con Af = 2, cada píxel de la imagen plegada `y_l` (bobina *l*) es la suma de dos píxeles verdaderos pesados por la sensibilidad de la bobina:

```
y = S · m        y ∈ ℂ^L,  S ∈ ℂ^(L×Af),  m ∈ ℂ^Af
m̂ = (Sᴴ S)⁻¹ Sᴴ y = U · y      con  U ∈ ℂ^(Af×L)
```

`U` es la matriz de *unfolding* de Pruessmann.

### Glosario de símbolos (2026-10-24)

Versión legible, con las ecuaciones renderizadas y un diagrama: doc **"SENSE en el FPGA: ecuaciones en limpio"** (https://claude.ai/artifact/H3aqtmtvK5coLz5koBDyZk).

El camino de una trama, de punta a punta:

```
escáner ─► K_l ──×g_l, a 16 bits──► K_q,l ──IFFT 2D (PL)──► A_l ──MAC con U''──► m̂ ──CORDIC──► |m̂| (imagen final, u16)
           (k-space, 8 bobinas)      (entra al FPGA)          (imagen plegada,      (imagen       (magnitud)
                                                               una por bobina)       compleja)
                       el PS calcula U'' una vez por sesión a partir de los mapas s_l
```

| Símbolo | Qué es | Dónde aparece |
|---|---|---|
| l, L | número de bobina (0 … 7) y cantidad de bobinas (L = 8) | todo |
| Af | factor de aceleración (2): se mide 1 de cada Af líneas de phase-encode | todo |
| m | la imagen verdadera (lo que se quiere reconstruir), 1024 × 1024 | §1 |
| s_l | mapa de sensibilidad de la bobina l: cuánto "ve" esa bobina de cada píxel | §1, PS |
| K_l | k-space medido por la bobina l (ya submuestreado): 512 filas × 1024 muestras, complejo | entrada |
| g_l | ganancia por bobina que lleva el pico de K_l a 0,9 del fondo de escala de 16 bits. Fija por sesión | §4.8, §4.9 |
| K_q,l | K_l·g_l redondeado a Q1.15 (16 bits): lo que realmente entra al FPGA | §4.9 |
| S1, S2 | cuántas etapas de la pasada 1 y de la pasada 2 dividen por 2 (máscaras 801 y 384 → S1 = 4, S2 = 2). **No confundir con la matriz S** | §3.5, §4.8 |
| A_l | salida de la IFFT 2D del FPGA para la bobina l: la imagen **plegada** de esa bobina, 512 × 1024, Q1.15 | §3.8, §4.9 |
| y_l | lo mismo que A_l pero en unidades físicas: y_l = A_l · 2^(S1+S2−19) / g_l | §4.9 |
| x, p, a | x: columna (readout, 0 … 1023); p: fila de la imagen plegada (0 … 511); a: cuál de los Af píxeles verdaderos se superponen en (x, p) (filas p y p + 512) | §4.9 |
| S | matriz de sensibilidades de un píxel plegado: L × Af (8 × 2), S[l, a] = s_l en el píxel a | §1, §4.10 |
| U | matriz de desplegado de Pruessmann, Af × L (2 × 8) por píxel plegado: U = (SᴴS)⁻¹Sᴴ. Combina las 8 bobinas para separar los 2 píxeles superpuestos | §1, §4.10 |
| U' | U con la ganancia de cada bobina adentro: U' = U / g_l (así el PL usa A_l directo, sin deshacer g_l) | §4.9 |
| E | exponente común (por sesión) para que U' entre en 16 bits: U'' = U'·2^−E. Como el exponente de un punto flotante, pero uno solo para toda la tabla | §4.9 |
| U'' | lo que el PS le entrega al PL: U' escalada por 2^−E (y por 1/Kc) en Q1.15. 32 MiB | §4.10 |
| λ | regularización de Tikhonov: (SᴴS + λI)⁻¹ en vez de (SᴴS)⁻¹, para que no se divida por cero donde los mapas se anulan | §4.10 |
| m̂ ("m sombrero") | la imagen reconstruida (estimación de m), compleja: m̂ = Σ_l U·y_l | §1, §4.9 |
| CORDIC | algoritmo de sumas y corrimientos que calcula la magnitud √(re² + im²) sin multiplicadores ni raíz: gira el vector hacia el eje x en n pasos de ángulo arctan(2^−i) y al final la coordenada x es la magnitud | §4.11 |
| Kc | ganancia del CORDIC (≈ 1,647 con 12 pasos): sale x = Kc·magnitud. 1/Kc se mete en U'' | §4.11 (allí figura como K_n) |
| G | bits de guarda del CORDIC (fracción extra para que los truncamientos no se acumulen) | §4.11 |
| D, F_OUT | D: corrimiento de sesión que lleva la suma del MAC a la escala de salida; F_OUT: bits fraccionales de la salida u16 (15) | §4.11 |
| T | muestras por bobina = N1·N2 = 512 × 1024 = 524 288 | §5.7 |

La cuenta del MAC, en palabras: **en cada píxel plegado (x, p), y para cada uno de los 2 píxeles verdaderos que se le superponen, la imagen final es la suma, sobre las 8 bobinas, de (coeficiente de desplegado) × (imagen plegada de esa bobina), por una potencia de 2 fija**:

  m̂(x, p + a·512) = 2^(S1+S2−19) · Σ_l U'(x, p, a, l) · A_l(x, p)

El 19 es log2(1024·512): la IFFT normaliza por el tamaño y el hardware no lo hace (solo divide por 2 en las S1 + S2 etapas que escalan).

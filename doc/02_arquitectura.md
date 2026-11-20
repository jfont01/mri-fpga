# Arquitectura del sistema

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).* Teoría de SENSE: [teoria_sense.md](teoria_sense.md). Integración en la placa: [07_integracion.md](07_integracion.md).

## 2. Arquitectura del sistema

### Partición PS / PL

**PS (ARM A53), una vez por sesión:**
1. Prescan → mapas de sensibilidad S (RSS o ESPIRiT).
2. A = SᴴS (hermitiana definida positiva) y factorización A = LDLᴴ, sin raíces cuadradas.
3. Resolver para obtener U = A⁻¹Sᴴ, de 2 × 8 complejos por píxel alias.
4. Pasar U a DDR4.

**PL (fabric), en cada trama:**
1. Por cada bobina: IFFT2D del k-space submuestreado → imagen plegada `y_l`. Pasada por filas (N = 1024), *corner-turn*, pasada por columnas (N = 512). El presupuesto de tiempo (abajo) necesita **P = 4 núcleos FFT en paralelo**; si se reparten por bobinas o dentro de una misma bobina se decide con la medición 4b ([07](07_integracion.md)). *(Hasta el 2026-11-16 decía "en serie sobre un único core", que no entra en el presupuesto.)*
2. MAC de unfolding: `m̂ = U · y`, una matriz-vector de 2 × 8 por posición alias. No hay divisiones ni sustituciones en tiempo real.
3. CORDIC para la magnitud `|m̂|`.

**Idea clave.** S no cambia entre tramas, así que todo el Cholesky (que originalmente iba a correr en tiempo real) se reduce a U precalculado. En tiempo real solo quedan la IFFT2D y un MAC complejo.

**Mapas en el PS.** Se calculan una vez por estudio, en punto flotante (A53 a 1,33 GHz con NEON). Estimación: ~79 Mflop, decenas de ms. 🟡 La dificultad real no es el cómputo:
- La división `I_c / I_ref` se va al infinito fuera del objeto, así que hace falta máscara o regularización.
- El cociente crudo trae ruido y hay que suavizarlo.
- Se necesita el centro del k-space completo (líneas ACS; fastMRI las trae).

### Memoria

- Todo lo del tamaño de una imagen o más (k-space, intermedios, `y`, U, `m̂`) va a **DDR4**.
- En chip (BRAM/URAM) quedan solo las estructuras internas de los cores: ROM de twiddles, líneas de retardo, FIFOs.
- Hay una FIFO en cada frontera donde un core de streaming toca AXI/DDR.
- **Excepción en estudio:** el *corner-turn* en URAM (estrategia 1, abajo).

### Presupuesto de tiempo (1024 × 1024, 200 MHz) 🟡

```
FFT2D:      8 bobinas × (512 filas × 1024 + 1024 columnas × 512) = 8,39 M muestras
Unfolding:  1024 × 512 posiciones × 16 MAC complejos            = 8,39 M MAC
A 200 MHz, una muestra o un MAC por ciclo:  41,9 ms / P    y    41,9 ms / M
```

| Configuración | FFT | Unfolding | Total | |
|---|---|---|---|---|
| P = 4, M = 1 | 10,5 ms | 41,9 ms | 52,4 ms | ✗ no entra |
| **P = 4, M = 4** | 10,5 ms | 10,5 ms | **21 ms** | ✓ |

**El unfolding cuesta lo mismo que la FFT**: paralelizar solo la FFT no alcanza. Los DSP no son la restricción: P = 4 y M = 4 son unos 80 de 1248 (6 %).

### Tráfico por trama 🟡

| Tráfico | Tamaño |
|---|---|
| Cuenta inicial, sin los coeficientes | ~82 MiB (~50 MiB con *corner-turn* en chip) |
| Del unfolding: imágenes plegadas (8 bobinas) | 16 MiB |
| **Coeficientes de unfolding (U)** | **32 MiB** = 524 288 × 16 complejos × 4 B |
| Salida (magnitud) | 2 MiB |

- **Los coeficientes son el doble que las imágenes**: 32 MiB × 20 fps = 640 MiB/s = 671 MB/s.
- Formas de bajarlos: guardar S y resolver el 2 × 2 en la PL (16 MiB); submuestrear los mapas, que son suaves, e interpolar; o reducir la precisión de U.
  - *⚠ A revisar (2026-11-17):* los 16 MiB de "guardar S" no cierran. S restringido a una posición plegada son L·Af = 16 complejos, igual que U'' (en total, L·N·N = Af·L·T = 8 388 608 complejos), así que a igual precisión ocupa lo mismo, y además habría que resolver el 2 × 2 en la PL en cada trama. Falta ver de dónde salía el número (¿menos bits para S?).
- Con P = 4 y 16 bits, y antes de contar los coeficientes, el sistema usaba ~80 % de un puerto HP y ~37 % de la DDR4. DS987 especifica 4 GB de DDR4 de 64 bits a 2400 Mb/s: **19,2 GB/s de pico teórico** (17,88 GiB/s).
  - La cifra de 17,07 GB/s usada antes coincide con el pico teórico de una DDR4-2133 (2133,3 MT/s × 8 B = 17,07 GB/s). Probablemente es una cuenta con 2133 MT/s y no una medición.
  - **H30 (2026-11-16, 🟡):** nuestros diseños configuran la DDR, muy probablemente, a **2133 MT/s**: se pidió 2400, pero el controlador quedó en 533,33 MHz (DPLL = 33,333 MHz × 64 / 2 = 1066,7 MHz; controlador = DPLL / 2; PHY : controlador = 2 : 1). Así que el 17,07 GB/s es, en la práctica, el pico de nuestros diseños. Se vio en el Tcl, en los hwh y en lo que escribe el FSBL (`psu_init.c`); ⏳ falta leerlo en la placa ([hallazgos](hallazgos.md), [07 §1](07_integracion.md)).
  - El "~37 %" sale de 6,4 GB/s (P = 4 núcleos × 200 MHz × 4 B × lectura + escritura) sobre 17,07. Contra 19,2 GB/s es el 33 %.
  - ⏳ Medir el ancho de banda **agregado** con varios puertos HP en paralelo (§7). El barrido de un solo puerto no dice cuánto da la DDR en total.

### Estrategias de paralelismo (en orden de valor)

1. ***Corner-turn* en chip con URAM** (URAM288 = 4K × 72, dos puertos: se escribe por filas y se lee por columnas en el mismo ciclo). Ahorra una pasada por DDR y evita el acceso con *stride* de 4 KiB. El tile mínimo lo fija N½ (§7): tiene que ser bastante más grande que 3 KiB.
   - **Capacidad.** Una bobina son 1024 × 512 × 4 B = **2 MiB**, que es justo lo que dan las 64 URAM guardando dos muestras de 32 bits en cada palabra de 72 (64 × 4096 × 64 bits = 2 MiB). **No entra un ping-pong de bobinas completas**: son 4 MiB, contra 2,25 MiB de URAM (64 × 288 Kib) + 0,63 MiB de BRAM (144 × 36 Kib). Verificado en NumPy (2026-11-16).
   - Opciones: una transpuesta *in situ* (alternando el orden de direcciones entre tramas, para escribir la bobina nueva donde se va leyendo la anterior), un *corner-turn* por bloques, o hacerlo en DDR.
   - **Estudio del 2026-11-19 (H33, verificado en NumPy a tamaño real):**
     - *In situ, un carril por pasada:* con la bobina escrita por filas en la dirección t (19 bits = 9 de fila + 10 de columna), la trama j se escribe en rotl(t, 10·j) y se lee en rotl(t, 10·(j+1)) (rotación de 19 bits; multiplicar por 2^10 módulo 2^19 − 1). El generador de direcciones es un rotador de 19 bits; el patrón se repite cada 19 tramas. 21 tramas al azar transpuestas bien.
     - *In situ, dos carriles por pasada (P = 4):* la pasada 1 escribe dos filas por ciclo y la pasada 2 lee dos columnas por ciclo; la dirección de cada trama sigue siendo una **permutación de bits** del índice (ciclo, carril), que cambia de trama a trama con período 34. Como las dos direcciones de un mismo ciclo difieren en un solo bit (el del carril), **repartir en dos bancos por la paridad de la dirección** garantiza que nunca chocan: 0 conflictos en 40 tramas, y es una prueba, no una muestra. Cada banco: 32 URAM = 1 MiB = media bobina; puerto A lee la palabra de 72 bits, puerto B escribe una mitad de 32 bits con *byte write enable* (UG573 v1.14, cap. 2: la URAM288 es 4K × 72, cada puerto hace una lectura o una escritura por ciclo y el puerto A se ejecuta antes que el B en el mismo ciclo; BWE de 8 o 9 bytes).
     - *Desacople:* la lectura de la trama j puede ir adelante de la escritura de la j + 1, nunca atrás. La pasada 2 lee columnas enteras sin huecos (lo que exige el R2²SDF) aunque la pasada 1 tenga huecos entre filas.
     - *Tiempo:* 262 144 ciclos por bobina con dos carriles (1,31 ms a 200 MHz); 8 bobinas en *pipeline* ≈ 9 × 1,31 = 11,8 ms (el presupuesto en papel daba 10,5 ms sin contar el llenado).
     - *Por bloques en DDR:* +3,2 GB/s en la fase FFT (1,6 escritura + 1,6 lectura); con bloques de 16 KiB (p. ej. 32 × 128) la eficiencia sería ~84 % (L / (L + N½), N½ ≈ 3 KiB) y los buffers ~768 KiB (en URAM). `tile_transpose` (tiles cuadrados con ping-pong) traería ráfagas de una fila de tile: 256 B con TILE = 64, ~8 % de eficiencia; habría que pasarlo a bloques rectangulares.
     - *Generador de direcciones (verificado):* la permutación de la trama j + 1 sale de la de la trama j con una regla fija, src_{j+1}[k] = src_j[σ[k]] con σ = (10, 9, 11, 12, …, 18, 0, 1, …, 8): un registro de 19 índices de 5 bits que se actualiza por cableado al cambiar de trama, y un mux 19:1 por bit de dirección. Con un carril es directamente una rotación.
     - *Una carril por pasada alcanza para 20 fps si las fases se solapan entre tramas (2026-11-19):* con P = 2 (un núcleo por pasada) la fase FFT tarda 9 × 2,62 = 23,6 ms (lo mismo que mide `fft2d_check_full.py`) y con un solo carril de `unfold` (2 MAC por ciclo) el desplegado tarda 21 ms. En serie suman 44,6 ms (22 fps, justo); solapando la FFT de la trama n + 1 con el desplegado de la n (A_l con doble buffer en la DDR, 2 × 16 MiB) la trama sale cada ~23,6 ms (≈ 42 fps). Tráfico solapado: ~4,1 GB/s en total, ≤ 1,6 GB/s por puerto HP (U'' baja a 1,6 GB/s y deja de llenar un puerto). Con un carril el corner-turn es la rotación sola, en un banco. Queda como alternativa a discutir frente a P = 4 / M = 4.
     - *Recomendación de Claude:* in situ en URAM con dos carriles y bancos por paridad: cero tráfico extra en la DDR, que es el recurso más ajustado. Contra: usa el 100 % de la URAM y no escala a bobinas más grandes. ⏳ A decidir con Julián. `modules/tile_transpose` (transpuesta por tiles en BRAM con ping-pong) es un candidato ([estado](estado.md)).
2. **Más puertos HP.** Hay tres caminos independientes al DDRC: Port3 = HP0, Port4 = HP1 + HP2 (compartido), Port5 = HP3 (UG1085). Conviene usar HP0 y HP3, y separar lecturas de escrituras.
3. **Paralelismo por bobina** (instancias independientes).
4. **Radix mayor** (R2⁴SDF, *multipath*): RTL nuevo, queda como última opción.

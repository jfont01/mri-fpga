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
1. Por cada bobina, en serie sobre un único core: IFFT2D del k-space submuestreado → imagen plegada `y_l`. Pasada por filas (N = 1024), *corner-turn*, pasada por columnas (N = 512).
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

- **Los coeficientes son el doble que las imágenes**: 640 MiB/s a 20 fps.
- Formas de bajarlos: guardar S y resolver el 2 × 2 en la PL (16 MiB); submuestrear los mapas, que son suaves, e interpolar; o reducir la precisión de U.
- Con P = 4 y 16 bits, y antes de contar los coeficientes, el sistema usaba ~80 % de un puerto HP y ~37 % de la DDR4. DS987 especifica 4 GB de DDR4 de 64 bits a 2400 Mb/s: **19,2 GB/s de pico teórico** (17,88 GiB/s).
  - La cifra de 17,07 GB/s usada antes coincide con el pico teórico de una DDR4-2133 (2133,3 MT/s × 8 B = 17,07 GB/s). Probablemente es una cuenta con 2133 MT/s y no una medición. ⏳ Verificar a qué velocidad configura la DDR el PS del diseño.
  - El "~37 %" sale de 6,4 GB/s (P = 4 núcleos × 200 MHz × 4 B × lectura + escritura) sobre 17,07. Contra 19,2 GB/s es el 33 %.
  - ⏳ Medir el ancho de banda **agregado** con varios puertos HP en paralelo (§7). El barrido de un solo puerto no dice cuánto da la DDR en total.

### Estrategias de paralelismo (en orden de valor)

1. ***Corner-turn* en chip con URAM** (URAM288 = 4K × 72, dos puertos: se escribe por filas y se lee por columnas en el mismo ciclo). Ahorra una pasada por DDR y evita el acceso con *stride* de 4 KiB. El tile mínimo lo fija N½ (§7): tiene que ser bastante más grande que 3 KiB.
2. **Más puertos HP.** Hay tres caminos independientes al DDRC: Port3 = HP0, Port4 = HP1 + HP2 (compartido), Port5 = HP3 (UG1085). Conviene usar HP0 y HP3, y separar lecturas de escrituras.
3. **Paralelismo por bobina** (instancias independientes).
4. **Radix mayor** (R2⁴SDF, *multipath*): RTL nuevo, queda como última opción.

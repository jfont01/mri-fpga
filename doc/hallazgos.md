# Hallazgos

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

Los descubrimientos que cambiaron el diseño o que hay que poder defender en la tesis. Cada uno dice **qué se encontró**, **cómo** (la herramienta o el experimento), **qué consecuencia tuvo** y **dónde está el detalle**. Cuando aparece uno nuevo, se agrega acá y en la sección que corresponda.

---

## Escalado y precisión

**H1. Dividir por 2 en todas las etapas destruye la imagen** (antes del 2026-10-10)
Con ½ en las 19 etapas de la IFFT 2D, el pico del Shepp-Logan quedaba en 0,9 LSB. El crecimiento real de la señal es de +3,87 bits pico, así que alcanzan 4 etapas escalando.
*Consecuencia:* `SCALE` por mariposa y `SCALE_MASK` en el núcleo. *Detalle:* [04 §4.5](04_precision.md).

**H2. Calcular todo sin redondear no entra y no serviría: la pared es la entrada** (2026-10-10)
Sin redondear, el ancho crece hasta 167 bits: 818 MiB por trama (17,15 GB/s, el 89 % de la DDR). Y con entrada de 16 bits el techo es 70,50 dB igual: un datapath de 20 bits ya llega a 70,32.
*Consecuencia:* datapath de 20 bits, twiddles de 18, puertos y memoria de 16. Lo que decide la calidad es el ancho de la entrada (y eso, el SNR de los datos reales). *Detalle:* [04 §4.3](04_precision.md).

**H3. Datos y twiddles tienen techos independientes, y el twiddle necesita ~4 bits menos** (2026-10-10)
Mapa de SQNR datos × twiddle con el modelo bit-exacto: manda el menor de los dos techos; cada uno sube ~6 dB por bit. El error del dato es absoluto y se amplifica aguas abajo; el del twiddle es relativo a una señal que ya viene atenuada en el mismo factor.
*Detalle:* [04 §4.2](04_precision.md).

**H4. Cota de peor caso del error de salida** (2026-10-12)
Suma de los errores de cada redondeo propagados por el resto de la FFT (lineal). Monte Carlo con ruido, tonos, DC y espectros al azar: nunca se superó, el margen mínimo fue 2,4 veces.
*Consecuencia:* tolerancia del gtest (⏳ reemplazar el `5e-3` fijo). *Detalle:* [04 §4.7](04_precision.md).

**H5. Con escalado completo, la SQNR cae ~6 dB cada vez que se duplica el lado** (2026-10-24)
Shepp-Logan 64 / 256 / 512: 47,6 / 36,5 / 30,7 dB. Por Parseval la energía de salida dividida por N² no depende de N, pero el ruido de redondeo crece con la cantidad de muestras.
*Consecuencia:* en la IFFT de SENSE se usan máscaras parciales. *Detalle:* [05 §5.7](05_verificacion.md).

**H6. En 2D lo que decide es cuántas etapas dividen y el margen, no cuáles** (2026-10-18)
Barrido exhaustivo con el fantoma (375 máscaras de la pasada 1, 46 de la 2): con el mismo S y sin saturar, la imagen cambia menos de 0,03 dB. Normalizar cada bobina a su propio pico da +1,9 dB gratis (la ganancia se compensa en U).
*Consecuencia:* máscaras provisorias 801 / 384 (51,03 dB, ≥ 6,5 dB de margen), normalización por bobina. ⏳ Validar con datos reales. *Detalle:* [04 §4.8](04_precision.md).

**H7. Entre pasadas se sabe exacto qué hay, y dividir al principio de cada pasada lo hace seguro** (2026-10-25)
Después de cada pasada de la ida y vuelta, el contenido es una FFT conocida por una potencia de 2, así que se puede calcular el mínimo de divisiones en cada frontera. Con las divisiones al principio, el pico de cualquier etapa interna queda acotado por el de la entrada o el de la salida de la pasada (demostrado y verificado en float y bit a bit); dividir al final satura.
*Consecuencia:* +18 dB en Shepp (54,3 contra 36,5); regla para los casos de ida y vuelta. *Detalle:* [04 §4.12](04_precision.md), [teoria_fft §6](teoria_fft.md).

## Redondeo

**H8. El redondeo *half-up* tiene sesgo** (2026-10-17)
En el gráfico de fase, los bins sin señal solo tomaban 0, π/4 y π/2: el error nunca era negativo. Al dividir por 2, la mitad de las veces el resultado queda justo en el medio, y *half-up* siempre sube: +¼ LSB por etapa en promedio. En 16 bits es casi la mitad del error.
*Detalle:* [04 §4.6](04_precision.md).

**H9. En 2D el sesgo no se reparte: se amontona en un punto** (2026-10-21 y 2026-10-25)
La transformada de una constante es un pico. En SENSE: la fila p = 0 de la imagen plegada tiene +7,5 LSB de error medio (contra +0,055 en el resto), y después del desplegado es una línea horizontal en el medio de la imagen. En la ida y vuelta: un pico de ~4 000 LSB en el píxel (0, 0).
*Detalle:* [04 §4.6](04_precision.md), [04 §4.12](04_precision.md).

**H10. Dos tercios del sesgo los pone el `cast` de salida, no las mariposas** (2026-10-26)
Midiendo punto por punto en el modelo bit a bit: convergente solo en las mariposas deja +5,6 de los +7,5 LSB de la fila p = 0; con convergente también en la salida baja a +0,02. El `cmul` no aporta (descarta 17 bits: un empate exacto casi nunca pasa).
*Consecuencia:* `ROUND_CONV` en mariposas y salida, no en el `cmul`. *Detalle:* [04 §4.6](04_precision.md).

**H11. En imágenes sintéticas, parte del error no es sesgo sino correlación con la señal** (2026-10-26)
En el seno y los anillos, el error en (0, 0) no baja con el convergente. No son los twiddles (con 24 bits da lo mismo). Se explica exacto por la media del error del espectro, y con un redondeo aleatorio (sin sesgo e independiente de la señal) baja a ~48 LSB.
*Consecuencia:* la medida del redondeo es Shepp y el fantoma, no el seno ni los anillos; con datos reales el ruido térmico hace de *dither*. *Detalle:* [04 §4.6](04_precision.md).

**H12. El redondeo convergente saca el sesgo, y en 16 bits además baja el error** (2026-10-26)
En el núcleo: sesgo de ~+0,06 a ~0 LSB en todos los casos (incluidas las máscaras de SENSE). Con datapath de 16 bits el error rms baja de ~1,0 a ~0,66 LSB (−3,7 dB); con 20 bits casi no cambia. En la `fft2d`: fila p = 0 del fantoma de +1,1 a +0,06 LSB; (0, 0) de Shepp 256 de 729 a 3 LSB y error máximo de 22,9 a 6,4 LSB.
*Consecuencia:* `ROUND_CONV = 1` por default (2026-11-15). *Detalle:* [04 §4.6](04_precision.md).

## Arquitectura y timing

**H13. Karatsuba no cierra** (antes del 2026-10-17)
El multiplicador complejo de 3 productos queda con WNS −0,918 ns a 200 MHz: las sumas extra van en fábrica, sobre el camino crítico. El directo (4 DSP, suma y resta adentro de los DSP) da 211 MHz.
*Detalle:* [03 §3.2](03_fft_r22sdf.md), [teoria_fft §7.3](teoria_fft.md).

**H14. El cuello de botella era sacar el producto del DSP sin registrar** (2026-10-17)
Después del ruteo, la configuración elegida cerraba con 50 ps: DSP sin PREG → redondeo en fábrica. Un registro antes del redondeo (Vivado lo absorbe como PREG) sube el margen a 0,903 ns, por 168 FF y 4 ciclos de latencia.
*Consecuencia:* `CMUL_PIPE = 1` por default (2026-10-18). El camino crítico pasó a la línea de retardo de la primera mariposa. *Detalle:* [05 §5.6](05_verificacion.md).

**H15. La IFFT sale gratis del mismo núcleo** (2026-10-18)
swap(FFT(swap(X))) = N·IFFT(X), con swap = cruzar real e imaginaria. Error 0; en hardware son cables.
*Detalle:* [teoria_fft §5](teoria_fft.md).

**H16. N = 512 (log₂N impar) entra en la misma estructura** (2026-10-18)
Las unidades R2² hasta M = 8 más una etapa radix-2 final sin twiddle. Usa los mismos 4 multiplicadores que N = 1024.
*Detalle:* [03 §3.1](03_fft_r22sdf.md).

**H17. El datapath 2D no necesita hardware de reordenamiento** (2026-10-18)
El orden bit-reversed de cada pasada es una permutación: la pasada 2 trabaja columna por columna y el desplegado es punto a punto, así que alcanza con guardar U en el mismo orden. Con k-space centrado la magnitud no cambia, así que tampoco hacen falta *shifts*.
*Detalle:* [03 §3.8](03_fft_r22sdf.md).

**H18. Las máscaras de SENSE y el convergente cierran a 200 MHz** (2026-10-26)
801 (N = 1024) y 384 (N = 512): +0,565 y +0,687 ns en *half-up*, +0,468 y +0,540 ns en convergente. El convergente cuesta 0,10–0,15 ns y ~15 LUT; la síntesis no lo distingue, aparece recién con el ruteo. El camino crítico es la mariposa escalada que alimenta un DSP.
*Detalle:* [05 §5.6](05_verificacion.md).

## Sistema, memoria y placa

**H19. El desplegado cuesta tanto como la FFT** (estudio de arquitectura, octubre de 2026)
8,39 M MAC por trama, igual que las muestras de la FFT 2D. Paralelizar solo la FFT no alcanza: hacen falta ~4 carriles de cada uno para 21 ms por trama.
*Detalle:* [02](02_arquitectura.md).

**H20. Los coeficientes de desplegado pesan el doble que las imágenes** (2026-10-21)
U'' son 32 MiB por trama (32 MiB × 20 fps = 640 MiB/s = 671 MB/s), contra 16 MiB de imágenes plegadas. Es el mayor riesgo de ancho de banda.
*Detalle:* [02](02_arquitectura.md), [04 §4.10](04_precision.md).

**H21. El pico de la DDR es 19,2 GB/s, no 17,07** (2026-10-17)
DS987: DDR4-2400 de 64 bits. El 17,07 que se venía usando coincide con una DDR4-2133. ⏳ Verificar a qué velocidad la configura el PS.
*Detalle:* [02](02_arquitectura.md). → Ver H30: nuestros diseños la configuran, muy probablemente, a 2133.

**H22. Un puerto HP rinde el 99 % del teórico, pero las transferencias chicas no** (bring-up, antes del 2026-10-10)
1586 MB/s de 1600 a 100 MHz con 128 bits; la mitad del pico se alcanza recién con ~3 KiB (N½), y el *flush* de caché es el 19 % del tiempo.
*Consecuencia:* buffers no cacheables; el *corner-turn* por DDR necesita bloques bastante más grandes que 3 KiB. *Detalle:* [07](07_integracion.md).

**H23. El núcleo acepta huecos entre tramas, pero no dentro de una trama** (2026-10-26)
Verificado en el RTL con 0, 1, 3, 5 y 100 ciclos entre tramas. Dentro de una trama, si `i_valid` cae, se reinicia el contador de la primera etapa.
*Consecuencia:* entre el DMA y el núcleo hace falta una FIFO de una fila entera. *Detalle:* [05 §5.7](05_verificacion.md), [07 §5](07_integracion.md).

**H30. La DDR de nuestros diseños corre, muy probablemente, a 2133 MT/s y no a 2400** (2026-11-16) 🟡
Se pidió DDR4-2400 (`PSU__DDRC__SPEED_BIN = DDR4_2400R`, controlador a 600 MHz), pero Vivado dejó el controlador en 533,33 MHz. La cuenta, con la referencia de 33,333 MHz:

```
VCO del DPLL   = 33,333 MHz × 64 (FBDIV)   = 2133,3 MHz
DPLL           = 2133,3 / 2 (DIV2)         = 1066,7 MHz
controlador    = 1066,7 / 2 (DIVISOR0)     =  533,33 MHz
DRAM (PHY 2:1) = 2 × 533,33                = 1066,7 MHz  →  2133 MT/s × 8 B = 17,07 GB/s
```

La relación PHY : controlador es 2 : 1 (UG1085 v2.5, `MEMC_FREQ_RATIO = 2`, §17; divisor en la Figura 37-6, cap. 37). Para 2400 haría falta el controlador a 600 MHz. El DPLL también alimenta TOPSW_MAIN (533,33 MHz), lo que probablemente explica por qué Vivado lo dejó así.
*Cómo se vio:* en la PC con WSL2 (revisión de la documentación), en `bringup/v0/build_block_design.tcl` y en los hwh de v0, v1 y v1.1. En la PC con Ubuntu (2026-11-16) se confirmó con los archivos generados de v0, que no están versionados: el hwh (`FBDIV = 64`, `DIV2 = 1`, `DDR_CTRL__DIVISOR0 = 2`, `DDR_CTRL__ACT_FREQMHZ = 533.328003`, `SRCSEL = DPLL`, `PSS_REF_CLK = 33.333`), **lo que escribe el FSBL** (`psu_init.c`: `DPLL_CTRL` ← `0x14000`, o sea FBDIV = 0x40 y DIV2 = 1; `DDR_CTRL` ← DIVISOR0 = 2, SRCSEL = DPLL) y el resumen `psu_init.html` (DDR_CTRL: pedido 600, real 533,328).
*Consecuencia:* el presupuesto se hace con 17,07 GB/s (la fase de FFT con P = 4 usa el 37 %; contra 19,2 sería el 33 %). ⏳ Confirmarlo en la placa (leer `DPLL_CTRL` y `DDR_CTRL` por JTAG, o en el paso 4b) y decidir si se reconfigura el DPLL. *Detalle:* [02](02_arquitectura.md), [07 §1](07_integracion.md).

## SENSE, desplegado y CORDIC

**H24. Un exponente común por sesión alcanza para U''** (2026-10-21)
U'' en 16 bits con un solo exponente para toda la tabla pierde 0,001 dB. El acumulador del MAC necesita NB_U + 20 bits y la suma no depende del orden.
*Detalle:* [04 §4.9](04_precision.md).

**H25. Con mapas enmascarados, sin regularizar falla el 83 %** (2026-10-21)
Donde los mapas se anulan, SᴴS es singular. Con Tikhonov λ = 10⁻⁴ no hay divisiones por cero y se pierde 0,02 dB.
*Consecuencia:* λ > 0 siempre (⏳ elegir el valor con datos reales). *Detalle:* [04 §4.10](04_precision.md), [teoria_sense §4](teoria_sense.md).

**H26. La magnitud mejora la SNR, y el CORDIC con n = 12 y 5 bits de guarda la calcula con menos de 1 LSB de error** (2026-10-21)
De 51,03 dB (complejo) a 51,99 dB (magnitud): se descarta la parte del error en cuadratura. Más iteraciones sin bits de guarda empeoran (n = 16 sin guarda: 15 LSB).
*Detalle:* [04 §4.11](04_precision.md), [teoria_cordic](teoria_cordic.md).

**H31. La parte imaginaria del MAC llega a 2^34: el acumulador necesita 36 bits, no 35** (2026-11-16)
En un producto complejo de 16 × 16 bits, ur·ai + ui·ar vale +2^31 exacto cuando las cuatro son −2^15; la parte real no pasa de 2^31 − 2^15. Con 8 bobinas, P_im llega a 2^34. La "cota refinada" de 35 bits (§4.9, 2026-10-21) solo valía para la parte real. Se vio recorriendo las 16 combinaciones de extremos en NumPy, al preparar los anchos de D2.
*Consecuencia:* P de 36 bits a la entrada del redondeo (en el DSP48E2 sobra: acumula en 48). *Detalle:* [04 §4.9 y §4.13](04_precision.md).

## Infraestructura y verificación

**H27. El vm no detecta twiddles de un ancho equivocado** (2026-10-17)
RTL y modelo leen el mismo archivo equivocado y coinciden; solo el gtest (contra la DFT) lo ve, con SQNR de ~2 dB.
*Detalle:* [08](08_trampas.md).

**H28. Dos errores de la infraestructura que daban PASS falsos** (2026-10-17 y 2026-10-20)
`make` no recompilaba al cambiar los defines del JSON (se probaba la configuración anterior sin aviso), y `run_gtest` daba PASS cuando los tests quedaban *skipped* por una excepción. Los dos corregidos.
*Detalle:* [08](08_trampas.md).

**H29. GCC para aarch64 usa FMA por default, y eso rompe el bit a bit del código del PS** (2026-11-15)
Con `-O2`, `aarch64-none-elf-g++` (GCC 13.3, Vitis 2024.2) junta `a*b + c` en `fmadd` aunque se compile con `-std=c++17`: 12 instrucciones FMA en `ps_sense.o`. El FMA redondea una sola vez, así que el resultado ya no es el de NumPy en el último bit. Se vio contando instrucciones con `objdump`, y con un chequeo en tiempo de ejecución (a·a − (1 + 2⁻²⁹) con a = 1 + 2⁻³⁰ da 2⁻⁶⁰ con FMA y 0 sin FMA).
*Consecuencia:* el código del PS se compila con `-ffp-contract=off` (y nunca con `-ffast-math`), y `build_buffer` se niega a correr si detecta contracción. Con eso, U'' sale idéntico bit a bit a `unfold_ref.py` en la PC y en el A53 (qemu). *Detalle:* [05 §5.8](05_verificacion.md), [08](08_trampas.md).

**H32. Los estudios del fantoma ya no reproducen sus tablas: heredan `ROUND_CONV = 1`** (2026-11-17)
`fft2d_mask_study.py` arma la `fft2d` con `CORE`, que no fija `ROUND_CONV`, así que desde que el default pasó a 1 (2026-11-15) `fft2d_mask_study`, `mac_unfold_study` y `mac_output_study` corren con redondeo convergente, y sus tablas de §4.8–4.11 son *half-up*. Se vio al verificar el parche de D2: `mac_output_study.py` daba 52,01 dB en vez de 51,96 (m̂ complejo 51,12 en vez de 51,03), igual antes y después del parche; con `ROUND_CONV = 0` inyectado en `CORE` reproduce la tabla de §4.11 exacta. Los otros dos no se corrieron: que hereden el modo sale de leer el código (importan `Chain` de `fft2d_mask_study`). Los estudios de redondeo (`fft_rounding_study`, `fft2d_roundtrip_study`) sí fijan *half-up*.
*Consecuencia:* las diferencias son chicas (+0,05 a +0,09 dB) y no cambian ninguna decisión, pero un estudio corrido hoy no coincide con la bitácora. Pendiente decidir cómo se arregla ([estado](estado.md)). *Detalle:* [04 §4.11](04_precision.md).


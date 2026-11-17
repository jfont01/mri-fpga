# Estado y roadmap

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

*Última actualización: 2026-11-17 (D2: P1–P7 y C1–C8 confirmados; modelo C++ de `cordic_mag` escrito y verificado con un arnés).* **Este es el archivo que se actualiza siempre**: cada vez que se cierra una tarea, una decisión o un resultado, se marca acá (y el detalle va al archivo de la sección que corresponda).

Marcas: ✅ hecho y verificado · 🟡 estimado o provisorio (cuenta en papel, fantoma, NumPy ideal) · ⏳ pendiente · — no aplica.

## La cadena completa

```
                      PL, en cada trama (objetivo: 20 fps = 50 ms)
k-space de 8 bobinas ──► IFFT 2D por bobina ──► MAC con U'' (desplegado) ──► CORDIC (magnitud) ──► imagen 1024 × 1024
 (16 bits, de DDR)        (2 núcleos FFT 1D        (8 bobinas → 2 píxeles                         (u16, a DDR)
                           + corner-turn)           verdaderos por píxel plegado)

                      PS (A53), una vez por sesión
mapas de sensibilidad S ──► U = (SᴴS + λI)⁻¹Sᴴ por LDLᴴ ──► U'' (16 bits, con g_l, 2^−E y 1/K_c adentro) ──► DDR (32 MiB)
```

Teoría de cada pieza: [teoria_sense.md](teoria_sense.md) (SENSE y el cálculo de U), [teoria_fft.md](teoria_fft.md) (R2²SDF, IFFT, 2D y arquitecturas comparadas), [teoria_cordic.md](teoria_cordic.md).

## Tabla de estado por bloque

| Bloque | Teoría / estudio | Modelo C++ | RTL | Síntesis e impl. | En placa | Dónde está el detalle |
|---|---|---|---|---|---|---|
| **Núcleo FFT 1D R2²SDF** (N = 1024 y 512, 20/18/16) | ✅ | ✅ bit-exacto (+ Python vectorizado) | ✅ | ✅ 200 MHz: +0,903 ns (784), +0,468 ns (801) y +0,540 ns (384, N = 512), 16 DSP | ⏳ | [03](03_fft_r22sdf.md), [04](04_precision.md), [05](05_verificacion.md) |
| **Redondeo convergente** (`ROUND_CONV`) | ✅ | ✅ | ✅ | ✅ cuesta 0,10–0,15 ns y ~15 LUT | — | [04 §4.6](04_precision.md), [05 §5.6](05_verificacion.md) |
| **FFT 2D** (dos núcleos + memoria entre pasadas) | ✅ | ✅ ciclo a ciclo, memoria ideal | ⏳ | ⏳ | ⏳ | [03 §3.8](03_fft_r22sdf.md), [05 §5.7](05_verificacion.md) |
| **Memoria y *corner-turn*** (URAM o DDR por bloques; cantidad de carriles) | 🟡 cuentas | — (ideal en el modelo) | ⏳ | ⏳ | ⏳ | [02](02_arquitectura.md), [07](07_integracion.md) |
| **Mapas de sensibilidad** (PS) | 🟡 solo mapas sintéticos | ⏳ | — (software) | — | ⏳ | [teoria_sense §6](teoria_sense.md) |
| **Cálculo de U** (PS: LDLᴴ, Tikhonov, escalas, buffer) | ✅ con el fantoma | ✅ `modules/ps_sense`, bit a bit con `unfold_ref.py`: regresión de 18 casos + 2 de 1024 × 1024, en la PC y en el A53 (qemu) | — (software) | — | ⏳ tiempo en el A53 | [04 §4.10](04_precision.md), [05 §5.8](05_verificacion.md), [teoria_sense](teoria_sense.md) |
| **MAC de desplegado** | ✅ con el fantoma; anchos de D2 ✅ (§4.13) | ⏳ D2 en diseño (bit-exacto en Python: `mac_cordic_fxp.py`, con el parche de D2) | ⏳ no empezó | ⏳ | ⏳ | [04 §4.9, §4.13](04_precision.md), [diseño D2](diseno_unfold_cordic.md) |
| **CORDIC de magnitud** | ✅ n = 12, G = 5; anchos ✅ (§4.13) | 🟡 `cordic_mag.hpp` escrito, verificado con un arnés contra el Python; falta el gtest (D2) | ⏳ | ⏳ | ⏳ | [04 §4.11, §4.13](04_precision.md), [teoria_cordic](teoria_cordic.md), [diseño D2](diseno_unfold_cordic.md) |
| **Reordenamiento de la imagen de salida** (bit-reversed en x y p) | 🟡 | — | ⏳ | ⏳ | ⏳ | [04 §4.11](04_precision.md) |
| **Integración PS–PL** (DMA, AXI, DDR, Vivado, Vitis) | 🟡 cuentas | — | ⏳ wrappers | ⏳ | 🟡 loopback v0 y 1 puerto medido | [07](07_integracion.md) |
| **Validación con datos reales** (fastMRI) | ⏳ | — | — | — | — | [04 §4.4, §4.8](04_precision.md) |
| **Documento de la tesis** | 🟡 la bitácora es la base | — | — | — | — | esta carpeta |

## Desglose de tareas por bloque

### Núcleo FFT 1D R2²SDF

- [x] RTL R2²SDF con N potencia de 2; con log₂N impar, etapa radix-2 final (`fft1d_r22sdf_tail`).
- [x] Formatos independientes (datapath, twiddles, puertos), `SCALE_MASK`, `INVERSE`, `CMUL_PIPE`, `ROUND_CONV`.
- [x] Modelo C++ bit-exacto y modelo Python vectorizado, los dos bit a bit iguales al RTL.
- [x] Regresión: gtest 102 corridas, sim y vm 51 casos (incluye tramas con huecos entre ellas).
- [x] Síntesis e implementación a 200 MHz con las máscaras 784, 801 y 384.
- [x] **`ROUND_CONV = 1` por default** (entrega 4, 2026-11-15): verificada (gtest, sim, vm y controles cruzados contra los gemelos).
- [ ] Tolerancia del gtest igual a la cota B (§4.7), en vez del `5e-3` fijo.
- [ ] Caso de gtest con espectro tipo k-space (pocas muestras grandes y muchas chicas).
- [ ] Wrapper AXI-Stream para la placa: el núcleo no tiene *backpressure* y exige N muestras seguidas por trama ([07 §5](07_integracion.md)).
- [ ] Decidir si la máscara pasa a ser configurable en tiempo de ejecución (con fastMRI).
- [ ] (Opcional) v1_1 a 250 MHz.

### FFT 2D

- [x] Convenciones y referencia NumPy (§3.8): pasada 1 por filas (1024), pasada 2 por columnas (512), salida bit-reversed, sin *shifts*.
- [x] IFFT con el mismo núcleo (swap real/imaginaria) y N = 512 con etapa final.
- [x] Modelo C++ ciclo a ciclo con memoria ideal (ping-pong por bobina); regresión de 28 casos (fantoma, imágenes, ida y vuelta).
- [x] Reparto de la escala: máscaras 801 / 384 (provisorias) y la regla "dividir lo justo y al principio" para la ida y vuelta.
- [x] Chequeo de tamaño completo (1024 × 512, 8 bobinas) contra Python: 0 diferencias (con *half-up*).
- [ ] Ese mismo chequeo con `ROUND_CONV = 1` (`fft2d_check_full.py` / `fft2d_dump`).
- [ ] Decidir la memoria del *corner-turn* y la cantidad de carriles (depende de la medición 4b en placa). Una bobina (2 MiB) entra justo en las 64 URAM, pero un ping-pong de bobinas completas (4 MiB) no: transpuesta *in situ*, por bloques o en DDR ([02](02_arquitectura.md), estrategia 1).
- [ ] Revisar `modules/tile_transpose` (2026-08-05), candidato para el *corner-turn* que la bitácora no mencionaba: transpuesta por tiles en BRAM con ping-pong; la dirección de lectura es un intercambio de bits del contador. Tiene RTL, modelo y wrappers, pero su JSON de regresión está vacío (0 casos) y no hay `build/`. Ojo: los puertos son de `2*NB_SAMPLE` bits con `NB_SAMPLE = 32` descrito como "{imag, real} de 16 bits": confirmar si van dos muestras por palabra.
- [ ] RTL: dos núcleos + controlador de memoria + *corner-turn*.
- [ ] vm de la `fft2d` (incluidos los casos de ida y vuelta) y síntesis e implementación.

### Cálculo de U y mapas (PS)

- [x] Algoritmo: A = SᴴS + λI, A = LDLᴴ sin raíces, U por sustitución; Tikhonov λ = 10⁻⁴; U'' con g_l, 2^−E y 1/K_c adentro; buffer en el orden del stream de la `fft2d` (§4.10). Referencia: `model/unfold_ref.py`.
- [x] `unfold_ref.py` a C++ (`modules/ps_sense/cpp/ps_sense.{hpp,cpp}`, 2026-11-15): 19 casos bit a bit (U, d, E, buffer), incluido 1024 × 1024, en la PC y en el A53 con qemu (§5.8). Compila bare-metal con `aarch64-none-elf-g++`.
- [x] Hallazgo H29: GCC para aarch64 contrae a FMA con `-O2`; hace falta `-ffp-contract=off` (chequeado en ejecución).
- [x] Infraestructura del código del PS (2026-11-15): `run_gtest` con vectores de `gen_ps_sense_vectors.py` (18 casos), `Makefile` del módulo (`compile-baremetal`, `compile-a53-linux`, `compile-host`, `run-host`, `run-qemu`, `run-full`), `-ffp-contract=off` en `scripts/Makefile`, plantilla de `create_module` borrada. Todo verde; mutaciones 9/9 (§5.8, §6.4).
- [ ] Integrar `libps_sense.a` en la app de Vitis (bare-metal): flags del compilador, pila, cómo llegan los mapas a la DDR.
- [ ] Medir el tiempo en el A53 (estimado: menos de 1 s para 172 Mflop).
- [ ] Mapas de sensibilidad: algoritmo (RSS o ESPIRiT con las líneas ACS) y máscara del objeto.
- [ ] Elegir λ con datos reales (también regula la amplificación de ruido, el g-factor).

### MAC de desplegado

- [x] Escalas, formato de U'' (16 bits), exponente común por sesión, acumulador de NB_U + 20 bits, suma independiente del orden (§4.9).
- [x] Modelo bit-exacto en Python del MAC + salida + CORDIC (`model/mac_cordic_fxp.py`).
- [x] D2, teoría (2026-11-16, §4.13): P de 36 bits (H31), M de 17 + G = 22 bits (suficiente y mínimo), CORDIC de 23 bits, redondeo *half-up* por defecto con `ROUND_CONV` (el modo mueve la media 0,016 LSB). Decisiones de Julián: dos módulos (`unfold` y `cordic_mag`), modelo rtlsim de un carril, *half-up* + parámetro, corregir antes `mac_cordic_fxp.py`. Diseño y propuesta en **[diseno_unfold_cordic.md](diseno_unfold_cordic.md)**. 2026-11-17: P1–P7 confirmados (P4: D por puerto; el código se muestra en el chat antes de entregarlo) y `modules/cordic_mag` creado. C1–C8 del modelo de `cordic_mag` confirmados; `cpp/cordic_mag.{hpp,cpp}` escritos, verificados contra el Python con un arnés (0 diferencias, 9/9 mutaciones; [diseño D2](diseno_unfold_cordic.md) §4.5) y **aplicados en el repo** (sin commitear). ⏳ **Próximo paso (en la PC con Ubuntu; cómo retomar y prompt en [diseño D2](diseno_unfold_cordic.md) §6):** generador de vectores (`model/gen_cordic_mag_vectors.py`), después `cordic_mag_stimulus.hpp` y el gtest; el parche, al final, con todo verificado con `run_gtest`.
- [x] D2, paso 1 (2026-11-17): parche de `model/mac_cordic_fxp.py` (`d2_mac_cordic_fxp.patch`, aplicado en `5c614a0`): defaults G = 5 y W_M = W_OUT + 1 + G = 22, `ROUND_CONV_M` y `ROUND_CONV_OUT`, `p_to_m` y `m_to_out` (los dos módulos), docstring con m̂ / K_n y H31; self-test 24/24, mutaciones 9/9, `mac_output_study.py` idéntico antes y después (detalle en [diseño D2](diseno_unfold_cordic.md) §5).
- [ ] Modelo C++ y su gtest contra Python (D2).
- [ ] RTL con M = 4 carriles (presupuesto: 10,5 ms con M = 4).
- [ ] **H32:** los estudios del fantoma (`fft2d_mask_study.py`, `mac_unfold_study.py`, `mac_output_study.py`) heredan `ROUND_CONV = 1` desde el 2026-11-15 (por el código: `CORE` no lo fija), así que no reproducen sus tablas (§4.8–4.11, *half-up*); corrido y comprobado solo con `mac_output_study.py`. Decidir: fijarles *half-up* (como se hizo con `fft_rounding_study` y `fft2d_roundtrip_study`), agregarles una opción, o rehacer las tablas con convergente.
- [ ] Cómo traer U'' de la DDR (32 MiB por trama × 20 fps = 640 MiB/s = 671 MB/s) y en qué orden (`pos_coil_a` o `coil_pos_a`). No entra en el chip: BRAM + URAM del XCK26 son 2,88 MiB (DS987 v1.6, *Programmable Logic*, tabla 5: 144 × 36 Kb y 64 × 288 Kb). Se lee entero en cada trama, en secuencia y en el orden en que lo consume el MAC. ⚠ Revisar la opción "guardar S y resolver el 2 × 2 en la PL (16 MiB)" de [02](02_arquitectura.md): S restringido a una posición plegada también son L·Af = 16 complejos, igual que U'' (8 388 608 en los dos casos), así que a igual precisión no ahorra memoria (2026-11-17).
- [ ] vm, síntesis e implementación.

### CORDIC de magnitud

- [x] Estudio: n = 12 iteraciones, G = 5 bits de guarda, error < 1 LSB; la ganancia 1/K_c va adentro de U'' (§4.11).
- [x] Modelo bit-exacto en Python.
- [ ] Modelo C++, RTL (unos 24 sumadores de ~23 bits), vm, síntesis.

### Integración PS–PL

- [x] v0: DMA en lazo DDR → AXIS FIFO → DDR a 100 MHz, `memcmp` OK.
- [x] Barrido de ancho de banda de un puerto HP: 1586 MB/s (99,1 % del teórico), N½ ≈ 3 KiB, el *flush* de caché es el 19 % del tiempo.
- [ ] 4b: ancho de banda **agregado** con varios puertos HP en paralelo.
- [ ] Verificar a qué velocidad configura la DDR el PS (2400 o 2133 MT/s). **🟡 H30 (2026-11-16): muy probablemente 2133 MT/s (17,07 GB/s).** En los archivos de diseño de v0: el Tcl pide `DDR4_2400R` (controlador a 600 MHz), pero el hwh y lo que escribe el FSBL (`psu_init.c`) dan DPLL = 33,333 MHz × 64 / 2 = 1066,7 MHz, controlador = DPLL / 2 = 533,33 MHz, y con PHY : controlador = 2 : 1 la DRAM queda a 1066,7 MHz. `psu_init.html`: DDR_CTRL pedido 600, real 533,328. Falta: leer `DPLL_CTRL` (0xFD1A002C) y `DDR_CTRL` (0xFD1A0080) en la placa (JTAG, o en el paso 4b) y decidir si se reconfigura el DPLL.
- [ ] v1: núcleo FFT en el datapath (DDR → DMA → FFT → DMA → DDR).
- [ ] FFT 2D en placa, después la cadena completa con el PS calculando U.
- [ ] Medir fps reales y comparar la imagen contra la referencia de Python.

### Datos reales (fastMRI)

- [ ] Conseguir y preprocesar datos *multicoil* (cómo se obtienen, formato, recorte a 1024 × 1024, submuestreo Af = 2).
- [ ] Medir el SNR de la adquisición → ancho de entrada 16 o 18 bits (§4.4).
- [ ] Revisar las máscaras 801 / 384 con datos reales, incluida la saturación interna (dividen casi todo al final).
- [ ] Repetir λ, NB_U y E con mapas reales.

### Documento de la tesis

- [x] Bitácora separada por secciones, con teoría, resultados (cómo se obtuvieron) y trampas.
- [ ] Esqueleto del informe y lista de figuras.
- [ ] Resultados de sistema: fps, recursos totales, calidad de imagen con datos reales.

## Roadmap (orden y dependencias)

1. **Cerrar B y C** (ahora, chico): ✅ `ROUND_CONV = 1` por default (2026-11-15); quedan los pendientes chicos del núcleo.
2. **D: software del PS y modelos C++ del resto de la cadena** (mediano, sin placa): infraestructura del PS, U en C++, MAC y CORDIC en C++. Con esto toda la cadena queda modelada bit a bit.
3. **Datos reales: fastMRI** (mediano, sin placa, en paralelo con 2): revisa máscaras, ancho de entrada, λ y mapas.
4. **Decisiones de arquitectura del sistema** (necesita la placa): medir el ancho de banda agregado y con eso decidir *corner-turn*, carriles (P = 4 FFT y M = 4 MAC para 21 ms por trama) y cómo traer U''.
5. **RTL de lo que falta** (lo más grande): `fft2d` con memoria, MAC con 4 carriles, CORDIC, wrappers AXI-Stream y DMA.
6. **Integración en placa** (grande): v1 con el núcleo, después la `fft2d`, después la cadena completa; medir fps y calidad.
7. **Escribir la tesis**.

El **camino largo** es 4 → 5 → 6. Los pasos 2 y 3 se pueden hacer sin la placa y alimentan las decisiones de 4.

## Riesgos y preguntas abiertas

Movido del §10 de la bitácora (2026-11-15); las notas en cursiva son el estado actual.

- **El unfolding sin RTL** es la mitad del presupuesto de tiempo.
- **Los coeficientes de unfolding (32 MiB por trama)** pueden ser el mayor consumidor de ancho de banda.
- **El ancho de entrada** depende de un dato que todavía no tenemos (el SNR de fastMRI).
- **El sesgo del redondeo** en 16 bits: ¿cuánto pesa en la imagen final? *Resuelto con el redondeo convergente (§4.6): desaparece, y con 16 bits el error rms baja de ~1,0 a ~0,66 LSB.*
- **El ancho de banda real de DDR4 para la PL**: el pico teórico según DS987 es 19,2 GB/s (el 17,07 de antes parece ser DDR4-2133). Falta medir el agregado con varios puertos HP (§7); la fase de FFT con P = 4 necesita ~6,4 GB/s. *H30 (2026-11-16, 🟡): nuestros diseños dejan la DDR, muy probablemente, a 2133 MT/s, así que el pico es 17,07 GB/s (37 % con P = 4, en vez de 33 %).*
- **Qué significa "20 fps a 1024 × 1024"** (2026-11-16). Weiger 2000 hace SENSE en tiempo real con 128 × 96 y 13–35 ms por imagen, y ningún escáner cartesiano adquiere 512 líneas de phase-encode cada 50 ms. Conviene presentarlo como **throughput de reconstrucción** (cuántas imágenes por segundo puede reconstruir el sistema), no como adquisición en tiempo real. Las matrices *multicoil* de fastMRI son más chicas que 1024 × 1024 (⏳ verificar con los datos).

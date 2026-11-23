# Bitácora técnica — Reconstrucción SENSE en tiempo real sobre Kria KV260

*Tesis de grado de Ingeniería Biomédica (UNC). Última actualización: 2026-11-16.*

Esta carpeta junta el contexto de la tesis, la teoría, las decisiones de diseño, los resultados medidos, la infraestructura y los pendientes. Tiene dos usos:

1. **Material para escribir la tesis.** Cada resultado dice cómo se obtuvo y con qué herramienta.
2. **Retomar el contexto** al empezar una sesión nueva.

Desde el 2026-11-15 está separada en archivos. Este es el índice. La numeración de secciones (§1 … §8) se mantiene: una referencia como "§4.6" se busca en el archivo `04_…`.

## Por dónde empezar

| Para… | Archivo |
|---|---|
| **Saber dónde estamos y qué sigue** (se actualiza siempre) | [estado.md](estado.md) |
| **Los descubrimientos importantes**, con cómo se obtuvieron | [hallazgos.md](hallazgos.md) |
| Entender la teoría de **SENSE y el cálculo de U** | [teoria_sense.md](teoria_sense.md) |
| Entender la teoría de la **FFT R2²SDF**, la IFFT, la 2D y las arquitecturas comparadas | [teoria_fft.md](teoria_fft.md) |
| Entender el **CORDIC** de magnitud | [teoria_cordic.md](teoria_cordic.md) |
| **Retomar en otra PC** (trabajo en curso de agentes) | [estado.md](estado.md) "Cómo retomar" y [wip/README.md](wip/README.md) |
| **Seguir con D2** (MAC y CORDIC: modelos C++ y RTL) | [diseno_unfold_cordic.md](diseno_unfold_cordic.md) |
| [datasets_reales.md](datasets_reales.md) | datasets públicos de k-space multibobina: comparación, licencias, recomendación, preprocesamiento a 8 × 512 × 1024 con Af = 2, medición de σ en LSB (2026-11-20) | — |

## Mapa de archivos

| Archivo | Contenido | Secciones |
|---|---|---|
| [01_contexto.md](01_contexto.md) | objetivo, parámetros del sistema, SENSE en una línea, glosario de símbolos | §1 |
| [02_arquitectura.md](02_arquitectura.md) | partición PS / PL, memoria, presupuesto de tiempo, tráfico por trama, paralelismo | §2 |
| [03_fft_r22sdf.md](03_fft_r22sdf.md) | diseño del núcleo FFT: estructura, multiplicador, `cast`, formatos, `SCALE_MASK`, redondeo, twiddles, convenciones de la FFT 2D | §3 |
| [04_precision.md](04_precision.md) | métrica, mapas datos × twiddle, *full precision*, ruido térmico, escalado, sesgo *half-up* y redondeo convergente, cota de error, máscaras 2D, MAC, U en el PS, CORDIC, ida y vuelta | §4 |
| [05_verificacion.md](05_verificacion.md) | modelo C++, etapas de verificación, gtest, casos de regresión, estado de verificación, síntesis e implementación, modelo de la FFT 2D | §5 |
| [06_infraestructura.md](06_infraestructura.md) | entorno, árbol del repo, estructura de un módulo, comandos, JSON, vectores, Makefile | §6 |
| [07_integracion.md](07_integracion.md) | plataforma, puertos HP, DDR, DMA, block design, Vivado y Vitis, restricciones de integración, decisiones de sistema, bring-up en placa | §7 |
| [08_trampas.md](08_trampas.md) | trampas conocidas | §8 |
| [diseno_unfold_cordic.md](diseno_unfold_cordic.md) | diseño del MAC de desplegado y del CORDIC (paso D2): decisiones, propuesta pendiente, anchos, interfaces, plan de verificación y cómo retomar | — |
| [estado.md](estado.md) | tabla de estado por bloque, desglose de tareas, roadmap, riesgos | (era §9 y §10) |
| [bibliografia.md](bibliografia.md) | bibliografía | §11 |
| [registro.md](registro.md) | registro de cambios (el más nuevo arriba), versiones por entrega, plan y pendientes anteriores | §12 |
| [teoria_sense.md](teoria_sense.md), [teoria_fft.md](teoria_fft.md), [teoria_cordic.md](teoria_cordic.md) | teoría, con la trazabilidad de cada fórmula | — |
| [hallazgos.md](hallazgos.md) | los descubrimientos, uno por uno | — |
| [SENSE en el FPGA ecuaciones en limpio.md](SENSE%20en%20el%20FPGA%20ecuaciones%20en%20limpio.md) | export de un documento de claude.ai con las ecuaciones de SENSE (el diagrama se perdió); lo reemplaza [teoria_sense.md](teoria_sense.md) | — |

## Cómo se mantiene

- **Cada resultado, decisión o cambio de infraestructura** se anota en el archivo de su sección, y además:
  - se marca en [estado.md](estado.md) (tabla y tareas);
  - si es un descubrimiento, se agrega a [hallazgos.md](hallazgos.md);
  - se registra en [registro.md](registro.md), arriba de todo.
- La teoría nueva va a los `teoria_*.md`, con la fuente o la verificación de cada fórmula.

## Convenciones

| Marca | Significa |
|---|---|
| ✅ | verificado: simulado, medido o comparado bit a bit |
| 🟡 | estimado: cuenta en papel o modelo simplificado (NumPy ideal) |
| ⏳ | pendiente |
| ❌ | descartado |

Los SQNR están en dB. "LSB" sin aclaración es el LSB del puerto de salida (2⁻¹⁵ en Q1.15).

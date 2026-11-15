# Integración en la placa: AXI, DDR, DMA, Vivado y Vitis

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md). Presupuestos de tiempo y tráfico: [02_arquitectura.md](02_arquitectura.md).*

Todo lo que se sabe (y lo que falta saber) para llevar el diseño a la KV260. Lo del block design sale de los Tcl de `bringup/`; lo de la plataforma, de los manuales de `books/amd-docs/`.

## 1. Plataforma

| Qué | Valor | Fuente |
|---|---|---|
| Placa | KV260 Starter Kit (SoM K26 sobre la carrier KV260) | Kria KV260 Starter Kit Datasheet |
| Chip | XCK26, Zynq UltraScale+ MPSoC; parte `xck26-sfvc784-2LV-c` | DS987, `bringup/scripts/config.mk` |
| Board part | `xilinx.com:kv260_som:part0:1.4`, conector `som240_1_connector` | `config.mk` |
| Recursos de la PL | 117 120 LUT, 234 240 FF, 1248 DSP48E2 | DS987, tabla 5 |
| Memoria en chip | BRAM + URAM ≈ 2,88 MiB (las 64 URAM288 son 2,25 MiB) | DS987, UG573 |
| DDR4 | 4 GB, 64 bits, 2400 Mb/s → **19,2 GB/s de pico teórico** (⏳ confirmar a qué velocidad la configura el PS) | DS987 |
| CPU | 4 × Cortex-A53 a 1333 MHz (block design de v1.1) | PG201, Tcl de v1.1 |

## 2. Caminos entre la PL y la DDR

- **Puertos HP** (`S_AXI_HPx_FPD`, esclavos del PS que usa la PL). Hay **tres caminos independientes al controlador de DDR (DDRC)**: Port3 = HP0, Port4 = HP1 + HP2 (compartido), Port5 = HP3 (UG1085). Conviene usar **HP0 y HP3** y separar lecturas de escrituras.
- **`M_AXI_HPM0_LPD`** (maestro del PS): con él el PS escribe los registros AXI-Lite de la PL (por ahora, los del DMA).
- Ancho del puerto HP en los diseños actuales: **128 bits** (`PSU__SAXIGP2__DATA_WIDTH`); a 100 MHz son 1600 MB/s teóricos por puerto.

## 3. Lo que hay armado: block design de v0 / v1 (medición de ancho de banda)

`bringup/v1/hardware/build_block_design.tcl`. v1 todavía no tiene la FFT: es el lazo de medición.

```
DDR ─HP0 (128 b)─ SmartConnect ─ AXI DMA (MM2S) ─AXIS─► AXIS Data FIFO ─AXIS─► AXI DMA (S2MM) ─ SmartConnect ─HP0─ DDR
PS ─HPM0_LPD─ SmartConnect ─ AXI-Lite del DMA (registros)
```

| IP | Versión | Configuración relevante | Manual |
|---|---|---|---|
| `zynq_ultra_ps_e` | 3.5 | `S_AXI_GP2` = HP0_FPD a 128 bits; reloj PL0 = 100 MHz | PG201 |
| `clk_wiz` | 6.0 | salida de 100 MHz | PG065 |
| `proc_sys_reset` | 5.0 | | PG164 |
| `axi_dma` | 7.1 | **sin Scatter/Gather**; MM2S y S2MM de 128 bits; ráfagas de 16; `sg_length_width` = 26 (hasta 64 MiB por transferencia) | PG021 |
| `smartconnect` | 1.0 | dos: control (HPM0_LPD → AXI-Lite) y datos (DMA → HP0) | PG247 |
| `axis_data_fifo` | 2.0 | la FIFO del lazo | PG085 |

**Mapa de direcciones:** registros del DMA en `0x8000_0000` (64 KiB); el DMA ve `0x0000_0000`–`0x7FFF_FFFF` (DDR_LOW, 2 GiB) a través de HP0.

**Firmware** (`bringup/v1/firmware/main.c`): bare-metal en el A53, *polling*, sin interrupciones ni SG. Barre tamaños hasta 8 MiB, mide el tiempo de DMA con `XTime` (contador genérico del A53, ~100 MHz) y aparte el de mantenimiento de caché (`Xil_DCacheFlushRange`); buffers alineados a 64 B (línea de caché del A53).

## 4. Flujo de herramientas (Vivado y Vitis)

- Infraestructura propia en `bringup/scripts/`: `Makefile` + `config.mk` (valores por defecto que cada versión puede pisar).
  - `make build` (proyecto, block design, síntesis, implementación, bitstream y XSA), `make bootbin`, `make elf` (plataforma y aplicación de Vitis, dominio `standalone_psu_cortexa53_0`), `make run` (por JTAG) y `make uart`.
  - Tcl de Vivado: `scripts/vivado/{project,block_design,synthesis,implementation,bit,xsa,common}.tcl`.
  - Tcl de JTAG (`xsdb`): `scripts/jtag/{check_target,init_pspl,program_pl,load_elf,run,reset_a53,common}.tcl`.
- **`make qspi` está prohibido** (`CONFIRM_QSPI=NO`) hasta validar el firmware por JTAG. Nada que toque la placa se corre sin que Julián lo pida en ese momento.
- Vivado y Vitis 2024.2, en `/tools/Xilinx/`.

## 5. Restricciones que el RTL actual le impone al sistema

Verificadas en simulación:

- **El núcleo FFT necesita N muestras seguidas por trama.** Si `i_valid` cae en el medio de una trama, el contador de la primera etapa se reinicia. Entre tramas sí puede haber huecos de cualquier largo (verificado en el RTL con 0, 1, 3, 5 y 100 ciclos, §5.7). Consecuencia: **entre el DMA y el núcleo hace falta una FIFO que junte una fila entera antes de soltarla**, o garantizar que el DMA no corte el stream dentro de una fila.
- **El núcleo no tiene *backpressure*:** `o_valid` sale sin `ready`. Del lado de la salida, el S2MM (o una FIFO) tiene que aceptar siempre.
- **La salida sale en orden bit-reversed** (en la `fft2d`, en los dos ejes). U'' se guarda en ese mismo orden, así que el MAC no necesita reordenar; la imagen final sí, en el PS o con direcciones en el escritor (§4.11).
- **Formato en los puertos:** complejos empaquetados `{imag[15:0], real[15:0]}` en 32 bits, igual que U'' y que la convención de AXI-Stream para I/Q.

## 6. Decisiones de sistema pendientes

| Decisión | De qué depende | Opciones | Detalle |
|---|---|---|---|
| Cantidad de carriles | presupuesto de 50 ms por trama | P = 4 FFT y M = 4 MAC dan 21 ms; con M = 1 no entra | [02](02_arquitectura.md) |
| *Corner-turn* | ancho de banda agregado (4b) | URAM en chip (dos puertos: se escribe por filas y se lee por columnas) o DDR por bloques grandes (N½ ≈ 3 KiB) | [02](02_arquitectura.md), §7 |
| Tráfico de U'' | ancho de banda agregado | 32 MiB por trama (640 MB/s a 20 fps). Opciones: guardar S y resolver el 2 × 2 en la PL (16 MiB), submuestrear los mapas e interpolar, bajar la precisión de U'' | [02](02_arquitectura.md), [04 §4.9](04_precision.md) |
| Orden de la imagen de salida | eficiencia de las ráfagas en DDR | reordenar en el PS o escribir con direcciones bit-reversed | [04 §4.11](04_precision.md) |
| Puertos HP | caminos independientes al DDRC | HP0 y HP3, lecturas y escrituras separadas | §2 |

---

## 7. Bring-up en placa

| Etapa | Estado |
|---|---|
| 1 · DMA vivo | ✅ (v0) |
| 2 · Transferencia chica (4 KB) | ✅ |
| 3 · Transferencia grande | ✅ (barrido hasta 8 MiB) |
| 4 · Curva de ancho de banda | ✅ |
| 4b · Ancho de banda **agregado** con varios HP en paralelo | ⏳ planeado |
| 5 · FFT en el datapath (v1) | ⏳ |
| v1_1 a 250 MHz | ⏳ planeado, no corrido |

- **v0:** DDR → MM2S → AXIS FIFO → S2MM → DDR a 100 MHz; éxito = `memcmp(A, B) == 0` ✅.
- **Barrido de ancho de banda** (100 MHz, puerto de 128 bits, teórico 1600 MB/s) ✅:
  - pico de **1586 MB/s (99,1 %)**;
  - **N½ ≈ 3 KiB** (tamaño al que se alcanza la mitad del pico);
  - **T_fijo ≈ 4,5 µs** por transferencia;
  - **el flush de caché es el 19 % del tiempo de DMA** (989 µs para 8 MiB), así que hay que usar buffers no cacheables.
- **Consecuencia para el *corner-turn*:** con N½ ≈ 3 KiB, un tile de lecturas de 256 B rinde muy poco. El tile tiene que ser bastante más grande que 3 KiB por acceso.
- **Notas prácticas:**
  - En WSL la UART es `/dev/ttyUSB0` (`config.mk` decía `ttyUSB1`), con `usbipd attach` en cada reconexión.
  - En el Tcl del `clk_wiz`: sacar `MMCM_*` y `CLKOUT*_JITTER`/`PHASE_ERROR` (Vivado los ignora) y agregar `PRIM_SOURCE {No_buffer}`.

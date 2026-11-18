# Bibliografía

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

## 11. Bibliografía

| Referencia | Para qué |
|---|---|
| Pruessmann et al. 1999 | SENSE, matriz de unfolding. ⏳ No está en `books/` |
| Weiger, Pruessmann & Boesiger 2000, *Cardiac Real-Time Imaging Using SENSE*, MRM 43:177–184 (`books/papers/mri/`) | SENSE en tiempo real (128 × 96, 13–35 ms por imagen); SNR con g y √R. Su ref. 12 (Eggers et al., ISMRM 1999) es un antecedente de SENSE en hardware dedicado: conviene conseguirla |
| Uecker, Zhang, Voit, Karaus, Merboldt & Frahm 2010, *Real-time MRI at a resolution of 20 ms*, NMR Biomed. 23(8):986–994, doi 10.1002/nbm.1585. ⏳ No está en `books/`; visto solo el resumen y notas de prensa (2026-11-18) | MRI en tiempo real: FLASH radial muy submuestreado + inversión no lineal regularizada, ~20 ms por imagen; la reconstrucción era el cuello de botella (según la prensa de la época, ~30 min de cómputo por minuto de película) |
| Nayak, Lim, Campbell-Washburn & Steeden, *Real-Time Magnetic Resonance Imaging*, JMRI 55(1):81–99, 2022 (online 2020). ⏳ No está en `books/` (PDF en el sitio del grupo de USC) | revisión del campo: hardware, adquisición, reconstrucción, compromisos y aplicaciones. Para el estado del arte y para definir "tiempo real" |
| Hansen & Sørensen 2013, *Gadgetron: An open source framework for medical image reconstruction*, MRM 69(6):1768–1776, doi 10.1002/mrm.24389. ⏳ No está en `books/` | reconstrucción en tiempo real en GPU (GRAPPA cartesiano, SENSE no cartesiano): la referencia de software |
| Siddiqui, Reza, Shafique, Omer & Kanesan 2017, *FPGA implementation of real-time SENSE reconstruction using pre-scan and Emaps sensitivities*, Magn. Reson. Imaging 44:82–91 (PubMed 28855113). ⏳ No está en `books/`; visto solo el resumen | **antecedente directo: SENSE en FPGA** (punto fijo de 16 bits en una Virtex-6, según otras fuentes); informa 0,164 ms a 200 MHz y SNR media > 29 dB. Falta ver qué tamaño y qué parte de la cadena (¿incluye la FFT?) |
| Inam, Basit, Qureshi & Omer 2020, *FPGA-based hardware accelerator for SENSE (a parallel MR image reconstruction method)*, Comput. Biol. Med. ⏳ No está en `books/`; visto solo el resumen | **el antecedente más cercano:** acelerador de SENSE en punto flotante de 32 bits, con Vivado HLS, en una **ZCU102 (Zynq UltraScale+, la misma familia que la K26)** junto al ARM; datos in vivo con 8, 12 y 30 bobinas. Leerlo para posicionar la tesis |
| Roemer et al. 1990 | *phased array*, RSS |
| Uecker et al. 2014 | ESPIRiT (mapas de sensibilidad) |
| He & Torkelson 1996, *A New Approach to Pipeline FFT Processor*, Proc. IPPS '96, pp. 766–770 (`books/papers/fft/`) | R2²SDF original; Tabla 1 (comparación de arquitecturas). Ojo: los nombres BF2I/BF2II del RTL están invertidos respecto del paper (Figura 4) |
| Garrido, *A Survey on Pipelined FFT Hardware Architectures*, J. Signal Process. Syst. 94:1345–1364, 2022 (online 2021), doi 10.1007/s11265-021-01655-1 (`books/papers/fft/`) | survey de arquitecturas FFT *pipelined*; la §5.1 (MDF) es la que sirve para P > 1 |
| Mukherjee & Choudhury 2018, *An Area Efficient 2D Fourier Transform Architecture for FPGA Implementation* (`books/papers/fft/`) | antecedente de FFT 2D: dos núcleos 1D con RAM ping-pong, pero solo 8 × 8 |
| Oppenheim & Weinstein 1972 / Welch 1969 | análisis de ruido de cuantización en FFT (base para §4.7) |
| Brown, Cheng & Haacke, *MRI: Physical Principles*, cap. 28 | *parallel imaging*: §28.4, ec. 28.42–28.47 (SENSE, factor g; usan B por S y R por Af) |
| Akçakaya et al., *MR Image Reconstruction*, caps. 2–3 y 6 | reconstrucción; §6.2: mapas de sensibilidad |
| Paul & Mathew, §2.5.1 y cap. 3 | regularización de Tikhonov (§2.5.1); cómo elegir λ: discrepancia, GCV, curva L (cap. 3) |
| Parhami, *Computer Arithmetic*, cap. 22 (*The CORDIC Algorithms*) | CORDIC; cita a Andraka 1998, que conviene conseguir |
| Khan, *Digital Design of Signal Processing Systems* | aritmética de punto fijo, arquitecturas |
| Smith & Smith 1995, *Handbook of Real-Time FFTs* | FFT en tiempo real |
| UG579 | DSP48E2: 27 × 18, *pattern detector*, redondeo convergente |
| UG573 | BRAM / URAM (URAM288 = 4K × 72, dos puertos) |
| UG1085 (TRM) | puertos HP y DDRC |
| DS987, DS925 | K26 SoM, Zynq UltraScale+ |
| PG021, PG085 | AXI DMA, AXIS FIFO |
| UG1089, UG1400, UG1137 | KV260, Vitis embebido, software MPSoC |
| UG1209, *Zynq UltraScale+ MPSoC Embedded Design Tutorial* (solo en línea, sin PDF: <https://docs.amd.com/r/en-US/ug1209-embedded-design-tutorial/Zynq-UltraScale-MPSoC-Embedded-Design-Tutorial-UG1209>) | flujo de diseño embebido en Vivado y Vitis |

*Las de Oppenheim/Welch las propuso Claude; conviene conseguirlas y verificar la cita exacta. He & Torkelson ya está en `books/papers/fft/` (verificado el 2026-11-16).*

*De `books/papers/fft/`: no citar el review de deep learning de 2025 (*A Comprehensive Review of Deep Learning-Based…*); el de WiMAX, como mucho, como ejemplo de aplicación.*

*Manuales de AMD: al citarlos, poner la versión del documento (p. ej. UG1085 v2.5, DS987 v1.6); los PDF son de distintas versiones y las herramientas, 2024.2. Dónde están: [06 §6.1](06_infraestructura.md).*

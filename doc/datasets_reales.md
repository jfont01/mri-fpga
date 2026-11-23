*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

# Datasets de MRI con k-space crudo multibobina para validar SENSE en la KV260

*Informe de investigación, 2026-11-20. Hecho por un agente; notas en `wip/datasets/NOTAS.md`. Scripts de verificación NumPy: `wip/datasets/scripts/verify_noise.py` y `wip/datasets/scripts/verify_snr.py` (copia durable; correr con `python3 -I`); los resultados están copiados en §3 y §5.*

Convenciones: "verificado" = lo leí en la fuente primaria (página oficial, paper del dataset, README/código del repositorio o el listado del servidor) y dejo la URL. "No verificado" = lo marco explícitamente. No me registré en ningún lado, no acepté licencias y no bajé datos: para los tamaños usé `HEAD` HTTP (solo cabeceras) o listados públicos de S3/Zenodo.

---

## 0. Resumen y recomendación

**Ningún dataset público tiene 1024 × 1024 nativo, y ninguno va a tener 1024 líneas de phase-encode** (a FOV clínico serían píxeles de ~0,2 mm). En cualquier camino hay zero-padding en PE de por lo menos ×2. Lo más cercano que encontré: el readout de 1024 muestras aparece *nativo* en fastMRI brain si hay FLAIR de matriz 512 (512 × 2 de sobremuestreo). La tabla del challenge 2020 muestra matrices FLAIR de hasta 512 × 512. Que haya archivos así en train/val **no lo verifiqué**.

### (a) Validación principal de la cadena (SNR, máscaras, λ, U'', figuras)

**Recomendado: fastMRI brain multicoil (NYU).**
- Crudo multibobina, completamente muestreado en train/val, Siemens 1,5 y 3 T, 6970 estudios.
- El readout viene con sobremuestreo ×2 (fastMRI recorta el GT "to compensate for readout-direction oversampling"). En cerebro axial la mitad externa del FOV de lectura es aire: ahí se mide el ruido sin pre-scan, y fastMRI no trae pre-scan.
- Bobinas: 2 a 44 según la tabla del challenge. Se filtran las de ≥ 8 y se comprimen a 8.
- Matriz: lo más cercano a 1024 que hay.
- Es el dataset que la bitácora ya planeaba usar, y es la referencia del campo.
- Contras:
  - registro y DSA;
  - tarballs grandes (el tamaño de los de brain no lo pude confirmar);
  - sin pre-scan de ruido;
  - para publicar imágenes la página pide escribir a `fastmri@med.nyu.edu`.

**Respaldo, y arranque inmediato: mridata.org, proyecto "NYU machine learning data"** (Hammernik et al. 2018).
- Rodilla, 2D TSE, Siemens Skyra 3 T, 15 bobinas, sin aceleración, ISMRMRD con el header completo.
- 100 archivos sueltos de 0,8 a 5,9 GB (los 768 × 770 pesan de 1,8 a 3,8 GB; el proyecto entero, 202 GB), **sin registro**, licencia CC BY-NC 4.0 (las figuras de la tesis están permitidas con atribución).
- Matriz codificada de hasta **768 × 770** (los 20 archivos de sagital PD).
- Sirve para armar y depurar todo el pipeline mientras se espera la aprobación de fastMRI, y alcanza por sí solo para las conclusiones si fastMRI se complica.

### (b) Narrativa de tiempo real

**Recomendado: OCMR, las series cine completamente muestreadas** (Ohio State).
- 139 archivos útiles: eco simétrico y sin aliasing de FOV. Mediana de 215 MB por archivo, bajada archivo por archivo desde S3 sin registro, CC BY-NC 4.0.
- Cada fase cardíaca es un k-space completo: con Af = 2 retrospectivo en cada fase se reconstruye una **película** a la tasa del hardware.
- Hay que decirlo con honestidad: esas series son, con casi seguridad, **cine segmentado en apnea** y no tiempo real (el paper dice que el cine completo se puede adquirir *"under long breath-holds"*; no lo vi escrito serie por serie). Las 212 series de tiempo real de OCMR tienen submuestreo **pseudoaleatorio**, que no sirve para SENSE clásico.

**Respaldo: CMRxRecon.**
- 2023: cine con 10 bobinas virtuales y un k-space "completo" **rellenado con GRAPPA**, porque se adquirió con R = 3.
- 2024: 30 a 34 bobinas sin preprocesar. Registro en Synapse.

**Hallazgo extra (licencia no encontrada):** los datos de prueba de Gadgetron incluyen cine cardíaco de tiempo real Siemens con TPAT R = 2 y R = 4, y *pre-scans* de ruido. Es el único caso que encontré de una adquisición **Af = 2 cartesiana uniforme hecha de verdad**, del estilo TSENSE (Kellman 2001). Sirve para mostrar SENSE clásico sobre datos acelerados prospectivamente y para validar el método de SNR.

### Para decidir el ancho de entrada (16 o 18 bits)

La regla de §4.4 está en **LSB**, así que conviene medir directamente el desvío del ruido de k-space en LSB y no un SNR en dB (§5):

```
σ_LSB,l = σ_k,l · g_l · 2^15        con  g_l = 0,9 / pico_l
```

- σ_k,l es el desvío del ruido por componente en el k-space de la bobina l, y pico_l su pico.
- La relación entre SNR y LSB depende del objeto. Con la elipse de prueba, 40 dB son 0,82 LSB; con el fantoma del repo eran 0,5 LSB (§4.4).
- Antecedente: Aali et al. (arXiv 2411.12919) dicen que *"The fastMRI dataset does not provide a noise pre-scan"*. Con su propia definición (pico y blanqueo) reportan **SNR nativo medio de 24 dB en knee PDFS y de 32 dB en brain T2**. Con la regla de §4.4, la rodilla estaría holgada en 16 bits y el cerebro cae en la zona gris (30–40 dB). Hay que medirlo con nuestra definición.

---

## 1. Tabla comparativa

Leyenda: ✅ sirve / 🟡 sirve con reparos / ❌ no sirve para nuestro caso.

| Dataset | Anatomía, adquisición | ¿Crudo multibobina? | Bobinas | k-space (RO × PE) | Casos | ¿Completo? | ACS / mapas | Formato | Descarga | Licencia | Dinámico | Para nosotros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fastMRI brain** | cerebro axial 2D: T1, T1post, T2, FLAIR; Siemens 1,5 y 3 T | sí | 2–44 (challenge Siemens) | variable; matriz 256×208 a 512×512, RO ×2 sobremuestreado (no verificado por archivo) | 6970 estudios; 70 748 cortes train, 21 842 val | train/val sí; test submuestreado | centro completo; sin mapas | HDF5 (`kspace`, `ismrmrd_header`) | tarballs por split; tamaño **no verificado** | DSA NYU: investigación y educación; publicaciones académicas con cita; permiso por mail para imágenes | no | ✅ **principal (a)** |
| fastMRI knee | rodilla coronal PD/PDFS, 2D TSE, ETL 4, 320×320, 0,5 mm | sí | 15 | 640 × 368 (ejemplo del paper; RO ×2; PE con columnas en cero) | 1398 públicos (1594 en el paper); train 973 vol / 34 742 cortes; val 199 / 7135 | train/val sí | centro completo | HDF5 | multicoil_val **192 GB**, train **931 GB** | DSA NYU | no | ✅ (alternativa a brain) |
| fastMRI prostate | próstata T2 TSE 320×320 (3 promedios); DWI EPI | sí | 10–30 | no informado | 312 pacientes, 1560 volúmenes | **cada promedio R = 2** (impares y pares) | `calibration_data` aparte | HDF5 | no informado | DSA NYU | no | 🟡 (Af = 2 "nativo", interesante) |
| fastMRI breast | mama DCE 3D, GRASP radial | sí | 16 | radial: 288 rayos, 83 particiones | 300 casos | — | — | HDF5 | ~4,5 GB por caso | DSA NYU | sí (DCE) | ❌ (no cartesiano) |
| **mridata "NYU machine learning data"** | rodilla, 5 secuencias 2D TSE, Siemens Skyra 3 T | sí | 15 | 640×368 (41), 640×484 (20), 640×646 (19), **768×770** (17), 768×672/676/768 (1 c/u) | 100 archivos (20 pacientes × 5) | sí ("acquired without acceleration") | mapas ESPIRiT 24×24 en el paper (no en mridata) | ISMRMRD | 0,8–5,9 GB por archivo (202 GB en total), sin registro | CC BY-NC 4.0 | no | ✅ **respaldo (a)** |
| mridata "Stanford Fullysampled 3D FSE Knees" | rodilla 3D FSE, GE 3 T | sí | **8** | 320 × 320 × 256 (kx, ky, kz) | 19 listados | sí | — | ISMRMRD | 1,71 GB por caso (32,6 GB los 19) | CC BY-NC 4.0 | no | 🟡 (8 bobinas nativas; matriz chica) |
| mridata "Stanford 2D FSE" | varias (miembro inferior, pelvis), GE 3 T | sí | 3–32 (16 archivos con 8) | 288–416 × 146–576 | 89 archivos | sí | — | ISMRMRD | 0,11–1,27 GB por archivo (42 GB en total; los 16 de 8 canales, 4,8 GB) | CC BY-NC 4.0 | no | 🟡 |
| Calgary-Campinas MC | cerebro 3D T1 GRE, GE 3 T | sí | 12 o 32 | 256 × 218 × 170–180 (x, ky, kz), **ya con IFFT en kx** | 167 volúmenes | 85 % de kz (resto en cero) | — | HDF5 | ~219 GB descomprimido; Google Form o CONP | **CC BY-ND 4.0** | no | 🟡 (matriz chica, ND) |
| SKM-TEA | rodilla 3D qDESS, 2 ecos, GE 3 T | sí (híbrido x, ky, kz) | 8 o 16 | DICOM 512×512×160 | 155 pacientes, ~25 000 cortes | **relleno con ARC** (PI 2×1 elíptico) | mapas JSENSE (24×24) | HDF5 | **900 GB comprimido**, sin descarga parcial | Stanford Research Use Agreement (AIMI, login) | no | ❌ (no es completo de verdad; 900 GB) |
| **OCMR** | corazón, cine; Siemens 0,55 / 1,5 / 3 T | sí | variable (ejemplo de 18) | RO ×2 sobremuestreado; ejemplo 160 × 120 | 165 series completas (279 cortes) + 212 de tiempo real (842) | completo (cine segmentado); tiempo real pseudoaleatorio | — | ISMRMRD | 132 GB el tar; **por archivo** desde S3 (mediana 208 MB) | CC BY-NC 4.0 + términos OSU | **sí** | ✅ **principal (b)** |
| CMRxRecon 2023 | corazón: cine y mapping; Siemens Vida 3 T | sí | 32 → **10 virtuales** | no informado | 300 sujetos (120 / 60 / 120) | "completo" = GRAPPA (R = 3) + POCS | 24 ACS | .mat | Synapse (registro) | investigación y educación | sí (cine) | 🟡 respaldo (b) |
| CMRxRecon 2024 | corazón multimodal, multivista | sí | 30–34 (tagging 15–41) | Acq X 288–512 | 330 voluntarios | no queda claro | sin mapas | .mat | Synapse (registro) | no comercial | sí | 🟡 |
| M4Raw | cerebro, 0,3 T; T1 SE, T2 FSE, FLAIR | sí | **4** | 256 × 195/198 | 183 sujetos, 2–3 repeticiones | sí | — | HDF5 estilo fastMRI | Zenodo: val 2,98 GB, train 12,69 GB | CC BY 4.0 (Zenodo) | no | 🟡 (validar el método de SNR por repeticiones) |
| Gadgetron test data | Siemens `.dat`: cine de tiempo real TPAT2/4, "generic" R1/R2, GRE, TSE, EPI | sí | no visto | no visto | 39 archivos | algunos R1 completos; TPAT R = 2 | pre-scan de ruido en los casos `snr` | Siemens `.dat` | 0,15–0,7 GB por archivo | **no encontrada** | sí | 🟡 (SNR con pre-scan; Af = 2 real) |
| Ejemplos de BART | `pics-brain`: k-space 180 × 230, 8 bobinas | sí | 8 | 180 × 230 | 1 | sí | — | `.cfl` | 2,6 MB | — | no | solo para probar código |
| IEEE DataPort "Free-breathing 2D Cartesian cardiac" | corazón sin gating, Philips 3 T | sí, con mapas del escáner | no informado | 264×186 y 480×334 | 1 voluntario | partial Fourier | mapas | .mat | 1,49 GB, **suscripción** | CC BY 4.0 | sí | 🟡 |
| USC speech RT-MRI / Harvard CMR | habla / corazón | sí | — | espiral / radial | 75 / 108 | — | — | — | — | — | sí | ❌ (no cartesianos) |

---

## 2. Detalle por dataset

### 2.1 fastMRI (NYU Langone + Meta AI), knee y brain

**Fuentes.**
- https://fastmri.med.nyu.edu/
- paper del dataset, arXiv 1811.08839v2: https://arxiv.org/abs/1811.08839
- knee: Knoll et al., RyAI 2020, https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599
- challenge 2020, brain: Muckley et al., IEEE TMI 2021, arXiv 2012.06318
- código: https://github.com/facebookresearch/fastMRI (MIT, **archivado el 2025-08-18**)

**Contenido (verificado).**
- Sitio:
  - knee raw: 1398 estudios (697 PD y 701 PDFS), sin contar lo reservado para el challenge;
  - brain raw: 6970 estudios (su tabla suma 7002);
  - prostate: 312 exámenes; breast: 300.
- Paper, §4.2:
  - knee: Siemens Skyra, Prisma y Biograph mMR a 3 T y Aera a 1,5 T; *"15 channel knee coil array and conventional Cartesian 2D TSE"*;
  - ETL 4, *"matrix size 320 × 320, in-plane resolution 0.5mm×0.5mm, slice thickness 3mm"*; TR 2200–3000 ms, TE 27–34 ms.
- Paper, apéndice B: el ejemplo de BART es *"a 640x368 undersampled MRI scan with 15 coils"*.
- Paper, §4.7: el GT se recorta a 320 × 320 *"to compensate for readout-direction oversampling that is standard in clinical MR examinations"*. O sea, **el readout viene ×2 sobremuestreado**.
- Brain (§4.3): 11 imanes en 5 sitios, Siemens 1,5 y 3 T, solo axiales 2D. Ojo: *"We used zero matrices to replace the k-space slices ≳5mm below the orbital rim"*. **Hay cortes enteros en cero** y hay que descartarlos.
- Tabla suplementaria del challenge 2020 (datos Siemens):

| Contraste | Matriz | Bobinas |
|---|---|---|
| T1 | 320×260 a 320×330 | 2–24 |
| T1POST | 320×260 a 320×320 | 2–24 |
| T2 | 320×240 a 384×384 | 2–28 |
| FLAIR | 256×208 a **512×512** | **2–44** |

  El texto agrega que *"the GE data did not contain frequency oversampling"* (los Siemens sí lo traen).
- Formato (paper, apéndice A):
  - un HDF5 por volumen; `kspace` con forma (cortes, bobinas, alto, ancho);
  - `reconstruction_rss` y `ismrmrd_header`, que es el XML de ISMRMRD copiado tal cual (FOV, dimensiones, parámetros);
  - atributos `norm` y `max`;
  - en test: `mask`, `acceleration` y `num_low_frequency`.
  - La máscara tiene tantos elementos como el **ancho**, así que **el ancho es PE** y el alto es RO.
- Splits: knee 34 742 / 7135 / 4092 cortes en train / val / test; brain 70 748 / 21 842 / 8852. Lo saqué de `tests/conftest.py` del repositorio.
- Convención (`fastmri/fftc.py`): `ifft2c` = `fftshift(ifft(ifftshift(·), norm="ortho"))`. **El k-space está guardado centrado y la normalización es unitaria.**
- `mri_data.py` lee del header `encodedSpace`, `reconSpace` y `encodingLimits.kspace_encoding_step_1`, y con eso calcula `padding_left` y `padding_right`. **En PE hay columnas en cero a los costados**, fuera de los límites de codificación.
- **No hay pre-scan de ruido.** Aali et al. (arXiv 2411.12919): *"The fastMRI dataset does not provide a noise pre-scan."*

**Descarga y tamaño.**
- Knee (Knoll 2020, tabla de archivos): multicoil_train 931 GB, multicoil_val 192 GB, multicoil_test 109 GB; singlecoil 88 / 19 / 7 GB.
- Brain: **no encontré el tamaño oficial**. Un mirror no oficial (hyper.ai) dice 1,3 TB para train y 1,65 TB para val, cifras raras que no uso.
- Los links son personales e intransferibles.

**Licencia (verificado, texto del DSA en el sitio).**
- Acceso no exclusivo y gratuito, "limited to internal research or education".
- ¶5.3: no distribuir, publicar ni reproducir *"other than for uses in academic publications and presentations with citations as provided in paragraph 6"*.
- ¶6: en abstracts citar "NYU fastMRI"; en manuscritos, Knoll et al. 2020 (doi 10.1148/ryai.2020190007) y arXiv 1811.08839, más un párrafo de agradecimiento **obligatorio** en métodos, que empieza *"Data used in the preparation of this article were obtained from the NYU fastMRI Initiative database…"*.
- Hay que destruir las copias al terminar el trabajo y no se puede usar el nombre de NYU ni el de fastMRI para promoción.
- Sección de contacto: *"For permission to publish images from the dataset,"* escribir a fastmri@med.nyu.edu.
- **Mi lectura:** una tesis es una publicación académica (¶5.3), pero conviene pedir el permiso por mail antes de poner figuras, y guardar la respuesta.

**Para nosotros.**
- Brain da la matriz más grande (FLAIR hasta 512 → RO de 1024) y fondo de aire para medir ruido.
- Knee tiene siempre 15 bobinas y matriz fija (640 × 368), pero el objeto ocupa casi todo el FOV de lectura (la pierna sigue fuera del FOV), así que el fondo para medir ruido es peor. Esto es razonamiento: no lo vi en los datos.

**Prostate** (Tibrewala et al., Sci Data 11:404, 2024, doi 10.1038/s41597-024-03252-w; https://pmc.ncbi.nlm.nih.gov/articles/PMC11032332/):
- T2 TSE axial 320 × 320, 10–30 canales, 3 promedios: *"first and third averages sampling the odd lines of k-space, and the second average sampling the even lines"*.
- `kspace` con forma (averages, slices, coils, readout, phase), y `calibration_data` aparte.
- Es un **Af = 2 adquirido de verdad**: sirve como control de que el submuestreo retrospectivo no esconde nada.
- Contras: matriz chica y bobinas de cuerpo y columna, no todas activas.

**Breast** (Solomon et al., RyAI 7(1), 2025, doi 10.1148/ryai.240345; arXiv 2406.05270):
- 16 canales, *"288 spokes were acquired continuously for 2.5 min with 83 partitions"*, GRASP radial, ~4,5 GB por caso.
- Dinámico pero **radial**: necesitaría *gridding*. Descartado.

### 2.2 mridata.org (Lustig, Berkeley, y Vasanawala, Stanford)

**Fuentes.**
- http://mridata.org/ (`/terms`, `/faq`, `/list`)
- Parseé las 27 páginas del listado: 256 entradas únicas.

**Términos (verificado).**
- *"Content on this website for research use only, and not for diagnostic or commercial use… licensed under a Creative Commons Attribution-NonCommercial 4.0 International License"*.
- Ojo: *"Use of mridata.org, including both the uploading and downloading of data, denotes agreement with the following terms"*. Bajar un archivo ya implica aceptar los términos, y por eso no bajé ninguno.
- La descarga individual no pide login: `/download/<uuid>` redirige a S3. La descarga en lote sí lo pide.
- Formato ISMRMRD: `/dataset/data` y `/dataset/xml`, según el FAQ.

**"NYU machine learning data"** (uploader `florianknoll`; ref. Hammernik et al., MRM 79:3055, 2018, doi 10.1002/mrm.26977).
- 100 archivos, todos de 15 canales, Siemens Skyra 3 T, `TxRx_15Ch_Knee`, TSE, "Fullysampled: Yes".
- Matriz codificada según el listado: 640×368 (41), 640×484 (20), 640×646 (19), **768×770** (17), 768×676, 768×672 y 768×768 (1 cada una).
- Ejemplo: UUID `936c76f0-b3af-41f0-9b2d-f468f6a71225`, sagital, FOV 280 × 280,7 mm × 4,5 mm, 33 cortes, 768 × 770; pesa 1 894 723 136 bytes (HEAD de S3).
- Tamaños por grupo (HEAD de los 100 archivos): 640×368, 41 archivos, 100,2 GB (0,91–5,88 GB c/u); 640×484, 20, 23,5 GB; 640×646, 19, 34,6 GB; 768×770, 17, 40,4 GB (1,78–3,82 GB c/u); los tres 768×67x/768, 0,8–1,9 GB.
- Paper (arXiv 1704.00447):
  - *"five 2D turbo spin echo (TSE) sequences… 20 patients… 15-element knee coil. All data were acquired without acceleration"*;
  - las 5 secuencias: cor PD 320×288, cor PDFS 320×288, ax T2FS 320×256, sag T2FS 320×256, **sag PD 384×307** (0,46 × 0,36 × 3 mm);
  - *"Coil sensitivity maps were precomputed from a data block of size 24 × 24 at the center of k-space using ESPIRiT"*.
- Inferencia, **no verificada**: los 768 × 770 son la sagital PD (768 = 2 × 384), y el "y" codificado (770, 646, 484, 368) incluiría sobremuestreo de PE. Solo se confirma leyendo el header (`encodedSpace`, `encodingLimits`).
- **Para nosotros:**
  - es el k-space nativo más grande con referencia completa que encontré sin registro;
  - con 768 → 1024 en RO hay que padear ×1,33;
  - en PE depende de cuántas líneas haya realmente.

**"Stanford Fullysampled 3D FSE Knees"** (ref. Epperson et al., SMRT 2013).
- 19 entradas, GE 3 T, **8 canales**, 320 × 320 con 256 "slices", FOV 160 × 160 × 153,6 mm, TR 1550 ms, TE 25 ms.
- Ejemplo: `9a740e7b-…`, 1 714 239 784 bytes.
- Es el que usan los tutoriales de BART (`webinar-preprocessing.ipynb` baja `P1.zip` de old.mridata.org).
- Ventaja: 8 bobinas sin compresión. Cada plano (ky, kz) de 320 × 256, tomado después de la IFFT en kx, es un k-space 2D completo.
- Contras: chico, y no sé si GE dejó el sobremuestreo.

**"Stanford 2D FSE"** (uploader `jycheng`, código https://github.com/MRSRL/mridata-recon/).
- 89 archivos GE 3 T, 3 a 32 canales (16 archivos con 8), matrices 288–416 × 146–576.
- Ejemplos: `6b0f4577-…` (8 canales, 320 × 512 × 24, FOV 280 × 560 mm), 514 MB; `20103d2f-…` (16 canales, 416 × 576 × 26), 808 MB.

**"Validation and Generalizability of Self-Supervised…"** (Yu et al. 2022): prospectivamente submuestreados (CS). No sirven como referencia.

### 2.3 Calgary-Campinas, multicoil

**Fuentes.**
- https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge
- https://sites.google.com/view/calgary-campinas-dataset/download

**Contenido (verificado).**
- 3D T1 GRE, 1 mm isotrópico sagital, GE Discovery MR750; 12 canales (117 estudios) o 32 (50).
- Matriz [Nx, Ny, Nz] = [256, 218, 170–180], con x readout, y PE y z slice-encode.
- *"Only 85% of kz is collected"*, el resto en cero.
- **Ya vienen con la IFFT en kx**: híbrido (x, ky, kz). 167 volúmenes, `.h5`.
- Descarga por Google Form o en el portal CONP; *"~219 GB uncompressed"* (multicanal).
- Licencia: *"Creative Commons Attribution-NoDerivatives 4.0 International Public License"*.
- Paper: Beauferris et al., Front. Neurosci. 16 (2022), doi 10.3389/fnins.2022.919186.

**Para nosotros:**
- matriz chica (218 × ~170 por plano);
- partial Fourier en kz;
- "NoDerivatives": si una reconstrucción propia publicada cuenta como obra derivada es discutible, y habría que consultarlo.
- No lo recomiendo.

### 2.4 SKM-TEA (Stanford AIMI)

**Fuentes.**
- https://github.com/StanfordMIMI/skm-tea, `DATASET.md`
- arXiv 2203.06823 (NeurIPS Datasets & Benchmarks 2021)

**Contenido (verificado).**
- *"the dataset is 900GB in a compressed format and 1.6TB in the expanded format… the download does not support downloading different directories for different tracks separately"*.
- Login en Stanford AIMI.
- *"All qDESS data was acquired with 2x1 parallel imaging with elliptical sampling. Missing k-space data was subsequently estimated using ARC (GE)… This data is considered the fully-sampled kspace."*
- `kspace` (Nx, Ny, Nz, ecos, bobinas), *"# coils is typically 8 or 16"*, complex64.
- `maps` estimados con JSENSE (calibración de 24 × 24).
- 155 pacientes, ~25 000 cortes.
- Licencia: *Stanford University Dataset Research Use Agreement*, no comercial. Lo vi solo en un mirror de Hugging Face y en la página general de AIMI; el texto exacto **no lo leí**.

**Para nosotros:** ❌. El k-space "completo" tiene la mitad de las líneas sintetizadas con ARC, así que el ruido no es independiente entre líneas, y además pesa 900 GB.

### 2.5 OCMR (Ohio State)

**Fuentes.**
- https://www.ocmr.info/ y `/download/` (términos)
- https://github.com/MRIOSU/OCMR
- arXiv 2008.03410v2
- https://registry.opendata.aws/ocmr_data/

**Contenido (verificado).**
- Versión del 2023-08-11: **165 series completas (279 cortes) y 212 submuestreadas prospectivamente (842 cortes)**. Se agregaron datos de 0,55 T (Free.Max).
- Paper: *"real-time, prospectively undersampled… free-breathing"*, con *"pseudo-random sampling"*.
- Las completas: *"no undersampling in the phase encoding direction"*. Que sean cine segmentado en apnea es una inferencia: el paper solo dice que *"it is possible to collect fully sampled cardiac cine and flow data under long breath-holds"*.
- ISMRMRD. `kData` de 9 dimensiones: [kx, ky, kz, coil, phase, set, slice, rep, avg].
- *"the readout dimension includes a factor of two oversampling"*.
- Ejemplo: 160 × 120 × 1 × 18 × 60 × …
- CSV de atributos (`ocmr_data_attributes.csv`, público), entre las 165 completas:
  - eco: 156 `sym` y 9 `asy`;
  - FOV: 148 `noa` y 17 `ali` (FOV con aliasing, que con SENSE pide varios juegos de mapas);
  - completas, simétricas y sin aliasing: **139**, de las cuales 50 son de 0,55 T, 44 de Sola 1,5 T, 35 de Prisma 3 T, 5 de Avanto y 5 de Vida.
- Listado S3 (`s3://ocmr`, sin firma):
  - `ocmr_cine.tar.gz` = 132 382 218 240 bytes (132 GB);
  - 165 archivos `fs` = 74,5 GB (mediana 208 MB, mínimo 72 MB, máximo 4,17 GB);
  - las 139 útiles suman 64 GB (mediana 215 MB).
- Licencia:
  - AWS Registry: CC BY-NC 4.0;
  - términos de OSU: investigación y educación, no vender ni monetizar, los derivados quedan sujetos a los mismos términos, *"all publications and presentations resulting from any use of this dataset must cite OCMR"* (el arXiv);
  - en presentaciones: *"OCMR Dataset from www.ocmr.info"*.
- El lector de Python ignora los *noise scans* (`# TODO: Currently ignoring noise scans`). **No verifiqué** que los archivos los traigan, pero el código sugiere que sí.

**Para nosotros:**
- sirve para la narrativa de película;
- las matrices cardíacas son chicas (no verificado), así que el padding hasta 1024 es grande y las conclusiones de precisión no deberían salir de acá.

### 2.6 CMRxRecon (MICCAI 2023, 2024 y 2025)

**2023.**
- Fuentes: Wang C, Lyu J et al., Sci Data 11:687 (2024), doi 10.1038/s41597-024-03525-4; arXiv 2309.10836; datos en Synapse, doi 10.7303/syn52965326.1.
- Siemens Vida 3 T, *"cardiac coil made up of 32 channels"*; cine TrueFISP segmentado con gating ECG y apnea, 12–25 fases de ~50 ms, 1,5 × 1,5 mm.
- *"The parallel imaging acceleration factor was R = 3"*.
- arXiv: *"The multi-coil k-space data were compressed to 10 virtual coils… The partial Fourier data were filled up using POCS… Each under-sampled k-space data was reconstructed using GRAPPA to acquire full k-space data"*.
- 24 líneas ACS, `.mat` (kx, ky, sc, sz, t), 120 / 60 / 120 sujetos.
- SNR medio (señal media sobre desvío del ruido): cine SAX 27,3 ± 9,7 y LAX 29,0 ± 8,9.
- *"Registered Synapse users are able to access the data without the need to get approved."*

**2024.**
- arXiv 2406.19043: 330 voluntarios, 30–34 bobinas (tagging 15–41), Acq X 288–512, *"no other preprocessing steps were performed"*, sin mapas.
- *"commercial use of the dataset itself is prohibited"*.

**2025.**
- Multicéntrico y multivendor (GE, Philips, Siemens, United Imaging), 1,5 y 3 T, Synapse syn59814210.
- Solo vi resúmenes en buscadores: **no verificado** en la fuente.

**Para nosotros:**
- 2023 tiene ruido "contaminado" por GRAPPA y por la compresión hecha sin un blanqueo documentado, así que no sirve para el SNR ni para λ;
- sí sirve para una película;
- 2024 está más crudo, pero no queda claro si las líneas están completas.

### 2.7 M4Raw (bajo campo)

**Fuentes.**
- Lyu M et al., Sci Data 10:264 (2023), doi 10.1038/s41597-023-02181-4
- Zenodo, doi 10.5281/zenodo.8056074 (V1.6)

**Contenido (verificado).**
- 0,3 T (Oper-0.3), **bobina de cabeza de 4 canales**, 183 voluntarios.
- Secuencias:

| Contraste | Secuencia | Matriz | Repeticiones |
|---|---|---|---|
| T1w | SE | 256 × 195 | 3 |
| T2w | FSE | 256 × 195 | 3 |
| FLAIR | IR-FSE | 256 × 198 | 2 |

- FOV 240 × 240, 18 cortes.
- HDF5 estilo fastMRI: (corte, bobina, FE, PE), con `ismrmrd_header`.
- SNR por repetición, con una ROI de señal de 100 × 100 y cuatro de ruido de 30 × 30 en las esquinas: T1 14,97, T2 12,32 y FLAIR 14,73 (lineal, ~22–23,5 dB).
- Zenodo (API): `cc-by-4.0`; train 12,69 GB, val 2,98 GB, test 4,98 GB, motion 2,48 GB, gre 4,78 GB.
- Un resultado de búsqueda decía que el repositorio de HKU lo lista como CC BY-NC-ND; la fuente primaria (Zenodo) dice CC BY 4.0.

**Para nosotros:**
- solo 4 bobinas: SENSE con Af = 2 anda, pero con 4 entradas en cero en el hardware no es representativo;
- su valor está en las **repeticiones**: permiten validar el método de SNR (la diferencia entre dos repeticiones da el ruido sin suposiciones de fondo).

### 2.8 Gadgetron y BART: datos de ejemplo

**Gadgetron** (`test/integration/data.json`; host `http://gadgetrondata.blob.core.windows.net/gadgetrontestdata/`).
- 39 archivos crudos Siemens `.dat`.
- Los casos `generic_grappa_snr_*` usan `measurement=1` (ruido, como dependencia) y `measurement=2` (imagen), **así que traen el pre-scan de ruido** y se reconstruyen "en unidades de SNR" (Kellman & McVeigh 2005).
- Tamaños (HEAD):

| Archivo | Tamaño |
|---|---|
| `R1_PEFOV100_PERes100` | 674,7 MB |
| `R2_tpat_PEFOV100_PERes100` | 365 MB |
| `R2_spat_PEFOV100_PERes100` | 365 MB |
| `rtgrappa/acc_data_with_device_2.dat` (el test lo compara con `grappa_rate2_cpu_out`) | 172 MB |
| `SAX_realtime_gt_TPAT4` | 255 MB |
| `PK_realtime_gt_TPAT4_6_8` | 280 MB |

- **No vi anatomía, matriz ni bobinas**: hay que convertir con `siemens_to_ismrmrd`, `twixtools` 0.24 o `mapVBVD`.
- Interpreto (**no verificado**) "tpat" como TGRAPPA o TSENSE (submuestreo uniforme intercalado en el tiempo) y "spat" como una referencia aparte.
- **Licencia de los datos: no la encontré.** Usarlos solo como control interno, sin figuras, salvo que se consiga el permiso.

**BART.**
- `bart-workshop/ismrm2016/pics-brain/ksp`: 1 × 180 × 230 × 8 (8 bobinas, 2,6 MB).
- `bart-webinars/webinar1/data/slice`: 512 × 2100 × 6 (por las dimensiones parece radial; no verificado).
- Sirven para probar el código y nada más.

### 2.9 Otros dinámicos

- **IEEE DataPort "Free-breathing 2D Cartesian cardiac MRI datasets"**:
  - Kunz, Ruschke y Heckel 2023, doi 10.21227/f057-dw29;
  - Philips Elition X 3 T, un voluntario, partial Fourier, mapas del escáner, una serie de alta resolución de 480 × 334;
  - 1,49 GB, CC BY 4.0, *"requires an IEEE DataPort Subscription"*.
  - Sería un Cartesiano de tiempo real genuino, pero pide suscripción y es partial Fourier.
- **USC speech RT-MRI** (Lim et al., Sci Data 8, 2021, doi 10.1038/s41597-021-00976-x): 75 hablantes, 83 fps, **espiral**.
- **Harvard CMR Dataverse** (El-Rewaidy): **radial**, según CMRxRecon y OCMR.
- Ninguno de estos es compatible con SENSE cartesiano.

---

## 3. Cómo llevar los datos a 8 × 512 × 1024 con Af = 2: opciones y qué implica cada una

Lo verifiqué con NumPy, con un objeto de prueba de 640 (RO, ×2) × 368 (PE). Los resultados numéricos están entre paréntesis.

**1. Zero-padding centrado en k-space hasta 1024 × 1024.**
- **No cambia σ_k ni el pico del k-space**, así que **no cambia σ en LSB** (0,590 contra 0,591 LSB).
- La imagen es una interpolación con sinc: no gana resolución y conserva el Gibbs.
- En la IDFT **sin normalizar**, que es lo que hace el hardware (solo divide por 2^(S1+S2)), el pico sale parecido al nativo (+18 % en el ejemplo, por el Gibbs de la interpolación en las dos direcciones) y el ruido de salida es **idéntico** al nativo (cociente 1,000).
- Consecuencias:
  - el estudio de máscaras (picos y saturación) es representativo, y el +18 % lo hace un poco conservador;
  - una adquisición *real* de 1024 × 1024 tendría ruido en todas las muestras, así que su ruido térmico a la salida sería √(N_1024 / N_nativo) veces mayor;
  - con datos padeados, el ruido térmico de la salida queda más bajo frente al redondeo interno de la FFT, que se reparte por toda la grilla: el caso es **pesimista** para los anchos internos.
  - Esto último es razonamiento, no lo simulé.

**2. Sobremuestreo ×2 del readout.**
- Quitarlo (recortar la mitad central del FOV en imagen y volver a k-space) deja **σ_k / √2 con el mismo pico** (0,708; pico 1,000). σ en LSB cae ×0,71 (−3 dB) y el cuantizador "ve" 3 dB más de margen.
- Mantenerlo (640 → 1024 en vez de 320 → 1024) interpola menos y deja una banda de aire en la imagen, útil para medir ruido también a la salida del FPGA.
- Para la decisión de 16 o 18 bits hay que usar la variante que represente al sistema real:
  - si el k-space de 1024 que entra al FPGA es "post-decimación" (un FOV de 1024 píxeles sin sobremuestreo), corresponde recortar;
  - si se interpreta como 512 píxeles con sobremuestreo ×2, corresponde mantenerlo.
  - Conviene decidirlo y dejarlo escrito en la bitácora.

**3. Compresión a 8 bobinas.** Blanquear primero con la covarianza de ruido Ψ y después aplicar una SVD con una sola matriz para todo el corte:

```
K_w = W · K,   W = chol(Ψ)⁻¹             (blanqueo, por muestra)
K_8 = P · K_w, P = 8 primeros vectores singulares izquierdos (8 × L)ᴴ
```

- En las virtuales el ruido queda blanco con σ = 1 (máximo de |cov − I| = 2·10⁻¹⁵).
- La energía de señal retenida hay que medirla (en el ejemplo de juguete, 91 %).
- Una sola matriz **conmuta con la FFT**: se puede aplicar al k-space antes del FPGA, como haría el escáner.
- La compresión geométrica (GCC, Zhang 2013) usa una matriz por posición x de lectura y necesita la IFFT en RO primero, así que **no es compatible** con mandar k-space crudo al FPGA. Usar SVD (SCC; en BART `bart cc -p 8 -S -A`).
- Comparar el g y el SNR con todas las bobinas contra las 8 virtuales.

**4. Af = 2 retrospectivo.** Tomar una de cada dos líneas de PE, incluida la de DC (índice 512, que es par). Verificado: `ifft2(K[:, ::2])` coincide **exactamente** con `full[:, :512] + full[:, 512:]` (error 7·10⁻¹⁶), que es el modelo de pliegue del hardware.
- El ruido por línea es el del dato original.
- Lo que el retrospectivo **no** reproduce:
  - el reordenamiento del tren de ecos de un TSE acelerado en serio (cambian el filtro T2 en k-space y el *blurring*);
  - el movimiento o las corrientes de Foucault entre líneas.
- El control es fastMRI prostate, que tiene promedios adquiridos con R = 2.

**5. Menos de 8 bobinas (M4Raw, con 4):** se pueden poner canales en cero (U'' con columnas nulas), pero el condicionamiento de SᴴS no se parece al de L = 8. No sirve para elegir λ ni E.

**6. Datos "completos" rellenados con GRAPPA o ARC (CMRxRecon 2023, SKM-TEA):** las líneas sintetizadas son combinaciones lineales de las medidas, así que el ruido **no** es independiente entre líneas, y con Af = 2 la mitad de lo que uno "mide" puede ser sintético, lo que hace la validación circular. Descartados para el SNR y para λ.

**7. Partial Fourier o eco asimétrico** (OCMR `asy`, CMRxRecon con POCS, Calgary al 85 % de kz): elegir archivos simétricos.

**8. Centrado.**
- fastMRI guarda el k-space centrado. Según `doc/03` §D, con k-space centrado |m̂| no cambia (aparece un factor (−1)^(x+p) común a todas las bobinas).
- Los **mapas** tienen que estar en la misma grilla que la salida del hardware, sin `fftshift` (o desplazados de forma consistente). Si no, U'' queda corrido N/2 respecto de A_l.

---

## 4. Plan de preprocesamiento concreto (fastMRI brain; igual para mridata NYU)

Librerías:
- `h5py` (fastMRI);
- `ismrmrd` 1.15.0 en PyPI (mridata, OCMR: `ismrmrd.Dataset(...)`, `ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())`, `acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)`);
- `lxml` (el header de fastMRI);
- `numpy`;
- `sigpy` 0.1.27 (`sigpy.mri.app.EspiritCalib`, con defaults `calib_width=24`, `thresh=0.02`, `kernel_width=6`, `crop=0.95`);
- BART (`cc`, `whiten`, `ecalib -r -m -c`, `resize -c`, `fft`, `pics`) y su wrapper de Python;
- opcionales: `fastmri` 0.3.0 (MIT) para las transformadas y `twixtools` 0.24 o `mapVBVD` para los `.dat` de Gadgetron.

1. **Selección.**
   - Leer `ismrmrd_header` de cada `.h5`: `encodedSpace.matrixSize` (x, y), `reconSpace`, `encodingLimits.kspace_encoding_step_1` (mínimo, máximo, centro), `acquisitionSystemInformation`, y la cantidad de bobinas (`kspace.shape[1]`).
   - Filtrar: `acquisition` igual a AXFLAIR o AXT2, x ≥ 768 (ideal 1024), ≥ 8 bobinas, 3 T y 1,5 T por separado.
   - Descartar los cortes con k-space en cero (orbital).
   - Contar las columnas de PE en cero (`padding_left` y `padding_right`) y revisar si hay partial Fourier.
2. **Ruido.**
   - Imagen compleja por bobina con `ifft2c` (ortho, así que σ_img = σ_k).
   - ROI en la banda externa del RO: fuera de `reconSpace`, dejando unos píxeles de margen respecto del borde extremo, donde el filtro anti-alias del receptor puede atenuar (no verificado).
   - Con esa ROI se estima Ψ (L × L) y σ por bobina.
   - Controles: que la ROI no tenga señal (histograma gaussiano, media ≈ 0) y que coincida con el ruido de las esquinas de alta frecuencia del k-space.
3. **Blanqueo** con W = chol(Ψ)⁻¹. En BART, `whiten` necesita las muestras de ruido; también se puede pasar la covarianza con `-c`.
4. **Compresión a 8** con SVD sobre el corte completo blanqueado. Guardar P y la energía retenida.
5. **Tamaño:** zero-padding centrado a 1024 en RO (manteniendo o recortando el sobremuestreo, según la decisión de §3.2) y a 1024 en PE (`bart resize -c 0 1024 1 1024` con el orden de BART: dimensión 0 = RO, 1 = PE, 3 = bobinas).
6. **Referencia completa** (1024 × 1024 × 8): sirve para los mapas y para la imagen de referencia.
7. **Af = 2:** `K_af = K[:, :, 0::2]`, 512 líneas de PE × 1024 de RO × 8 bobinas.
8. **Entrada al FPGA:**
   - g_l = 0,9 / pico_l por bobina y redondeo a Q1.15, exactamente como en §4.9;
   - pasar por el modelo bit a bit (`fft1d_r22sdf_fxp.py` / `fft2d_mask_study.py` / C++) y, para algunos cortes, por el vm.
   - Conviene que `gen_fft2d_stim.py` acepte k-space externo: no revisé si ya lo hace.
9. **Mapas** (en el PS, una vez por sesión), tres variantes para comparar:
   - **(i) ESPIRiT:** `bart ecalib -r 24 -m 1 ksp_full_1024 maps` (o `sigpy EspiritCalib(ksp, calib_width=24)`) sobre la región central de 24 líneas del k-space **completo**. Así se emula un *pre-scan* de calibración aparte y las 512 líneas que entran al FPGA no llevan ACS, que es SENSE clásico. Con `-c` (*crop*) los mapas quedan enmascarados, SᴴS es singular afuera y λ es imprescindible (ya está anotado en `teoria_sense`).
   - **(ii) RSS:** imágenes de baja resolución del centro de 24–32 × 24–32 (con ventana de Hann), padeadas a 1024 × 1024:
     ```
     s_l = I_l / sqrt( Σ_l' |I_l'|² )
     ```
     Sin máscara, con fondo ruidoso.
   - **(iii) De referencia:** mapas del k-space completo en resolución nativa, para separar el error de los mapas del error de SENSE.
10. **Imagen de referencia:** la combinación con mapas del dato completo, no RSS, para comparar magnitudes equivalentes:
    ```
    m_ref = Σ_l s_l* · y_l / Σ_l |s_l|²
    ```
    Además, SENSE R = 2 en float (`unfold_ref.py`) para separar el error del hardware del de SENSE.
11. **λ, E y NB_U:**
    - barrer λ (10⁻⁶ … 10⁻¹ × la mediana de diag(SᴴS), como propone la bitácora);
    - para cada λ: NRMSE contra la referencia, g medio y máximo, y rango de U' (de ahí salen E y NB_U);
    - elegir con la curva L o con el criterio de discrepancia (Paul & Mathew, cap. 3, ya en la bibliografía).
12. **Máscaras 801 / 384:**
    - con muchos cortes, bobinas y contrastes: picos por etapa, saturaciones y margen (peor caso de la población, no el promedio);
    - comparar contra 15 y 3 (escalar al principio) y contra 641 / 384;
    - si el margen cambia mucho entre contrastes o cortes, eso es un argumento a favor de la máscara configurable en tiempo de ejecución.
13. **Figuras para la tesis:** recortar la imagen final al FOV útil para mostrarla, y poner la atribución que pida cada licencia.

---

## 5. Cómo medir el SNR (y qué número usar para 16 o 18 bits)

1. **σ en LSB** (lo que decide el ancho, por la regla de §4.4: importa si queda por debajo de ~1 LSB), por bobina y después de g_l:
   ```
   σ_LSB,l = σ_k,l · g_l · 2^15,   g_l = 0,9 / pico_l
   ```
   - La bobina con el pico más grande es la que tiene σ_LSB más chico, así que es la que decide.
   - Reportar la distribución sobre cortes y contrastes, no solo el promedio.
2. **SNR en potencia**, con la misma definición de §4.1, para compararlo con la tabla de §4.4. Por Parseval, para ruido blanco, da lo mismo calcularlo en imagen o en k-space:
   ```
   SNR = 10·log10( (Σ|K_medido|² − N·2σ_k²) / (N·2σ_k²) )
   ```
   - Verificado: con 24, 32 y 40 dB verdaderos y σ_k estimado del fondo de la banda externa del RO, sale 24,01, 32,02 y 40,03 dB.
   - Esta definición **depende del FOV** (cuánto aire entra), así que es otra razón para decidir con σ_LSB.
3. **Fuentes de σ, en orden de preferencia:**
   - un pre-scan de ruido (Gadgetron, quizás OCMR);
   - la diferencia entre repeticiones, σ = std(I₁ − I₂) / √2 (M4Raw);
   - el fondo de las imágenes complejas por bobina, antes del RSS (fastMRI brain, mridata): es gaussiano y no hace falta la corrección de Rician o chi no central (Dietrich 2007);
   - la periferia del k-space (sesgada por la señal; solo como control).
   - El método del fondo se puede **validar** en un archivo que tenga las dos cosas: el pre-scan de Gadgetron, o M4Raw contra su diferencia de repeticiones.
4. **SNR y g de la reconstrucción SENSE: pseudo-réplicas** (Robson 2008).
   - Sumarle al k-space blanqueado 100–200 realizaciones de ruido con σ = 1 y reconstruir con la cadena bit a bit.
   - El desvío por píxel da el mapa de SNR, y de ahí g:
     ```
     g = SNR_completo / (SNR_R2 · √2)
     ```
   - Comparar contra el g analítico de U (`teoria_sense` §3).
   - Así se mide también cuánto ruido agrega la aritmética del FPGA en la imagen final con datos reales.

---

## 6. Lo que no pude verificar

- **fastMRI brain:**
  - tamaño de los tarballs;
  - forma real de `kspace` por contraste: si en train/val hay FLAIR con matriz 512, o sea 1024 de RO; la tabla de 2–44 bobinas y matrices es de los datos del *challenge*;
  - la cantidad típica de bobinas.
- **fastMRI knee:** si el "y" de 368 incluye sobremuestreo de PE; cuántas líneas reales hay entre los límites de codificación.
- **mridata NYU:** qué es exactamente el "y" de 770 (o 646, 484, 368) y si los ISMRMRD traen *noise scans*.
- Si OCMR trae *noise scans* (solo lo sugiere el código), las matrices y bobinas típicas de OCMR, y que sus series completas sean cine segmentado en apnea (inferencia).
- El texto exacto de la licencia de SKM-TEA (Stanford Research Use Agreement), y los datos de CMRxRecon 2025 (solo resúmenes de buscadores).
- Licencia, anatomía, matriz y bobinas de los datos de prueba de Gadgetron; la interpretación de "tpat" y "spat".
- La conveniencia de evitar los bordes extremos del FOV sobremuestreado (filtro anti-alias del receptor): es una precaución, no un dato.
- La afirmación de §3.1 sobre el redondeo interno (que el zero-padding es pesimista para los anchos internos) es razonamiento, no simulación.
- Los números de SNR de Aali et al. (24 y 32 dB) usan otra definición (pico y blanqueo) y otro subconjunto (5 cortes centrales; brain T2 y knee PDFS). No son trasladables directo a §4.4.
- La versión V1.6 de M4Raw en Zenodo dice CC BY 4.0; no leí la página de HKU que, según un buscador, dice otra cosa.

---

## 7. Qué tiene que hacer el autor

1. **mridata.org**, para arrancar ya, sin registro:
   - leer http://mridata.org/terms (CC BY-NC 4.0; bajar implica aceptarlos);
   - bajar 2 o 3 archivos de "NYU machine learning data": por lo menos uno de 768 × 770 (p. ej. UUID `936c76f0-b3af-41f0-9b2d-f468f6a71225`, 1,89 GB) y uno de 640 × 368;
   - opcional: uno de "Stanford Fullysampled 3D FSE Knees" (8 bobinas nativas, 1,71 GB);
   - en total, ~4 a 8 GB.
2. **fastMRI**:
   - en https://fastmri.med.nyu.edu/, leer el **Data Sharing Agreement de brain** (y de knee si se quiere) y mandar la solicitud con afiliación y uso propuesto;
   - al recibir los links, mirar los tamaños de los archivos de brain y bajar **un solo** archivo de `multicoil_val` o un lote de `multicoil_train`, lo que pese menos;
   - knee `multicoil_val` pesa 192 GB.
   - Mandar un mail a fastmri@med.nyu.edu pidiendo permiso para publicar imágenes reconstruidas en una tesis de grado, y guardar la respuesta.
   - Copiar el párrafo de agradecimiento del DSA (¶6.2) para la sección de métodos.
3. **OCMR**:
   - leer https://www.ocmr.info/download/ (términos de OSU) y https://registry.opendata.aws/ocmr_data/ (CC BY-NC 4.0);
   - usar `ocmr_data_attributes.csv` para elegir 3 a 5 archivos `fs` con `sym` y `noa` (SAX y LAX; 1,5 o 3 T) y bajarlos de `https://ocmr.s3.amazonaws.com/data/<archivo>` (~0,2 GB cada uno);
   - **no** bajar el tar de 132 GB.
4. **Opcionales:**
   - **M4Raw**: el zip de val pesa 2,98 GB (Zenodo, CC BY 4.0) y sirve para validar el SNR con repeticiones.
   - **Gadgetron**: `generic_grappa_snr_R1_PEFOV100_PERes100` (675 MB) y `R2_tpat` (365 MB), si se aclara la licencia; para el pre-scan de ruido y Af = 2 real.
   - **CMRxRecon**: cuenta en Synapse, solo si OCMR no alcanza.
5. **Bibliografía:** pasar las referencias de §8 a `doc/bibliografia.md` y, para cada dataset usado, agregar la nota de licencia y de atribución.

---

## 8. Referencias (verificadas en Crossref o en la fuente; formato para `doc/bibliografia.md`)

**Datasets**
- Zbontar J, Knoll F, Sriram A, et al. *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*. arXiv:1811.08839 (v2, 2019). https://arxiv.org/abs/1811.08839
- Knoll F, Zbontar J, Sriram A, Muckley MJ, et al. *fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning*. Radiol Artif Intell 2(1):e190007, 2020. doi:10.1148/ryai.2020190007
- Muckley MJ, Riemenschneider B, Radmanesh A, Kim S, et al. *Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction*. IEEE Trans Med Imaging 40(9):2306–2317, 2021. doi:10.1109/TMI.2021.3075856
- Tibrewala R, Dutt T, Tong A, Ginocchio L, et al. *FastMRI Prostate: A public, biparametric MRI dataset to advance machine learning for prostate cancer imaging*. Sci Data 11:404, 2024. doi:10.1038/s41597-024-03252-w
- Solomon E, Johnson PM, Tan Z, Tibrewala R, et al. *FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast Dynamic Contrast-enhanced MRI*. Radiol Artif Intell 7(1), 2025. doi:10.1148/ryai.240345
- Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. *Learning a variational network for reconstruction of accelerated MRI data*. Magn Reson Med 79(6):3055–3071, 2018. doi:10.1002/mrm.26977 (datos: mridata.org, "NYU machine learning data")
- Epperson K, Sawyer AM, Lustig M, Alley M, Uecker M. *Creation of Fully Sampled MR Data Repository for Compressed Sensing of the Knee*. SMRT 22nd Annual Meeting, Salt Lake City, 2013 (datos: mridata.org, "Stanford Fullysampled 3D FSE Knees"; cita tomada del listado de mridata)
- Beauferris Y, Teuwen J, Karkalousos D, Moriakov N, et al. *Multi-Coil MRI Reconstruction Challenge—Assessing Brain MRI Reconstruction Models and Their Generalizability to Varying Coil Configurations*. Front Neurosci 16, 2022. doi:10.3389/fnins.2022.919186
- Souza R, Lucena O, Garrafa J, Gobbi D, et al. *An open, multi-vendor, multi-field-strength brain MR dataset and analysis of publicly available skull stripping methods agreement* (CC-359). NeuroImage 170:482–494, 2018. doi:10.1016/j.neuroimage.2017.08.021
- Desai AD, Schmidt AM, Rubin EB, Sandino CM, et al. *SKM-TEA: A Dataset for Accelerated MRI Reconstruction with Dense Image Labels for Quantitative Clinical Evaluation*. NeurIPS Datasets and Benchmarks, 2021. arXiv:2203.06823
- Chen C, Liu Y, Schniter P, Tong M, Zareba K, Simonetti O, Potter L, Ahmad R. *OCMR (v1.0)—Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging*. arXiv:2008.03410, 2020. https://www.ocmr.info
- Wang C, Lyu J, Wang S, Qin C, et al. *CMRxRecon: A publicly available k-space dataset and benchmark to advance deep learning for cardiac MRI*. Sci Data 11:687, 2024. doi:10.1038/s41597-024-03525-4 (datos: doi:10.7303/syn52965326.1)
- Wang Z, Wang F, Qin C, Lyu J, et al. *CMRxRecon2024: A Multi-Modality, Multi-View K-Space Dataset Boosting Universal Machine Learning for Accelerated Cardiac MRI*. arXiv:2406.19043, 2024
- Lyu M, Mei L, Huang S, Liu S, et al. *M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research*. Sci Data 10:264, 2023. doi:10.1038/s41597-023-02181-4 (datos: doi:10.5281/zenodo.8056074)
- Kunz J, Ruschke S, Heckel R. *Free-breathing 2D Cartesian cardiac MRI datasets*. IEEE DataPort, 2023. doi:10.21227/f057-dw29
- Lim Y, Toutios A, Bliesener Y, Tian Y, et al. *A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images*. Sci Data 8, 2021. doi:10.1038/s41597-021-00976-x

**Métodos (para el preprocesamiento y el SNR)**
- Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. *SENSE: Sensitivity encoding for fast MRI*. Magn Reson Med 42(5):952–962, 1999. doi:10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S
- Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM. *The NMR phased array*. Magn Reson Med 16(2):192–225, 1990. doi:10.1002/mrm.1910160203
- Uecker M, Lai P, Murphy MJ, Virtue P, et al. *ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA*. Magn Reson Med 71(3):990–1001, 2014. doi:10.1002/mrm.24751
- Griswold MA, Jakob PM, Heidemann RM, Nittka M, et al. *Generalized autocalibrating partially parallel acquisitions (GRAPPA)*. Magn Reson Med 47(6):1202–1210, 2002. doi:10.1002/mrm.10171
- Kellman P, Epstein FH, McVeigh ER. *Adaptive sensitivity encoding incorporating temporal filtering (TSENSE)*. Magn Reson Med 45(5):846–852, 2001. doi:10.1002/mrm.1113
- Kellman P, McVeigh ER. *Image reconstruction in SNR units: A general method for SNR measurement*. Magn Reson Med 54(6):1439–1447, 2005. doi:10.1002/mrm.20713
- Robson PM, Grant AK, Madhuranthakam AJ, Lattanzi R, et al. *Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions*. Magn Reson Med 60(4):895–907, 2008. doi:10.1002/mrm.21728
- Dietrich O, Raya JG, Reeder SB, Reiser MF, Schoenberg SO. *Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils, parallel imaging, and reconstruction filters*. J Magn Reson Imaging 26(2):375–385, 2007. doi:10.1002/jmri.20969
- Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. *Array compression for MRI with large coil arrays*. Magn Reson Med 57(6):1131–1139, 2007. doi:10.1002/mrm.21237
- Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. *A software channel compression technique for faster reconstruction with many channels*. Magn Reson Imaging 26(1):133–141, 2008. doi:10.1016/j.mri.2007.04.010
- Zhang T, Pauly JM, Vasanawala SS, Lustig M. *Coil compression for accelerated imaging with Cartesian sampling*. Magn Reson Med 69(2):571–582, 2013. doi:10.1002/mrm.24267
- Inati SJ, Naegele JD, Zwart NR, Roopchansingh V, et al. *ISMRM Raw data format: A proposed standard for MRI raw datasets*. Magn Reson Med 77(1):411–421, 2017. doi:10.1002/mrm.26089
- Aali A, Arvinte M, Kumar S, Arefeen YI, Tamir JI. *Robust multi-coil MRI reconstruction via self-supervised denoising*. arXiv:2411.12919 (v4, 2025). Fuente de "fastMRI no trae pre-scan de ruido" y de los SNR nativos de 24 y 32 dB

*Nota sobre los años:* Crossref da el año de la versión en línea para ESPIRiT (2013), la variational network (2017), ISMRMRD (2016) y Zhang (2012). Arriba puse el año del volumen impreso.

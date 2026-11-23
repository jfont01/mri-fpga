# Notas de avance: datasets de MRI multibobina (2026-11-20)

Requisitos (del repo): 8 bobinas x 512 PE x 1024 RO, Af = 2 cartesiano en PE, SENSE clásico,
mapas en el PS (RSS o ESPIRiT con ACS). Usos: SNR (16 vs 18 b, regla §4.4: <30 dB -> 16 b, >=40 dB -> 18 b),
máscaras 801/384 (§4.8), λ y anchos de U'' con mapas reales, imágenes para la tesis.
Nota repo: doc/03 §D dice que k-space centrado (fftshift) da |m̂| idéntico.

## Avance

### fastMRI (verificado 2026-11-20)
- Sitio https://fastmri.med.nyu.edu/ : knee raw 1398 scans (PD 697, PDFS 701; excluye challenge), brain raw 6970 (tabla suma 7002),
  prostate 312 exámenes (T2 + DWI), breast 300 (DCE 3D GRASP radial). DSA por dataset, solicitud online con afiliación y uso.
  Licencia: uso interno de investigación/educación, no redistribuir "other than for uses in academic publications and presentations
  with citations" (¶5.3). "For permission to publish images from the dataset," -> contactar fastmri@med.nyu.edu (sección Contact).
  Citas: abstract "NYU fastMRI"; manuscrito Knoll 2020 RyAI doi 10.1148/ryai.2020190007 + arXiv 1811.08839; texto de agradecimiento obligatorio.
  Destruir copias al terminar.
- Paper arXiv 1811.08839v2 (Zbontar et al.): knee 15 canales, Cartesian 2D TSE, ETL 4, matriz 320x320, 0.5x0.5 mm, 3 mm, TR 2200-3000, TE 27-34,
  Siemens Skyra/Prisma/Biograph mMR 3T + Aera 1.5T. Ejemplo 640x368 con 15 bobinas (App. B). GT recortada a 320x320 "to compensate for
  readout-direction oversampling". HDF5: kspace (slices, coils, height, width), reconstruction_rss, ismrmrd_header XML. Test: submuestreado
  4x/8x con 8%/4% centro. Train/val completos.
- Knoll 2020 RyAI (PMC6996599): knee multicoil_train 931 GB, multicoil_val 192 GB, multicoil_test 109 GB, singlecoil train 88/val 19/test 7 GB.
- Muckley 2021 (challenge 2020, arXiv 2012.06318), Tabla 1 suplementaria (datos del challenge brain Siemens): matriz T1 320x260-320x330,
  T2 320x240-384x384, FLAIR 256x208-512x512; bobinas T1 2-24, T2 2-28, FLAIR 2-44. GE sin sobremuestreo de frecuencia.
- Prostate (Tibrewala 2024 Sci Data 11:404, doi 10.1038/s41597-024-03252-w, PMC11032332): T2 TSE 320x320, 10-30 canales, 3 promedios,
  1° y 3° líneas impares, 2° pares (R=2 por promedio), kspace (averages, slices, coils, readout, phase), calibration_data aparte.
  DWI EPI 100x100. 312 pacientes, 1560 volúmenes.
- Breast (Solomon 2025 RyAI doi 10.1148/ryai.240345, arXiv 2406.05270): 16 canales, radial stack-of-stars GRASP, 288 spokes en 2.5 min,
  83 particiones, ~4.5 GB por caso. NO cartesiano.
- Tamaños brain: no encontré oficial (hyper.ai mirror dice train 1.3 TB, val 1.65 TB, no confiable).

### fastMRI extra
- Repo facebookresearch/fastMRI: MIT (código), archivado 2025-08-18 (read-only). mri_data.py lee del ismrmrd_header encodedSpace,
  reconSpace y encodingLimits.kspace_encoding_step_1 -> padding_left/right (columnas de PE en cero a los costados).
- Prostate T2: cada promedio R=2 (impares / pares), interesante como Af=2 "nativo".

### Calgary-Campinas MC (verificado)
- https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge : 3D T1 GRE 1 mm iso sagital, GE Discovery MR750,
  12 canales (117) o 32 canales (50), matriz [256, 218, 170-180] (x readout, y PE, z slice), 85% de kz (resto cero), ya IFFT en kx (híbrido).
  .h5. 167 volúmenes. Descarga por Google Form o CONP. Licencia (página download): CC BY-ND 4.0. Multi-coil ~219 GB descomprimido.
  Paper: Beauferris et al. 2022 Front. Neurosci. doi 10.3389/fnins.2022.919186 (arXiv 2011.07952).

### SKM-TEA (verificado)
- DATASET.md del repo StanfordMIMI/skm-tea: 900 GB comprimido / 1.6 TB; no se puede bajar por track. Login en Stanford AIMI.
  qDESS 3D, 2 ecos, adquirido con PI 2x1 + muestreo elíptico, relleno con ARC (GE) => "fully-sampled" es ARC-completado.
  kspace híbrido (x, ky, kz, echo, coils), maps JSENSE con calib 24x24, # coils típicamente 8 o 16, complex64. matrix_shape [512,512,160].
  155 pacientes, ~25000 cortes (arXiv 2203.06823, NeurIPS D&B 2021).

### mridata.org (verificado, parseé las 27 páginas del listado)
- Terms: CC BY-NC 4.0, investigación, no diagnóstico ni comercial. Acceso abierto; descarga individual sin login (redirección a S3);
  batch requiere login. ISMRMRD (HDF5 /dataset/data, /dataset/xml).
- "Stanford Fullysampled 3D FSE Knees": 19 entradas listadas (paper dice 20), GE 3T, 8 canales, matriz 320x320 x 256 "slices",
  FOV 160x160x153.6 mm, TR 1550 TE 25. Ref Epperson et al. SMRT 2013. Ej. 9a740e7b...: 1,714,239,784 B (1.71 GB).
- "Stanford 2D FSE": ~90 entradas, GE 3T, 3-32 canales, matrices 288-416 x 146-576, fully sampled. Ej. 6b0f4577 (8 ch, 320x512x24,
  FOV 280x560 mm): 514 MB; 20103d2f (16 ch, 416x576x26): 808 MB. Código MRSRL/mridata-recon.
- "NYU machine learning data" (Hammernik 2018, MRM 79:3055, doi 10.1002/mrm.26977): Siemens Skyra, 15 canales, 2D TSE, fully sampled,
  encoded 640x368, 640x484, 640x646, 768x770 ... ; 936c76f0 (768x770, 33 cortes): 1.89 GB.
  Paper (arXiv 1704.00447): 5 secuencias x 20 pacientes, sin aceleración, mapas ESPIRiT con bloque 24x24; sag PD 384x307.
- "Validation and Generalizability of Self-Supervised..." (Yu 2022): prospectivamente submuestreados (no sirven como referencia).

### OCMR (verificado)
- www.ocmr.info: 165 fully sampled (279 cortes) + 212 prospectivamente submuestreados (842 cortes), versión 2023-08-11 (agregó 0.55T Free.Max).
  Términos (www.ocmr.info/download/): investigación y educación, no vender, derivados sujetos a los mismos términos, citar el arXiv;
  en presentaciones "OCMR Dataset from www.ocmr.info". Sin registro (aceptación de términos). AWS Open Data.
- S3 (listado público): ocmr_cine.tar.gz = 132,382,218,240 B (132 GB). Archivos individuales: fs 165 archivos, 74.5 GB total,
  mediana 208 MB (72 MB - 4.17 GB); us 212 archivos, 57.9 GB, mediana 90 MB. CSV de atributos: scanners 15sola 126, 30pris 115, 15avan 66,
  0_55freemax 61, 30vida 9. smp fs/pse. fov 'ali' 34 (FOV con aliasing -> SENSE con varios mapas).
- Paper Chen et al. arXiv 2008.03410v2 (2020): ISMRMRD HDF5, kData 9D [kx, ky, kz, coil, phase, set, slice, rep, avg],
  "readout dimension includes a factor of two oversampling". Ejemplo 160x120x1x18x60... Prospectivos: real-time, free-breathing, pseudo-random.
  fs: cine segmentado en apnea. Reader Python ignora scans de ruido (TODO) -> no confirmé si los h5 traen ruido.

### CMRxRecon (verificado)
- 2023: Wang C, Lyu J et al., Sci Data 11:687 (2024), doi 10.1038/s41597-024-03525-4; datos Synapse doi 10.7303/syn52965326.1
  (usuarios registrados de Synapse, sin aprobación). Siemens Vida 3T, bobina de 32 canales; cine TrueFISP segmentado, gating ECG, apnea,
  12-25 fases ~50 ms, 1.5x1.5 mm, GRAPPA R=3 en la adquisición (mapping R=2, PF 6/8). arXiv 2309.10836: "compressed to 10 virtual coils",
  PF rellenado con POCS, "fully sampled" = reconstruido con GRAPPA. 24 líneas ACS. .mat (kx,ky,sc,sz,t). 120 train / 60 val / 120 test.
  SNR medio (señal/sd ruido) cine SAX 27.3±9.7, LAX 29.0±8.9.
- 2024: Wang Z et al. arXiv 2406.19043: 330 voluntarios, 30-34 bobinas (tagging 15-41), Acq X 288-512, "no other preprocessing";
  sin mapas; uso educativo/investigación, prohibido uso comercial del dataset; Synapse syn54951257.
- 2025: MICCAI 2025, multi-centro/multi-vendor (GE, Philips, Siemens, UIH), 1.5/3T, Synapse syn59814210 (solo resúmenes de búsqueda).

### M4Raw (verificado)
- Lyu M et al., Sci Data 10:264 (2023), doi 10.1038/s41597-023-02181-4. 0.3 T (Oper-0.3), bobina de cabeza de 4 canales, 183 voluntarios,
  T1w SE 256x195 (3 rep), T2w FSE 256x195 (3 rep), FLAIR 256x198 (2 rep), FOV 240x240, 18 cortes, BW 31.25 kHz.
  h5 estilo fastMRI (slice, coil, FE, PE), ismrmrd_header. SNR por repetición (ROI 100x100 señal / 4 ROI 30x30 esquinas):
  T1 14.97, T2 12.32, FLAIR 14.73 (lineal); promediado 24.4/20.1/20.2.
- Zenodo API: record 8056074 (V1.6) licencia cc-by-4.0; train 12.69 GB, val 2.98, test 4.98, motion 2.48, gre 4.78 GB.
  (Un resultado de búsqueda decía que el repositorio de HKU lo lista como CC BY-NC-ND; Zenodo dice CC BY 4.0.)

### Otros (verificado)
- Gadgetron test data (github gadgetron/gadgetron test/integration/data.json; host gadgetrondata.blob.core.windows.net/gadgetrontestdata/):
  39 .dat Siemens crudos. Casos generic_grappa_snr_* usan measurement=1 (ruido, dependencia) y measurement=2 (imagen) -> traen pre-scan de ruido.
  R1_PEFOV100_PERes100 (674.7 MB), R2_tpat (365 MB), R2_spat (365 MB), rtgrappa/acc_data_with_device_2.dat (172 MB; GRAPPA tasa 2),
  SAX_realtime_gt_TPAT4 (255 MB), PK_realtime_gt_TPAT4_6_8 (280 MB). Licencia de los datos: no encontrada (código MIT? no verificado).
- BART: ismrm2016/pics-brain ksp 1x180x230x8 (8 bobinas, 2.6 MB); webinar1 usa mridata 3D FSE knee P1. BART: cc (-p N, -S/-G/-E),
  whiten (necesita datos de ruido), ecalib (-r cal, -m maps, -c crop), resize -c.
- IEEE DataPort "Free-breathing 2D Cartesian cardiac MRI" (Kunz, Ruschke, Heckel 2023, doi 10.21227/f057-dw29): 1 voluntario, Philips
  Elition X 3T, partial Fourier, mapas del escáner, 264x186 y 480x334, 1.49 GB, CC BY 4.0, requiere suscripción IEEE DataPort.
- OCMR en AWS Registry: licencia CC BY-NC 4.0; s3://ocmr (us-east-2) --no-sign-request.
- Harvard CMR Dataverse (El-Rewaidy): radial (no cartesiano). USC speech RT-MRI (Lim 2021 Sci Data, doi 10.1038/s41597-021-00976-x): espiral.
- Aali et al. arXiv 2411.12919 ("Robust multi-coil MRI reconstruction via self-supervised denoising"): "The fastMRI dataset does not
  provide a noise pre-scan."; SNR nativo medio brain T2 32 dB, knee PDFS 24 dB (def: 10log10(1/sigma^2) tras blanquear y normalizar
  por percentil 99 de RSS de 24x24 ACS).

### Verificación NumPy (scripts/verify_noise.py en el scratchpad)
1) sigma_img = sigma_k/sqrt(N) con ifft2 (backward). 2) Recortar el sobremuestreo x2 de RO: sigma_k -> sigma_k/sqrt(2), pico igual,
sigma en LSB x0.71. 3) Zero-padding a 1024x1024: pico de la IDFT sin normalizar ~igual (+18% en el ejemplo, Gibbs 2D), ruido igual,
sigma en LSB igual. 4) Af=2: ifft2(K[:, ::2]) == full[:, :512] + full[:, 512:] exacto. 5) Blanqueo + SVD 15->8: ruido blanco sigma 1.

### Más datos (verificado)
- fastMRI tests/conftest.py: brain multicoil_train 70748 cortes, val 21842, test 8852; knee 34742/7135/4092.
- fastMRI fftc.py: ifft2c = fftshift(ifft(ifftshift(.), norm="ortho")) -> k-space guardado centrado, normalización ortho.
- OCMR fs: 156 sym / 9 asy; 148 noa / 17 ali; fs+sym+noa = 139 archivos (64 GB; mín 72 MB fs_0152_0_55T.h5; mediana 215 MB).
  Por scanner: 0.55T 50, 1.5 Sola 44, 3T Prisma 35, Avanto 5, Vida 5.
- mridata: 256 entradas únicas. NYU ML data: 100 archivos, todos 15 canales: 640x368 (41), 640x484 (20), 640x646 (19), 768x770 (17),
  768x676/672/768 (1 c/u). Stanford 3D FSE: 19 listados. Stanford 2D FSE: 89 (8 canales: 16).
- Crossref verificó todos los DOI de la bibliografía (ver informe).
- SNR (verify_snr.py): estimar sigma_k del fondo en la banda externa del RO y el SNR en potencia (def §4.1) recupera el valor
  verdadero (24/32/40 dB -> 24.01/32.02/40.03). Con la elipse de prueba 40 dB = 0.82 LSB (fantoma del repo: 0.5 LSB) -> la relación
  SNR -> LSB depende del objeto: medir sigma en LSB directo.

### Tamaños mridata (HEAD de 208 archivos)
- NYU ML: 640x368 41 arch 100.2 GB (0.91-5.88); 640x484 20 arch 23.5 GB; 640x646 19 arch 34.6 GB; 768x770 17 arch 40.4 GB (1.78-3.82);
  768x672/676/768 0.83/1.91/0.93 GB. Total ~202 GB.
- Stanford 3D FSE: 19 x 1.71 GB = 32.6 GB. Stanford 2D FSE: 89 arch 42.2 GB (0.11-1.27); los 16 de 8 canales 4.8 GB.

## Estado: informe final escrito en informe_datasets.md (2026-11-20).

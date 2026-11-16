# Trampas conocidas

*Parte de la bitácora de la tesis. Índice general: [bitacora.md](bitacora.md).*

## 8. Trampas conocidas

| Trampa | Síntoma | Qué hacer |
|---|---|---|
| Twiddles de un ancho distinto a `NB_TW` | gtest con SQNR ~2 dB; **el vm pasa igual** | una carpeta por formato; ⏳ chequear `tw_re[0]` al cargar |
| `make` no seguía `DEFINES` | se probaba la configuración anterior sin aviso | corregido el 2026-10-17 (`makefile_fix`); con el Makefile viejo, `rm -rf build` |
| Binario `.exe` viejo en `build/` | gtest sin salida ni XML, 0,06 s | `rm -rf build`; ⏳ sacar el `.exe` de `run_gtest.resolve_executable` |
| Proceso que muere sin salida | "no se pudo parsear el XML" | ⏳ mostrar el código de salida o la señal |
| Excepción en `SetUpTestSuite` | gtest sale con 1, los tests quedan *skipped* y `run_gtest` decía PASS | corregido el 2026-10-20 (código ≠ 0 = FAIL). En los tests, simular adentro del test y no en `SetUpTestSuite` |
| `create_module` con `<PREFIX>_GTEST_CPP` mal armado | `run_gtest` busca `/<module>_gtest.cpp` | corregido el 2026-10-20 en `module_utils.sh`; `refresh_module_vars <module>` regenera el `_vars.sh` |
| Plantilla de tb de `create_module` | sim da PASS con 16 ciclos que no prueban nada (pasó con la `fft2d`, 2026-10-21) | reemplazar `<module>_tb.cpp` antes de confiar en sim (en la `fft2d`, resuelto con `fft2d_tb_plot`) |
| `-O2` con `ap_fixed` | decenas de `-Wuninitialized` dentro de `ap_private` (falsos positivos) | `-Wno-uninitialized -Wno-maybe-uninitialized` solo en los builds `-O2`; con `-O0` no aparecen |
| Redondeo *half-up* al escalar y a la salida | sesgo +¼ LSB por etapa que escala y en la salida; en 2D se concentra (fila p = 0, píxel (0, 0) de la ida y vuelta) | `ROUND_CONV = 1` en `cast` y en el núcleo (2026-10-26); **default en el núcleo y la `fft2d` desde el 2026-11-15**; los casos `_halfup` lo conservan como referencia |
| Parche con archivos CRLF copiado del chat | se pierden los `\r`: el parche no aplica o mezcla fines de línea | entregar el parche como archivo y aplicarlo con `git apply <archivo>` |
| `cast` con NBF_OUT > NBF_IN | X | ⏳ `$fatal` |
| iverilog ignora `$fatal` en elaboración | el diseño inválido corre | doble `$fatal` (`generate` + `initial`) |
| N no potencia de 2 (o N < 4) | basura silenciosa | corta con `$fatal` / `static_assert` (desde `fft_n512`; antes exigía potencia de 4) |
| Máscara en hex en el JSON | Verilog no la entiende | decimal (784) |
| Texto después de `` `endif `` (p. ej. `` `endif __SYNTHESIS__ ``) | queda como código: Vivado da *syntax error near 'localparam'* en la línea siguiente | comentarlo: `` `endif // __SYNTHESIS__ `` (2026-10-17) |
| Default de `TW_DIR` **con comillas** en los wrappers | `STRINGIFY` produce `""twiddles/n1024""` y no elabora si el JSON no pasa `TW_DIR` | corregido el 2026-10-17 (default sin comillas) |
| Twiddles que no se cargan en síntesis | ROM en cero, multiplicadores optimizados y **menos de 16 DSP**: `MAX_DSP` pasa igual | revisar en el log de Vivado los avisos de `$readmem` y que el conteo sea **exactamente** 16 |
| Nombres dentro de la unidad | `BF2II` es la **primera** mariposa | bit 2u = primera, 2u + 1 = segunda |
| CRLF / LF mezclados | `patch` rechaza bloques | pasar los parches como archivo y respetar el fin de línea de cada uno. `.gitattributes` (2026-11-16) solo marca los binarios (`*.bin binary`, ...): **a propósito sin `eol=lf` ni `text=auto`**, que normalizarían los 141 archivos CRLF reescribiéndolos enteros |
| `run_plot_fft1d` con NBF del datapath | figuras 16 veces más chicas | corregido el 2026-10-17 (usa NBF_IN / NBF_OUT) |
| Modo `DOUBLE` con twiddles cuantizados | `double` fallaba 5/8 | corregido el 2026-10-12 (twiddles exactos) |
| Chequear el fin de línea con `file` | `file` no dice "CRLF" en los JSON; Python en modo texto convierte `\r\n` a `\n` sin avisar | contar con `grep -c $'\r$'` y abrir en binario (`'rb'`). Pasó con el JSON de regresión el 2026-10-17; se corrigió antes de entregar |
| GCC para aarch64 con `-O2`: contrae `a*b + c` a FMA aunque sea `-std=c++17` | el código del PS deja de ser bit a bit igual a NumPy (último bit) | `-ffp-contract=off` siempre (y sin `-ffast-math`); `ps_sense::build_buffer` lo chequea en ejecución y devuelve `fp_contract_on` (2026-11-15, H29) |

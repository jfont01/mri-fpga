# NOTAS: agente "chicos" (2026-11-20)

Clon: /home/jfont/mri-fpga-wip/chicos/repo, rama local `chicos`, commit base 9dc9d16
= 480a141 + d2_unfold + rtl_unfold + rtl_cordic_mag. El arbol es IDENTICO a ddf9b72
del repo de Julian (git diff ddf9b72 HEAD vacio): Julian ya commiteo los tres parches
(a8f5fac, 2c91b4b, ddf9b72) y despues e4974c3 (solo doc/). El parche aplica igual
sobre su HEAD (e4974c3) mientras no toque doc/ ni flists (las flists de cast estan
modificadas en su arbol: no incluir flists en el parche).

## Tarea 1 (unfold JSON)
- [x] casos small y g0 agregados al final del JSON (mismo formato: linea de
      parametros + vec_gen indentado, faults en linea aparte)
- [x] vectores: `--only small,g0` y despues regenerado TODO con --from-json: los 18
      viejos salen byte a byte iguales (git status limpio en vec/), los 2 nuevos
      reproducibles (diff -r contra la primera generacion vacio).
      small: 824 muestras, 414 salidas, err=6, s=[-5,58]; g0: 1536 muestras, 192 salidas.
- [x] recorte (tools/recorte_check.py, replica en Python de f_p_to_m del RTL, da
      0 diferencias contra m_ref en los 20 casos):
      * los 18 viejos tienen NB_T = NB_D = 6: hay D > NB_X (el comparador actua) pero
        un mutante SIN recorte (t = D mod 2^NB_T) da identico (0 diferencias): con
        t >= NB_X el corrimiento ya da 0 / -1+1 y las mascaras son todas 1/0. O sea:
        con NB_T >= NB_D el recorte es inobservable.
      * small: NB_P=18, NB_X=23, NB_T=5 < NB_D=6. 209 de 414 salidas con D > 23;
        el mutante difiere en 247 componentes de 70 posiciones (las de D en 32..55,
        donde D mod 32 < 23).
      * g0: NB_P=36, G=0 -> NB_X=36, NB_T=6 > NB_D=5: D <= 31 < 36, el recorte NUNCA
        actua (el comparador r_d2 > NB_X es constante 0). g0 NO ejercita el recorte;
        ejercita G = 0 (x = P, s = D >= 0, sin corrimiento a la izquierda; NB_D = 5 es
        el maximo con G = 0 por S_MAX <= 62) y t con NB_T > NB_D (d extendido con 0).
        Con G = 0 el recorte solo seria observable con NB_P <= 15 (NB_T < 5).

## Tarea 2 (guarda en cast.v)
### Condicion exacta (razonamiento cerrado)
D = LSB_TO_DROP = NBF_IN - NBF_OUT; W_SHIFT = NB_IN + 2; UPPER_W = max(W_SHIFT - NB_OUT, 0).
- shifted_word (W_SHIFT bits) no tiene X en ninguna de sus tres ramas (con D <= NB_IN,
  que vale con las precondiciones del modelo).
- gen_out_sat se elige con NEED_SAT && UPPER_W > 0; UPPER_W > 0 <=> NB_OUT <= NB_IN + 1:
  sus dos selects ([W_SHIFT-1:NB_OUT] y [NB_OUT-1:0]) estan en rango. Sin X.
- gen_out_no_sat en cualquier otro caso: shifted_word[NB_OUT-1:0] esta en rango <=>
  NB_OUT <= W_SHIFT = NB_IN + 2.
- Si NB_OUT > NB_IN + 2 -> UPPER_W = 0 -> SIEMPRE gen_out_no_sat (sea cual sea NEED_SAT)
  -> fuera de rango -> los bits o_word[NB_OUT-1 : NB_IN+2] son X para TODA entrada
  (IEEE 1800: leer fuera de rango da X en 4 estados).
=> X por este motivo <=> NB_OUT >= NB_IN + 3. No depende de NBF_IN, NBF_OUT, ROUND_MODE
   ni ROUND_CONV. La "aproximacion" NB_OUT > NB_IN + 2 es EXACTA.
- El "sin saturacion" de la trampa en 08_trampas.md sobra: con D >= 0, NB_OUT >= NB_IN + 3
  implica NBI_OUT - NBI_IN = (NB_OUT - NB_IN) + D >= 3 -> NEED_SAT = 0 solo. Con D < 0
  (trampa vieja) puede haber NEED_SAT = 1 y X igual (rama no_sat por UPPER_W = 0).
- Equivalente en bits enteros: NBI_OUT >= NBI_IN + D + 3.

### Barrido XSim (sweep/, fuera del parche) sobre el cast.v base
sweep_tb.sv: 3096 configuraciones en una sola elaboracion. NB_IN 2..6 exhaustivo
(todas las entradas), NBF_IN 0..NB_IN, NB_OUT 2..NB_IN+5, NBF_OUT 0..min(NB_OUT,
NBF_IN+2) (incluye D = -1, -2), modos (RM,RC) = (0,0),(1,0),(1,1); y NB_IN 12, 16, 20
con NB_OUT = NB_IN-2 ... NB_IN+4 (2000 entradas al azar), incluido el ejemplo 12,5->17,0.
Referencia entera (trunc / half-up / convergente + saturacion; D < 0: x << -D).
Resultado (analyze.py): 0 violaciones.
- X <=> NB_OUT > NB_IN + 2 en las 3096; con X, TODAS las entradas dan X y exactamente
  en los bits [NB_OUT-1 : NB_IN+2].
- D >= 0: 2106 configs; 756 con X (todas predichas), 1350 sin X y todas con valor
  correcto (incluye los bordes NB_OUT = NB_IN+1 y NB_IN+2).
- D < 0 (trampa vieja): 990 configs; 450 con X (justo las de NB_OUT > NB_IN+2), las
  540 restantes SIN X pero con valor equivocado (todas): gen_no_shift no corre a la
  izquierda. O sea la trampa vieja da basura silenciosa, no X (08_trampas dice X).
- XSim avisa VRFC 10-3705 una sola vez por linea fuente (no por instancia).

### Verificacion de unfold (con los 20 casos)
- run_gtest --keep-going: 20/20 PASS (19 x 11 tests + phantom 14 = 223 tests, 0 fail, 0 skip)
- run_regression_sim: 20/20 PASS
- update_flist + run_regression_vm --keep-going: 20/20 PASS, 0 errores (7 min 41 s)
- mutante del RTL (w_t2 = NB_T'(r_d2), sin recorte), vm de 5 casos: small FAIL (686
  errores), def_rand, dsweep, l1, g0 PASS -> solo small detecta el recorte. RTL restaurado.

### Guarda implementada
- cast.v: `if (NB_OUT > W_SHIFT) begin : gen_err_nb_out` con doble $fatal en el mismo
  generate del ifndef __SYNTHESIS__ que gen_err_conv; mensaje con valores.
- cast.hpp: static_assert(NB_OUT <= NB_IN + 2, ...) despues del de ROUND_CONV.
- xelab: 12,5->17,0 (RM=1,RC=1 y RM=0) y 12,5->15,0 cortan con VRFC 10-8279 y el mensaje;
  8,0->12,5 (D<0) tambien corta (por NB_OUT). Bordes 12,5->14,0 (conv y trunc),
  12,5->14,5, 20,19->22,19, 12,5->13,0: elaboran sin ningun WARNING y dan el valor bien.
- iverilog NO esta instalado en esta PC: el $fatal del initial no se pudo probar.
- g++ -fsyntax-only del modelo: 12,5->17,0 y 12,5->15,0 fallan con el static_assert
  nuevo; 12,5->14,0, 8,5->6,4 y 20,19->22,19 compilan (los 41 warnings son de
  ap_fixed.h / ap_private.h, iguales en todos).
- cast run_regression_sim: 14/14 (el modelo compila con el static_assert en los 14).

### Regresiones de los modulos que usan cast (con la guarda), logs en logs_guard/
- cast: sim 14/14, vm 14/14 (ver numeros en el log)
- cmul: sim 4/4, vm 4/4
- fft1d_r2sdf: sim 12/12; vm con el run_xsim.tcl del repo 0/12: "twiddle file not found
  .../rtl/twiddles/n64/tw_re.mem" (el tcl exige tw_re.mem y el R2SDF tiene tw_s<k>_re.mem).
  PRE-EXISTENTE: en la base (worktree ../base = 9dc9d16, sin la guarda) da lo mismo,
  12/12 FAIL con el mismo error (logs_base/). Con una copia local del tcl que acepta
  tw_s0_re.mem (tools/run_xsim_twany.tcl, RUN_XSIM_TCL=..., fuera del parche): 12/12 PASS,
  12 512 vectores, 0 errores. No se arreglo en el repo (no es de esta tarea).
- La auditoria con bind + $warning de elaboracion NO sirve: XSim imprime un solo
  $warning por linea fuente (dedup). Reemplazada por elab/elab_check.py: solo xelab, con
  --incdir para probar un cast.v alternativo (castold/ = la guarda nueva + un $fatal si
  NBF_OUT > NBF_IN). Probado que el include path toma la copia (castprobe: $fatal
  incondicional -> 4/4 cortan con OVERRIDE_OK).
- cmul_karatsuba: sim 4/4, vm 4/4 (16 384 vectores, 0 errores)
- fft1d_r2: sim 0/3 en el clon: el MODELO C++ no compila con g++ 13.3 ("declaration of
  enum fft1d_r2_model::state_t changes meaning of state_t" [-Wchanges-meaning], es error
  por defecto). Sin relacion con cast (fft1d_r2.hpp no incluye cast.hpp). Corriendo: base
  (para demostrar que falla igual) y el clon con EXTRA_CPPFLAGS=-fpermissive solo en la
  env (fuera del parche) para tener vectores y poder correr el vm con la guarda.
- Trampa vieja en el caso de la FFT: Q1.15 -> Q1.19 (16 -> 20 bits) cae en NB_OUT > NB_IN+2
  -> la guarda nueva CORTA (de ahi venia el "X" de la trampa vieja). Q1.15 -> Q1.17
  (16 -> 18) pasa la guarda y da 100 en vez de 400: basura silenciosa.
- fft1d_r2 base (worktree, sin guarda): sim 0/3 con el mismo error de compilacion ->
  PRE-EXISTENTE. Clon con EXTRA_CPPFLAGS=-fpermissive (solo env): sim 3/3; vm en curso.

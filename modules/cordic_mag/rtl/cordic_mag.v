`timescale 1ns/1ps
`default_nettype none
`ifndef CORDIC_MAG_V
`define CORDIC_MAG_V
// -----------------------------------------------------------------------------
// cordic_mag -- magnitud de M = (re, im) por CORDIC en modo vectoring (paso D2).
//
// Es el espejo, ciclo a ciclo y bit a bit, del modelo cpp/cordic_mag.hpp (la
// especificacion) y por lo tanto de model/mac_cordic_fxp.py (m_to_out). Diseno y
// decisiones: doc/diseno_unfold_cordic.md, secciones 4.3 y 4.5 (C1 a C8).
//
// ARITMETICA
//
//   etapa 0          x = |re|, y = |im|                     (primer cuadrante)
//   iteracion k      si y >= 0:  x <- x + (y >>> k),  y <- y - (x >> k)
//   (k = 0..N_IT-1)  si no:      x <- x - (y >>> k),  y <- y + (x >> k)
//                    (los dos lados con los x, y de antes; >>> aritmetico = floor)
//   salida           o_mag = sat_u(round(x / 2^G)), NB_OUT bits sin signo
//
// x termina en ~K_n |M| (K_12 = 1,64676). La ganancia no se corrige aca: el PS
// la pliega en U'' (1/K_n), asi que M ya llega dividido por K_n.
//
// SIGMA: es el bit de signo de y (sigma = + si y >= 0). Tiene que ser
// exactamente eso: con "y > 0" cambian los vectores sobre un eje, por ejemplo
// (1, 0) da x_n = 12 con y >= 0 y 1 con y > 0 (diseno D2, 4.3).
//
// ANCHOS (bitacora 4.13), con NB_W = NB_M + 1:
//
//   x   NB_W bits SIN signo. Llega a K sqrt(2) 2^(NB_M-1) < 2^NB_W, y nunca
//       baja: en cada iteracion se le suma |y >>> k|.
//   y   NB_W bits CON signo. Queda en [-2^(NB_M-1), 2^(NB_M-1)].
//
// Las sumas se hacen en NB_W bits (modulo 2^NB_W). Como los resultados
// verdaderos entran en esos rangos (el modelo los chequea en cada iteracion y
// tira std::range_error si no), los bits son exactos aunque se mezclen x sin
// signo e y con signo en la misma suma.
//
// REDONDEO DE SALIDA (G bits): la misma regla que cast.v y que
// mac_cordic_fxp.round_shift. Con m = floor(x / 2^G):
//
//   ROUND_CONV = 0 (half-up)        floor((x + 2^(G-1)) / 2^G)
//   ROUND_CONV = 1 (empates al par) m + rb & (sticky | lsb(m))
//                                   (rb: bit G-1 de x; sticky: OR de los de abajo)
//
// y despues saturacion a 2^NB_OUT - 1. No se instancia cast.v: es con signo
// (habria que agregar un bit de signo en 0 y sacarlo a la salida) y, si la
// salida tiene mas bits enteros que la entrada (NB_M chico), no extiende y
// devuelve X. Aca todo es sin signo y anda en todo el rango del modelo.
//
// PIPELINE: N_STAGES = N_IT + 2 registros en serie (|.|, una iteracion por
// registro, la salida). Cada etapa tiene su valid y sus datos se cargan solo con
// ese valid (clock enable): entre salidas, o_mag se queda con el ultimo valor.
// Importa porque el vm compara todos los ciclos, validos o no. (Estrictamente
// alcanza con el clock enable del registro 0, el unico que ve la basura de la
// entrada: cada registro siguiente es funcion pura del anterior, asi que si el
// anterior se queda quieto, recalcula el mismo valor. Un mutante sin el clock
// enable de las iteraciones pasa el vm completo. Se deja en todas las etapas
// para espejar el modelo, C4.)
//
// Un dato por ciclo y huecos permitidos en cualquier lugar. o_valid y o_mag
// salen directo de registros, asi que con la convencion de la FFT
// (LatencyMatchesFormula) la entrada que se pone antes del flanco c se ve
// despues del flanco c + LATENCY, con LATENCY = N_STAGES - 1 = N_IT + 1 (13 con
// los defaults).
//
// SIN RESET (decision C3): los registros arrancan en 0 con valor inicial (en la
// FPGA lo carga la configuracion, INIT de cada FF). El valid es un registro de
// desplazamiento que se carga en todos los ciclos, asi que con i_valid = 0 el
// pipeline se vacia solo en N_STAGES ciclos: no hace falta reset para salir de
// un estado cualquiera.
//
// COSTO (estimado, sin sintetizar; con los defaults): 2 negadores de NB_W bits
// para |.|, 2 sumadores/restadores de NB_W bits por iteracion (24), un
// incrementador y la saturacion; (N_IT + 1) * 2 * NB_W + N_STAGES + NB_OUT
// = 628 FF. El y de la ultima iteracion no lo lee nadie y sintesis lo saca.
// -----------------------------------------------------------------------------
module cordic_mag #(
  parameter int NB_OUT     = 16,                 // bits de la salida, sin signo
  parameter int G          = 5,                  // bits de guarda: M tiene G fraccionales mas que la salida
  parameter int N_IT       = 12,                 // iteraciones del CORDIC
  parameter int NB_M       = NB_OUT + 1 + G,     // ancho de M con signo (la misma regla que el modelo)
  parameter int ROUND_CONV = 0                   // 0: half-up (default de D2); 1: empates al par
)(
  input  wire                   i_clock,
  input  wire                   i_valid,
  input  wire signed [NB_M-1:0] i_m_re,
  input  wire signed [NB_M-1:0] i_m_im,
  output wire                   o_valid,
  output wire [NB_OUT-1:0]      o_mag
);

  localparam int NB_W     = NB_M + 1;            // x (sin signo) e y (con signo)
  localparam int N_STAGES = N_IT + 2;            // registros en serie
  localparam int LATENCY  = N_STAGES - 1;        // convencion de la FFT (solo documenta)

  // ------------------------------------------------- chequeo de parametros
  // Los mismos rangos que los static_assert del modelo. Doble $fatal: el de
  // elaboracion lo respetan Vivado y XSim; iverilog lo ignora en silencio, asi
  // que el del initial lo hace cortar en tiempo 0.
  `ifndef __SYNTHESIS__
    generate
      if (NB_OUT < 1 || NB_OUT > 32) begin : g_err_nb_out
        $fatal(1, "cordic_mag: NB_OUT=%0d tiene que estar en [1, 32]", NB_OUT);
        initial $fatal(1, "cordic_mag: NB_OUT=%0d tiene que estar en [1, 32]", NB_OUT);
      end
      if (G < 0 || G > 30) begin : g_err_g
        $fatal(1, "cordic_mag: G=%0d tiene que estar en [0, 30]", G);
        initial $fatal(1, "cordic_mag: G=%0d tiene que estar en [0, 30]", G);
      end
      if (N_IT < 1 || N_IT > 32) begin : g_err_n_it
        $fatal(1, "cordic_mag: N_IT=%0d tiene que estar en [1, 32]", N_IT);
        initial $fatal(1, "cordic_mag: N_IT=%0d tiene que estar en [1, 32]", N_IT);
      end
      if (NB_M < 2 || NB_W > 62) begin : g_err_nb_m
        $fatal(1, "cordic_mag: NB_M=%0d tiene que estar en [2, 61]", NB_M);
        initial $fatal(1, "cordic_mag: NB_M=%0d tiene que estar en [2, 61]", NB_M);
      end
      if (ROUND_CONV != 0 && ROUND_CONV != 1) begin : g_err_round_conv
        $fatal(1, "cordic_mag: ROUND_CONV=%0d tiene que ser 0 o 1", ROUND_CONV);
        initial $fatal(1, "cordic_mag: ROUND_CONV=%0d tiene que ser 0 o 1", ROUND_CONV);
      end
    endgenerate
  `endif // __SYNTHESIS__

  // Salida de cada registro: w_v[s] es el valid del registro s (0 .. N_STAGES-1);
  // w_x[s], w_y[s], los datos del registro s (0: |.|; k + 1: tras la iteracion k).
  wire                   w_v [0:N_STAGES-1];
  wire        [NB_W-1:0] w_x [0:N_IT];
  wire signed [NB_W-1:0] w_y [0:N_IT];

  // ------------------------------------------- registro 0: valor absoluto
  // Extension de signo a NB_W bits antes de negar: |-2^(NB_M-1)| = 2^(NB_M-1)
  // no entra en NB_M bits con signo, en NB_W si.
  wire signed [NB_W-1:0] w_re     = {i_m_re[NB_M-1], i_m_re};
  wire signed [NB_W-1:0] w_im     = {i_m_im[NB_M-1], i_m_im};
  wire        [NB_W-1:0] w_abs_re = w_re[NB_W-1] ? -w_re : w_re;
  wire        [NB_W-1:0] w_abs_im = w_im[NB_W-1] ? -w_im : w_im;

  reg                   r_abs_v = 1'b0;
  reg        [NB_W-1:0] r_abs_x = {NB_W{1'b0}};
  reg signed [NB_W-1:0] r_abs_y = {NB_W{1'b0}};

  always @(posedge i_clock) begin
    r_abs_v <= i_valid;
    if (i_valid) begin
      r_abs_x <= w_abs_re;
      r_abs_y <= w_abs_im;
    end
  end

  assign w_v[0] = r_abs_v;
  assign w_x[0] = r_abs_x;
  assign w_y[0] = r_abs_y;

  // ------------------------------- registros 1 .. N_IT: una iteracion cada uno
  // Corrimientos fijos por etapa (k es constante): son cableado, no hay
  // barrel shifter.
  genvar k;
  generate
    for (k = 0; k < N_IT; k = k + 1) begin : gen_it
      wire                   w_pos = ~w_y[k][NB_W-1];     // sigma: y >= 0
      wire        [NB_W-1:0] w_xs  = w_x[k] >> k;         // x >= 0: logico = floor
      wire signed [NB_W-1:0] w_ys  = w_y[k] >>> k;        // aritmetico = floor
      wire        [NB_W-1:0] w_xn  = w_pos ? (w_x[k] + w_ys) : (w_x[k] - w_ys);
      wire        [NB_W-1:0] w_yn  = w_pos ? (w_y[k] - w_xs) : (w_y[k] + w_xs);

      reg                   r_v = 1'b0;
      reg        [NB_W-1:0] r_x = {NB_W{1'b0}};
      reg signed [NB_W-1:0] r_y = {NB_W{1'b0}};

      always @(posedge i_clock) begin
        r_v <= w_v[k];
        if (w_v[k]) begin
          r_x <= w_xn;
          r_y <= w_yn;
        end
      end

      assign w_v[k+1] = r_v;
      assign w_x[k+1] = r_x;
      assign w_y[k+1] = r_y;
    end
  endgenerate

  // El y de la ultima iteracion no se usa (sintesis saca el registro).
  wire _unused_y = &{1'b0, w_y[N_IT]};

  // ------------------------ registro N_IT + 1: redondeo de G bits y saturacion
  // x se extiende con ceros a NB_R bits: lugar para el bit G - 1 (con NB_M
  // chico), para el acarreo del redondeo y para al menos un bit por encima de
  // NB_OUT, que es el que dice si satura. Con los defaults, NB_R = 24.
  localparam int NB_X = (NB_W > G) ? NB_W : G + 1;
  localparam int NB_R = ((NB_X > NB_OUT) ? NB_X : NB_OUT) + 1;

  localparam [NB_R-1:0]   ONE_R   = {{(NB_R-1){1'b0}}, 1'b1};
  localparam [NB_OUT-1:0] MAG_MAX = {NB_OUT{1'b1}};

  wire [NB_R-1:0] w_xr = {{(NB_R-NB_W){1'b0}}, w_x[N_IT]};
  wire [NB_R-1:0] w_rnd;

  generate
    if (G == 0) begin : gen_no_round
      assign w_rnd = w_xr;
    end
    else if (ROUND_CONV == 0) begin : gen_half_up
      // floor((x + 2^(G-1)) / 2^G); no desborda: x < 2^(NB_R-1).
      localparam [NB_R-1:0] HALF = ONE_R << (G - 1);
      assign w_rnd = (w_xr + HALF) >> G;
    end
    else begin : gen_conv
      // m + (r > h) + (r == h) lsb(m) = m + rb & (sticky | lsb(m)), como cast.v.
      localparam [NB_R-1:0] STICKY_MASK = (ONE_R << (G - 1)) - ONE_R;
      wire [NB_R-1:0] w_floor  = w_xr >> G;
      wire            w_rbit   = w_xr[G-1];
      wire            w_sticky = |(w_xr & STICKY_MASK);
      wire            w_carry  = w_rbit & (w_sticky | w_floor[0]);
      assign w_rnd = w_floor + {{(NB_R-1){1'b0}}, w_carry};
    end
  endgenerate

  wire              w_sat = |w_rnd[NB_R-1:NB_OUT];
  wire [NB_OUT-1:0] w_mag = w_sat ? MAG_MAX : w_rnd[NB_OUT-1:0];

  reg              r_out_v = 1'b0;
  reg [NB_OUT-1:0] r_mag   = {NB_OUT{1'b0}};

  always @(posedge i_clock) begin
    r_out_v <= w_v[N_IT];
    if (w_v[N_IT]) begin
      r_mag <= w_mag;
    end
  end

  assign w_v[N_IT+1] = r_out_v;

  assign o_valid = r_out_v;
  assign o_mag   = r_mag;

endmodule
`endif // CORDIC_MAG_V
`default_nettype wire

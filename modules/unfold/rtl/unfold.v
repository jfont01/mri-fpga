`timescale 1ns/1ps
`default_nettype none
`ifndef UNFOLD_V
`define UNFOLD_V
// -----------------------------------------------------------------------------
// unfold -- MAC de desplegado de SENSE (paso D2), un carril. Ciclo a ciclo y bit
// a bit igual al modelo cpp/unfold.hpp, que es la especificacion (diseno en
// doc/diseno_unfold_cordic.md, secciones 4.2 y 4.6, decision 13).
//
// Por cada posicion plegada entran las L bobinas de a una (una muestra valida
// por ciclo, con huecos en cualquier lugar, tambien adentro de una posicion).
// Cada muestra trae A_l y sus AF coeficientes U''_{a,l}. Para a = 0 ... AF-1:
//
//   P_a = sum_l U''_{a,l} * A_l           producto complejo exacto:
//                                         re = ur*ar - ui*ai, im = ur*ai + ui*ar
//   M_a = sat_NB_M(round(P_a / 2^s))      s = D - G, de -G a 2^NB_D - 1 - G
//
// round, con s > 0: half-up, floor((P + 2^(s-1)) / 2^s), como cast.v; con
// ROUND_CONV = 1, empates al par. Con s <= 0: corrimiento a la izquierda exacto
// (y despues la saturacion). Igual que mac_cordic_fxp.p_to_m.
//
// CIERRE DE UNA POSICION (decision 13). Un contador l cuenta las muestras
// validas de la posicion actual. La posicion se cierra con la muestra valida que
// trae i_last = 1 o con la muestra numero L - 1 (l == L - 1), lo que pase
// primero. Al cerrar sale o_valid con los AF valores de M, y o_err = 1 junto con
// ese o_valid si las dos condiciones no coincidieron; despues l vuelve a 0.
// o_err no es pegajoso (el pegajoso va en sense_ctrl). i_d se toma con la
// muestra que cierra la posicion. La primera muestra de una posicion carga el
// acumulador y las siguientes suman.
//
// PUERTOS. Empaquetados como el buffer de U'' ({im, re}):
//   i_a  [2*NB_A-1:0]        {im, re} de A_l, NB_A bits con signo cada uno
//   i_u  [AF*2*NB_U-1:0]     el elemento a en [a*2*NB_U +: 2*NB_U], {im, re}
//   i_d  [NB_D-1:0]          D sin signo (decision 11); s = D - G
//   o_m  [AF*2*NB_M-1:0]     el elemento a en [a*2*NB_M +: 2*NB_M], {im, re}
//
// PIPELINE: los mismos N_STAGES = 4 registros del modelo, con clock enable por
// valid (los datos de cada etapa se cargan solo con el valid de su entrada):
//
//   1. entrada      v, last, l, d, A, U[AF]; contador de bobinas     CE: i_valid
//   2. productos    4 por a, exactos; first, close y err de l, last  CE: v1
//   3. acumulador   P[AF] de NB_P bits                               CE: v2
//                   v3 = v2 && close; err y el corrimiento
//                   decodificado de d                                CE: v2 && close
//   4. salida       M[AF] = sat(round(P / 2^s)), o_err               CE: v3
//
// Entre salidas validas o_m y o_err quedan quietos (el vm compara todos los
// ciclos). Sin reset: todo arranca en 0 (valor inicial de los registros), como
// el modelo; los valid tambien, asi que el primer o_valid es de verdad.
//
// LATENCY, con la convencion de la FFT (gtest LatencyMatchesFormula): si la
// muestra que cierra la posicion se pone antes del flanco numero c, o_valid se
// ve despues del flanco numero c + LATENCY. LATENCY = N_STAGES - 1 = 3.
//
// DSP48E2 (UG579 v1.11). Las etapas 1 a 3 son el patron multiplicar-acumular
// del DSP con todos sus registros: A/B (etapa 1), M (etapa 2) y P (etapa 3); el
// cap. 2 (p. 18) pide esos tres para la frecuencia maxima. Los productos son de
// NB_U x NB_A bits (16 x 16 con los defaults: entran en el 27 x 18 con signo),
// uno por DSP: 4 por a, 4*AF por carril (8 con AF = 2). El acumulador esta
// escrito como el MACC con carga sincronica (old = first ? 0 : P; P <= old +
// productos), para que Vivado lo pueda meter en el post-sumador: el producto ii
// (o ir) baja por la cascada PCIN al DSP de rr (o ri), que suma W = P (o 0 con
// first, OPMODE dinamico) + M + PCIN. Es el uso que el cap. 1 (p. 10) le da al
// multiplexor W ("another accumulator feedback path to reduce the size of the
// complex multiply-accumulate"). Si Vivado no lo absorbe, el acumulador queda en
// fabrica (2*AF sumadores de 3 entradas de NB_P bits) con el mismo
// comportamiento. NO SE SINTETIZO: mirar el "DSP Report" del log de sintesis.
//
// ETAPA 4: REDONDEO CON CORRIMIENTO VARIABLE. Con x = P * 2^G (NB_X = NB_P + G
// bits, cableado) y t = D = s + G en [0, 2^NB_D - 1]:
//
//   round(P / 2^s) = q + c,  q = floor(x / 2^t)
//     c = rb                       half-up      rb = bit t-1 de x (0 si t = 0)
//     c = rb & (sticky | q[0])     convergente  sticky = OR de los bits t-2 ... 0
//
// (half-up: floor((P + 2^(s-1)) / 2^s) = floor(P / 2^s) + bit s-1 de P; la regla
// convergente es la de cast.v). Con s <= 0 (t <= G) los t bits que se descartan
// son ceros de x, asi que c = 0 y q = P * 2^(-s): el corrimiento a la izquierda
// exacto del modelo sale del mismo camino. Con t >= NB_X el resultado es 0 en los
// dos modos (|P| <= 2^(NB_P-1)), asi que t se recorta a NB_X. Un solo
// corrimiento aritmetico a la derecha de {x, 0} da q y rb juntos (rb queda en el
// bit 0). La saturacion:
//
//   fits = los bits de x desde NB_M - 1 + t para arriba son todos iguales al
//          signo (q entra en NB_M bits)
//   M    = !fits ? (signo ? MIN : MAX) : (q + c desborda ? MAX : q + c)
//
// (q + c solo desborda hacia arriba, con q = MAX y c = 1; si q no entra, el caso
// q = MIN - 1 con c = 1 da MIN, igual que saturar). Chequeado contra la regla del
// modelo con enteros de Python: 636 650 combinaciones de P, D y modo en 12
// configuraciones de anchos, 0 diferencias.
//
// t recortado y su termometro th[k] = (k < t), del que salen las mascaras de
// sticky y de fits, dependen solo de D: se calculan en la etapa 3 con el d de la
// etapa 2 y se registran con el mismo clock enable que d3 en el modelo, asi que
// la etapa 4 no decodifica nada.
//
// CAMINO CRITICO ESPERADO (no sintetizado), etapa 4: registro P (PREG del DSP ->
// fabrica) -> corrimiento de NB_Y = 42 bits por t de 6 bits (3 niveles de LUT6
// como mux 4:1) -> c (1 nivel) -> q + c de NB_M + 1 = 23 bits (3 CARRY8) ->
// desborde y mux de saturacion (1-2 niveles) -> registro M. En paralelo, fits
// (XOR, AND con la mascara y OR de 41 bits, ~3 niveles) y sticky (~3 niveles).
// Unos 5-6 niveles de LUT mas la cadena de carry: se estima que entra en 5 ns,
// pero es el camino a mirar. Si no cierra, partir la etapa 4 en dos
// (corrimiento | redondeo y saturacion) sube LATENCY a 4 y obliga a cambiar el
// modelo (no se hizo).
//
// PRECONDICIONES (las del modelo; doble $fatal): L en [1, 256], AF en [1, 16],
// NB_A y NB_U en [2, 31], G en [0, 30], NB_M en [2, 61], ROUND_CONV 0 o 1,
// NB_D en [1, 7], NB_P + G <= 63 y S_MAX = 2^NB_D - 1 - G <= 62.
// -----------------------------------------------------------------------------
module unfold #(
  parameter int L          = 8,              // bobinas por posicion plegada
  parameter int AF         = 2,              // factor de aceleracion: pixeles por posicion
  parameter int NB_A       = 16,             // A_l (salida de la fft2d), con signo
  parameter int NB_U       = 16,             // U'' (buffer del PS), con signo
  parameter int G          = 5,              // bits de guarda de M
  parameter int NB_M       = 17 + G,         // M con signo (misma regla que el modelo y el Python)
  parameter int ROUND_CONV = 0,              // 0: half-up (default de D2); 1: empates al par
  parameter int NB_D       = 6               // i_d, sin signo
)(
  input  wire                    i_clock,
  input  wire                    i_valid,
  input  wire                    i_last,     // ultima bobina de la posicion
  input  wire [2*NB_A-1:0]       i_a,        // {im, re}
  input  wire [AF*2*NB_U-1:0]    i_u,        // AF x {im, re}
  input  wire [NB_D-1:0]         i_d,
  output wire                    o_valid,
  output wire                    o_err,      // la posicion no cerro con i_last y l == L - 1 juntos
  output wire [AF*2*NB_M-1:0]    o_m         // AF x {im, re}
);

  // ------------------------------------------------------- locales del modelo
  localparam int NB_L     = $clog2(L);                   // ceil(log2 L), 0 con L = 1
  localparam int NB_P     = NB_A + NB_U + 1 + NB_L;      // acumulador (H31)
  localparam int S_MIN    = -G;                          // s con D = 0
  localparam int S_MAX    = (1 << NB_D) - 1 - G;         // s con D = 2^NB_D - 1
  localparam int N_STAGES = 4;                           // registros en serie
  localparam int LATENCY  = N_STAGES - 1;                // convencion de la FFT

  // ------------------------------------------------------- locales del RTL
  localparam int NB_CNT = (NB_L > 0) ? NB_L : 1;         // contador de bobinas
  localparam int NB_AU  = NB_A + NB_U;                   // un producto real, exacto
  localparam int NB_X   = NB_P + G;                      // x = P * 2^G
  localparam int NB_Y   = ((NB_X > NB_M + 1) ? NB_X : (NB_M + 1)) + 1;   // {x extendido, 0}
  localparam int NB_T   = $clog2(NB_X + 1);              // t recortado, en [0, NB_X]

  localparam [NB_M-1:0] M_MAX = {1'b0, {(NB_M-1){1'b1}}};
  localparam [NB_M-1:0] M_MIN = {1'b1, {(NB_M-1){1'b0}}};

  // --------------------------------------------------- chequeo de parametros
  // Los static_assert del modelo. Doble $fatal: el de elaboracion lo respetan
  // Vivado y XSim; iverilog lo ignora en silencio, asi que el del initial lo hace
  // cortar en tiempo 0.
  `ifndef __SYNTHESIS__
  generate
    if (L < 1 || L > 256) begin : g_err_l
      $fatal(1, "unfold: L=%0d tiene que estar en [1, 256]", L);
      initial $fatal(1, "unfold: L=%0d tiene que estar en [1, 256]", L);
    end
    if (AF < 1 || AF > 16) begin : g_err_af
      $fatal(1, "unfold: AF=%0d tiene que estar en [1, 16]", AF);
      initial $fatal(1, "unfold: AF=%0d tiene que estar en [1, 16]", AF);
    end
    if (NB_A < 2 || NB_A > 31) begin : g_err_nb_a
      $fatal(1, "unfold: NB_A=%0d tiene que estar en [2, 31]", NB_A);
      initial $fatal(1, "unfold: NB_A=%0d tiene que estar en [2, 31]", NB_A);
    end
    if (NB_U < 2 || NB_U > 31) begin : g_err_nb_u
      $fatal(1, "unfold: NB_U=%0d tiene que estar en [2, 31]", NB_U);
      initial $fatal(1, "unfold: NB_U=%0d tiene que estar en [2, 31]", NB_U);
    end
    if (G < 0 || G > 30) begin : g_err_g
      $fatal(1, "unfold: G=%0d tiene que estar en [0, 30]", G);
      initial $fatal(1, "unfold: G=%0d tiene que estar en [0, 30]", G);
    end
    if (NB_M < 2 || NB_M > 61) begin : g_err_nb_m
      $fatal(1, "unfold: NB_M=%0d tiene que estar en [2, 61]", NB_M);
      initial $fatal(1, "unfold: NB_M=%0d tiene que estar en [2, 61]", NB_M);
    end
    if (ROUND_CONV != 0 && ROUND_CONV != 1) begin : g_err_conv
      $fatal(1, "unfold: ROUND_CONV=%0d tiene que ser 0 o 1", ROUND_CONV);
      initial $fatal(1, "unfold: ROUND_CONV=%0d tiene que ser 0 o 1", ROUND_CONV);
    end
    if (NB_D < 1 || NB_D > 7) begin : g_err_nb_d
      $fatal(1, "unfold: NB_D=%0d tiene que estar en [1, 7]", NB_D);
      initial $fatal(1, "unfold: NB_D=%0d tiene que estar en [1, 7]", NB_D);
    end
    if (NB_P + G > 63) begin : g_err_nb_p
      $fatal(1, "unfold: NB_P + G = %0d > 63 (P * 2^G no entra en 64 bits): bajar NB_A, NB_U, L o G", NB_P + G);
      initial $fatal(1, "unfold: NB_P + G = %0d > 63 (P * 2^G no entra en 64 bits): bajar NB_A, NB_U, L o G", NB_P + G);
    end
    if (S_MAX > 62) begin : g_err_s_max
      $fatal(1, "unfold: s maximo = 2^NB_D - 1 - G = %0d > 62 (con G = 0, NB_D <= 5)", S_MAX);
      initial $fatal(1, "unfold: s maximo = 2^NB_D - 1 - G = %0d > 62 (con G = 0, NB_D <= 5)", S_MAX);
    end
  endgenerate
  `endif // __SYNTHESIS__

  // =========================================================================
  // Etapa 1: registro de entrada y contador de bobinas (CE: i_valid)
  // =========================================================================
  wire signed [NB_A-1:0] w_a_re = i_a[0    +: NB_A];
  wire signed [NB_A-1:0] w_a_im = i_a[NB_A +: NB_A];

  reg  [NB_CNT-1:0]      r_l     = '0;     // l de la proxima muestra valida (cnt_r del modelo)
  reg                    r_v1    = 1'b0;
  reg                    r_last1 = 1'b0;
  reg  [NB_CNT-1:0]      r_l1    = '0;
  reg  [NB_D-1:0]        r_d1    = '0;
  reg  signed [NB_A-1:0] r_a1_re = '0;
  reg  signed [NB_A-1:0] r_a1_im = '0;

  // La posicion se cierra con i_last o con la bobina L - 1.
  wire w_l_end = (r_l == NB_CNT'(L - 1));

  always @(posedge i_clock) begin
    r_v1 <= i_valid;
    if (i_valid) begin
      r_last1 <= i_last;
      r_l1    <= r_l;
      r_d1    <= i_d;
      r_a1_re <= w_a_re;
      r_a1_im <= w_a_im;
      r_l     <= (i_last || w_l_end) ? '0 : r_l + 1'b1;
    end
  end

  // =========================================================================
  // Etapa 2: banderas de la posicion (CE: v1); los productos van por carril
  // =========================================================================
  reg               r_v2     = 1'b0;
  reg               r_first2 = 1'b0;
  reg               r_close2 = 1'b0;
  reg               r_err2   = 1'b0;
  reg  [NB_D-1:0]   r_d2     = '0;

  wire w_l1_end = (r_l1 == NB_CNT'(L - 1));

  always @(posedge i_clock) begin
    r_v2 <= r_v1;
    if (r_v1) begin
      r_first2 <= (r_l1 == '0);
      r_close2 <= r_last1 | w_l1_end;
      r_err2   <= r_last1 ^ w_l1_end;
      r_d2     <= r_d1;
    end
  end

  // =========================================================================
  // Etapa 3: acumulador (por carril, CE: v2); err y el corrimiento decodificado
  // (CE: v2 && close, como err3 y d3 del modelo)
  // =========================================================================
  wire w_close = r_v2 & r_close2;

  // t = min(D, NB_X) y su termometro th[k] = (k < t), k = 0 ... NB_X.
  wire [NB_T-1:0] w_t2 = (r_d2 > NB_X) ? NB_T'(NB_X) : NB_T'(r_d2);
  wire [NB_X:0]   w_th2;

  genvar k;
  generate
    for (k = 0; k <= NB_X; k = k + 1) begin : g_th
      assign w_th2[k] = (w_t2 > k);
    end
  endgenerate

  reg               r_v3   = 1'b0;
  reg               r_err3 = 1'b0;
  reg  [NB_T-1:0]   r_t3   = '0;
  reg  [NB_X:0]     r_th3  = '0;

  always @(posedge i_clock) begin
    r_v3 <= w_close;
    if (w_close) begin
      r_err3 <= r_err2;
      r_t3   <= w_t2;
      r_th3  <= w_th2;
    end
  end

  // Mascaras de la etapa 4 (cableado e inversores sobre r_th3):
  //   sticky: bits de x por debajo del de redondeo, k < t - 1  ->  th[k + 1]
  //   fits:   bits de x desde NB_M - 1 + t                     ->  !th[k - NB_M + 1]
  wire [NB_X-1:0] w_mask_st = r_th3[NB_X:1];
  wire [NB_X-1:0] w_mask_fit;

  generate
    for (k = 0; k < NB_X; k = k + 1) begin : g_mask_fit
      if (k >= NB_M - 1) begin : g_hi
        assign w_mask_fit[k] = ~r_th3[k - NB_M + 1];
      end
      else begin : g_lo
        assign w_mask_fit[k] = 1'b0;
      end
    end
  endgenerate

  // =========================================================================
  // Etapa 4: M = sat_NB_M(round(P / 2^s)) (por carril, CE: v3) y o_err
  // =========================================================================
  reg r_v4   = 1'b0;
  reg r_err4 = 1'b0;

  always @(posedge i_clock) begin
    r_v4 <= r_v3;
    if (r_v3) begin
      r_err4 <= r_err3;
    end
  end

  // Redondeo y saturacion de una componente (ver la cabecera). t, mask_st y
  // mask_fit salen de los registros de la etapa 3.
  function automatic [NB_M-1:0] f_p_to_m(
    input [NB_P-1:0] p,
    input [NB_T-1:0] t,
    input [NB_X-1:0] mask_st,
    input [NB_X-1:0] mask_fit
  );
    reg signed [NB_X-1:0] x;
    reg signed [NB_Y-1:0] y;
    reg signed [NB_Y-1:0] r;
    reg                   rb, sticky, c, sgn, fits;
    reg        [NB_M:0]   sum;
    begin
      x      = NB_X'($signed(p)) <<< G;               // P * 2^G, exacto
      y      = $signed({(NB_Y-1)'(x), 1'b0});         // {x, 0}: rb queda en el bit 0
      r      = y >>> t;                               // aritmetico
      rb     = r[0];                                  // bit t-1 de x (0 con t = 0)
      sticky = |(x & mask_st);
      c      = (ROUND_CONV != 0) ? (rb & (sticky | r[1])) : rb;   // r[1] = q[0]
      sgn    = x[NB_X-1];
      fits   = ~|((x ^ {NB_X{sgn}}) & mask_fit);
      sum    = r[NB_M+1:1] + c;                       // q + c en NB_M + 1 bits
      if (!fits)
        f_p_to_m = sgn ? M_MIN : M_MAX;
      else if (sum[NB_M] != sum[NB_M-1])
        f_p_to_m = M_MAX;                             // q = MAX y c = 1
      else
        f_p_to_m = sum[NB_M-1:0];
    end
  endfunction

  // =========================================================================
  // Carriles: un pixel verdadero por a (etapas 1 a 4 de los datos)
  // =========================================================================
  genvar a;
  generate
    for (a = 0; a < AF; a = a + 1) begin : g_lane
      wire signed [NB_U-1:0] w_u_re = i_u[a*2*NB_U        +: NB_U];
      wire signed [NB_U-1:0] w_u_im = i_u[a*2*NB_U + NB_U +: NB_U];

      // 1: entrada (registros A/B del DSP)
      reg signed [NB_U-1:0]  r_u1_re = '0;
      reg signed [NB_U-1:0]  r_u1_im = '0;

      always @(posedge i_clock) begin
        if (i_valid) begin
          r_u1_re <= w_u_re;
          r_u1_im <= w_u_im;
        end
      end

      // 2: productos exactos (registro M del DSP)
      reg signed [NB_AU-1:0] r_rr2 = '0;    // ur * ar
      reg signed [NB_AU-1:0] r_ii2 = '0;    // ui * ai
      reg signed [NB_AU-1:0] r_ri2 = '0;    // ur * ai
      reg signed [NB_AU-1:0] r_ir2 = '0;    // ui * ar

      always @(posedge i_clock) begin
        if (r_v1) begin
          r_rr2 <= r_u1_re * r_a1_re;
          r_ii2 <= r_u1_im * r_a1_im;
          r_ri2 <= r_u1_re * r_a1_im;
          r_ir2 <= r_u1_im * r_a1_re;
        end
      end

      // 3: acumulador (registro P del DSP). La primera bobina carga, las
      // siguientes suman. NB_P bits alcanzan (H31): el modelo lo chequea.
      reg signed [NB_P-1:0]  r_p_re3 = '0;
      reg signed [NB_P-1:0]  r_p_im3 = '0;

      wire signed [NB_P-1:0] w_p_re_old = r_first2 ? '0 : r_p_re3;
      wire signed [NB_P-1:0] w_p_im_old = r_first2 ? '0 : r_p_im3;

      always @(posedge i_clock) begin
        if (r_v2) begin
          r_p_re3 <= w_p_re_old + NB_P'(r_rr2) - NB_P'(r_ii2);
          r_p_im3 <= w_p_im_old + NB_P'(r_ri2) + NB_P'(r_ir2);
        end
      end

      // 4: redondeo y saturacion
      reg [NB_M-1:0] r_m_re4 = '0;
      reg [NB_M-1:0] r_m_im4 = '0;

      always @(posedge i_clock) begin
        if (r_v3) begin
          r_m_re4 <= f_p_to_m(r_p_re3, r_t3, w_mask_st, w_mask_fit);
          r_m_im4 <= f_p_to_m(r_p_im3, r_t3, w_mask_st, w_mask_fit);
        end
      end

      assign o_m[a*2*NB_M        +: NB_M] = r_m_re4;
      assign o_m[a*2*NB_M + NB_M +: NB_M] = r_m_im4;
    end
  endgenerate

  assign o_valid = r_v4;
  assign o_err   = r_err4;

endmodule
`endif
`default_nettype wire

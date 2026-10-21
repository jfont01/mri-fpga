`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_V
`define FFT1D_R22SDF_V
`include "fft1d_r22sdf_unit.v"
`include "fft1d_r22sdf_tail.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf -- FFT radix-2^2 Single-path Delay Feedback, N puntos.
//
// Cadena de floor(log2(N)/2) unidades radix-2^2. Cada unidad procesa un nivel
// radix-4 (dos etapas radix-2: BF2II con el twiddle trivial -j, y BF2I) mas el
// multiplicador que cierra el par. Si log2(N) es par, la ultima unidad (M = 4)
// no multiplica:
//
//   x(n) -> [unidad M=N] -> [unidad M=N/4] -> ... -> [unidad M=4] -> X(k)
//
// Si log2(N) es impar (p. ej. 512), las unidades llegan hasta M = 8, todas con
// multiplicador, y se agrega una etapa radix-2 final de span 1 sin twiddle
// (fft1d_r22sdf_tail):
//
//   x(n) -> [unidad M=N] -> ... -> [unidad M=8] -> [tail] -> X(k)
//
//   multiplicadores : log4(N) - 1 si log2(N) es par; floor(log2(N)/2) si es impar
//                     (N = 1024 y N = 512: 4 multiplicadores, 16 DSP)
//   memoria         : N - 1           (identica al R2SDF)
//   orden de salida : BIT-REVERSED
//   escala          : 1/2^s, con s = cantidad de bits en 1 de SCALE_MASK[LOG2N-1:0]
//                     (bit k = 1: la etapa radix-2 k divide por 2; bit k = 0: no divide
//                     y el cast satura). Con la mascara por defecto, s = log2(N): 1/N.
//
// PRECONDICION: N potencia de 2, N >= 4. El bit log2(N)-1 de SCALE_MASK es, con
// log2(N) impar, el de la etapa final.
//
// SEGMENTACION: a diferencia de una cadena puramente combinacional, cada unidad
// registra entre sus dos mariposas y a la salida del multiplicador. El camino
// critico queda acotado a una mariposa o a un multiplicador complejo, no a la
// cadena entera. El precio es latencia (varios ciclos por unidad), no throughput:
// se sigue aceptando una muestra por ciclo.
//
// Con CMUL_PIPE = 1 cada multiplicador registra ademas la suma completa antes de
// su cast (PREG del DSP48E2, ver cmul.v): el camino DSP -> cast queda partido en
// dos y cada unidad con multiplicador suma un ciclo. Latencia hasta el primer
// o_valid, contando desde la primera muestra (U = floor(log2(N)/2)):
//
//   log2(N) par   : LATENCY = N + (3 + CMUL_PIPE) * (U - 1)
//                 = 1036 (CMUL_PIPE = 0) o 1040 (CMUL_PIPE = 1) para N = 1024
//   log2(N) impar : LATENCY = N + (3 + CMUL_PIPE) * U - 1
//                 = 523  (CMUL_PIPE = 0) o 527  (CMUL_PIPE = 1) para N = 512
//
// Los valores de salida no cambian; solo llegan mas tarde.
//
// Default CMUL_PIPE = 1 desde el 2026-10-18: sin el registro, el camino DSP ->
// cast dejaba 50 ps de margen a 200 MHz despues del ruteo; con el registro,
// 0,903 ns (xck26-2LV, Vivado 2024.2). CMUL_PIPE = 0 queda como referencia.
//
// FORMATOS: el puerto de entrada, el datapath interno, el puerto de salida y los
// twiddles tienen cada uno su formato:
//
//   entrada  Q(NB_IN,  NBF_IN)  --ensancha-->  datapath Q(NB, NBF)  --cast-->  salida Q(NB_OUT, NBF_OUT)
//
//   * Entrada -> datapath: SOLO ensancha (NBF >= NBF_IN y no pierde bits enteros).
//     Es cableado: ceros en los LSB y extension de signo. No pierde informacion.
//   * Datapath -> salida: cast con redondeo y saturacion. Es el UNICO redondeo que
//     no es interno a una etapa: el pipeline calcula con NB bits y recien aca se
//     reduce al ancho que va a la memoria. Combinacional: no agrega latencia.
//   * Si no se fijan los bits fraccionales de un formato, hereda la cantidad de
//     bits enteros del datapath. Por defecto los tres formatos son el del datapath.
//
// TWIDDLES: todas las unidades leen la MISMA tabla de N entradas (W_N^k,
// k = 0..N-1); cada una la direcciona distinto segun su resolucion M. Por eso
// hace falta un unico par de archivos, tw_re.mem y tw_im.mem, en TW_DIR.
//
// INVERSA (INVERSE = 1): el mismo nucleo calcula la IDFT sin normalizar,
//
//   sum_k X[k] e^{+j 2 pi k n / N}  =  swap( FFT( swap(X) ) ),   swap(a + jb) = b + ja
//
// (por 2^-s, igual que la directa). El swap es cruzar real e imaginaria a la
// entrada y a la salida: no niega nada, no agrega logica ni latencia, y el error
// numerico es exactamente el de la FFT directa sobre la entrada cruzada.
//
// Los twiddles tienen formato propio, Q(NB_TW, NBF_TW), independiente del de los
// datos: los .mem tienen que tener valores de NB_TW bits. Por defecto NB_TW = NB y
// NBF_TW = NBF. En el DSP48E2 (27 x 18), twiddle de hasta 18 bits y dato de hasta 27
// mantienen un DSP por producto real.
// -----------------------------------------------------------------------------
module fft1d_r22sdf #(
  parameter int    N   = 64,
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_DIR = "twiddles",
  parameter bit    TW_FROM_FILE = 1'b1,
  parameter logic [31:0] SCALE_MASK = 32'hFFFF_FFFF,
  parameter int    NB_TW   = NB,                 // formato de los twiddles
  parameter int    NBF_TW  = NB_TW  - (NB - NBF),
  parameter int    NB_IN   = NB,                 // formato del puerto de entrada
  parameter int    NBF_IN  = NB_IN  - (NB - NBF),
  parameter int    NB_OUT  = NB,                 // formato del puerto de salida
  parameter int    NBF_OUT = NB_OUT - (NB - NBF),
  parameter bit    CMUL_PIPE = 1'b1,             // 1: registro antes del cast de cada cmul
  parameter bit    INVERSE   = 1'b0              // 1: IDFT (swap real/imag a la entrada y a la salida)
)(
  input  wire                 i_clock,
  input  wire                 i_rst,          // sincrono, activo alto
  input  wire                 i_valid,
  input  wire [2*NB_IN-1:0]   i_cplx_sample,
  output wire                 o_valid,
  output wire                 o_last,
  output wire [2*NB_OUT-1:0]  o_cplx_sample
);

  localparam int LOG2N   = $clog2(N);
  localparam int N_UNITS  = LOG2N / 2;            // floor(log2(N)/2) unidades radix-2^2
  localparam bit HAS_TAIL = (LOG2N % 2) != 0;     // log2(N) impar: etapa radix-2 final

  // ------------------------------------------------- chequeo de parametros
  // Doble $fatal: el de elaboracion lo respetan Vivado y XSim; iverilog lo ignora
  // en silencio, asi que el del initial lo hace cortar en tiempo 0.
  `ifndef __SYNTHESIS__
    generate
      if (N != (1 << LOG2N) || LOG2N < 2) begin : g_err_n
        $fatal(1, "fft1d_r22sdf: N=%0d tiene que ser potencia de 2 y >= 4", N);
        initial $fatal(1, "fft1d_r22sdf: N=%0d tiene que ser potencia de 2 y >= 4", N);
      end
      if (NBF < NBF_IN || (NB - NBF) < (NB_IN - NBF_IN)) begin : g_err_in
        $fatal(1, "fft1d_r22sdf: la entrada Q(%0d,%0d) no entra sin perdida en el datapath Q(%0d,%0d)", NB_IN, NBF_IN, NB, NBF);
        initial $fatal(1, "fft1d_r22sdf: la entrada Q(%0d,%0d) no entra sin perdida en el datapath Q(%0d,%0d)", NB_IN, NBF_IN, NB, NBF);
      end
      if (NBF < NBF_OUT) begin : g_err_out
        $fatal(1, "fft1d_r22sdf: la salida (NBF_OUT=%0d) no puede tener mas bits fraccionales que el datapath (NBF=%0d)", NBF_OUT, NBF);
        initial $fatal(1, "fft1d_r22sdf: la salida (NBF_OUT=%0d) no puede tener mas bits fraccionales que el datapath (NBF=%0d)", NBF_OUT, NBF);
      end
    endgenerate
  `endif // __SYNTHESIS__

  // ------------------------------------------- entrada: Q(NB_IN) -> Q(NB)
  // Solo ensancha: el desplazamiento agrega NBF-NBF_IN ceros abajo y la
  // extension de signo (contexto de NB bits) completa arriba. No usa cast,
  // porque cast no admite agregar bits fraccionales.
  localparam int SH_IN = NBF - NBF_IN;

  // INVERSE: swap a la entrada (cableado; INVERSE es constante).
  wire signed [NB_IN-1:0] w_port_re = i_cplx_sample[0     +: NB_IN];
  wire signed [NB_IN-1:0] w_port_im = i_cplx_sample[NB_IN +: NB_IN];
  wire signed [NB_IN-1:0] w_in_re   = INVERSE ? w_port_im : w_port_re;
  wire signed [NB_IN-1:0] w_in_im   = INVERSE ? w_port_re : w_port_im;
  wire signed [NB-1:0]    w_din_re = NB'(w_in_re) <<< SH_IN;
  wire signed [NB-1:0]    w_din_im = NB'(w_in_im) <<< SH_IN;

  wire [2*NB-1:0] w_data  [0:N_UNITS];
  wire            w_valid [0:N_UNITS];

  assign w_data[0]  = {w_din_im, w_din_re};
  assign w_valid[0] = i_valid;

  genvar u;
  generate
    for (u = 0; u < N_UNITS; u = u + 1) begin : gen_unit
      localparam int M_U = N >> (2 * u);    // N, N/4, N/16, ... , 4

      fft1d_r22sdf_unit #(
        .N            (N),
        .M            (M_U),
        .NB           (NB),
        .NBF          (NBF),
        .SCALE_BF2II  (SCALE_MASK[2*u]),
        .SCALE_BF2I   (SCALE_MASK[2*u+1]),
        .NB_TW        (NB_TW),
        .NBF_TW       (NBF_TW),
        .TW_RE_FILE   ({TW_DIR, "/tw_re.mem"}),
        .TW_IM_FILE   ({TW_DIR, "/tw_im.mem"}),
        .TW_FROM_FILE (TW_FROM_FILE),
        .CMUL_PIPE    (CMUL_PIPE)
      ) u_unit (
        .i_clock (i_clock),
        .i_rst   (i_rst),
        .i_valid (w_valid[u]),
        .i_din   (w_data[u]),
        .o_valid (w_valid[u+1]),
        .o_dout  (w_data[u+1])
      );
    end
  endgenerate

  // ------------------------------------- etapa radix-2 final (log2 N impar)
  wire [2*NB-1:0] w_fin_data;
  wire            w_fin_valid;

  generate
    if (HAS_TAIL) begin : gen_tail
      fft1d_r22sdf_tail #(
        .N       (N),
        .NB      (NB),
        .NBF     (NBF),
        .SCALE   (SCALE_MASK[LOG2N-1])
      ) u_tail (
        .i_clock (i_clock),
        .i_rst   (i_rst),
        .i_valid (w_valid[N_UNITS]),
        .i_din   (w_data[N_UNITS]),
        .o_valid (w_fin_valid),
        .o_dout  (w_fin_data)
      );
    end
    else begin : gen_no_tail
      assign w_fin_data  = w_data[N_UNITS];
      assign w_fin_valid = w_valid[N_UNITS];
    end
  endgenerate

  // ------------------------------------------------------------ o_last
  reg [LOG2N-1:0] r_out_count;
  always @(posedge i_clock) begin
    if (i_rst)             r_out_count <= {LOG2N{1'b0}};
    else if (w_fin_valid)  r_out_count <= r_out_count + 1'b1;
  end

  /*
   * Gating de salida (igual que el R2SDF): con o_valid=0 lo que sale del
   * pipeline sin cebar es basura (X en simulacion, porque las lineas de retardo
   * arrancan indefinidas). Se fuerza a 0 para espejar el modelo C++ y para que
   * el vm no compare muestras invalidas. Es solo un mux; la muestra valida no
   * se altera.
   */
  // -------------------------------------------- salida: Q(NB) -> Q(NB_OUT)
  wire signed [NB_OUT-1:0] w_out_re, w_out_im;

  cast #(.NB_IN(NB), .NBF_IN(NBF), .NB_OUT(NB_OUT), .NBF_OUT(NBF_OUT), .ROUND_MODE(1'b1))
    u_out_re (.i_word($signed(w_fin_data[0  +: NB])), .o_word(w_out_re));
  cast #(.NB_IN(NB), .NBF_IN(NBF), .NB_OUT(NB_OUT), .NBF_OUT(NBF_OUT), .ROUND_MODE(1'b1))
    u_out_im (.i_word($signed(w_fin_data[NB +: NB])), .o_word(w_out_im));

  assign o_valid       = w_fin_valid;
  // INVERSE: swap a la salida. El cast es el mismo para las dos componentes, asi
  // que cruzar antes o despues de el da lo mismo.
  wire [2*NB_OUT-1:0] w_out_cplx = INVERSE ? {w_out_re, w_out_im} : {w_out_im, w_out_re};

  assign o_cplx_sample = w_fin_valid ? w_out_cplx : {(2*NB_OUT){1'b0}};
  assign o_last        = w_fin_valid && (r_out_count == LOG2N'(N - 1));

endmodule
`endif
`default_nettype wire
`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_V
`define FFT1D_R22SDF_V
`include "fft1d_r22sdf_unit.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf -- FFT radix-2^2 Single-path Delay Feedback, N puntos.
//
// Cadena de log4(N) unidades radix-2^2. Cada unidad procesa un nivel radix-4
// (dos etapas radix-2: BF2II con el twiddle trivial -j, y BF2I) mas el
// multiplicador que cierra el par. La ultima unidad no multiplica.
//
//   x(n) -> [unidad M=N] -> [unidad M=N/4] -> ... -> [unidad M=4] -> X(k)
//
//   multiplicadores : log4(N) - 1     (contra log2(N)-1 del R2SDF)
//   memoria         : N - 1           (identica al R2SDF)
//   orden de salida : BIT-REVERSED
//   escala          : 1/N (cada mariposa divide por 2)
//
// PRECONDICION: N potencia de 4 (log2(N) par). Para N = 2^impar (p.ej. 512) hace
// falta una etapa radix-2 suelta adicional, no cubierta por este modulo.
//
// SEGMENTACION: a diferencia de una cadena puramente combinacional, cada unidad
// registra entre sus dos mariposas y a la salida del multiplicador. El camino
// critico queda acotado a una mariposa o a un multiplicador complejo, no a la
// cadena entera. El precio es latencia (varios ciclos por unidad), no throughput:
// se sigue aceptando una muestra por ciclo.
//
// TWIDDLES: todas las unidades leen la MISMA tabla de N entradas (W_N^k,
// k = 0..N-1); cada una la direcciona distinto segun su resolucion M. Por eso
// hace falta un unico par de archivos, tw_re.mem y tw_im.mem, en TW_DIR.
// -----------------------------------------------------------------------------
module fft1d_r22sdf #(
  parameter int    N   = 64,
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_DIR = "twiddles",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                 i_clock,
  input  wire                 i_rst,          // sincrono, activo alto
  input  wire                 i_valid,
  input  wire [2*NB-1:0]      i_cplx_sample,
  output wire                 o_valid,
  output wire                 o_last,
  output wire [2*NB-1:0]      o_cplx_sample
);

  localparam int LOG2N   = $clog2(N);
  if (LOG2N % 2 == 1) begin $fatal(1, "LOG2N odd, not supported yet"); end
  localparam int N_UNITS = LOG2N / 2;      // log4(N)

  wire [2*NB-1:0] w_data  [0:N_UNITS];
  wire            w_valid [0:N_UNITS];

  assign w_data[0]  = i_cplx_sample;
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
        .TW_RE_FILE   ({TW_DIR, "/tw_re.mem"}),
        .TW_IM_FILE   ({TW_DIR, "/tw_im.mem"}),
        .TW_FROM_FILE (TW_FROM_FILE)
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

  // ------------------------------------------------------------ o_last
  reg [LOG2N-1:0] r_out_count;
  always @(posedge i_clock) begin
    if (i_rst)                  r_out_count <= {LOG2N{1'b0}};
    else if (w_valid[N_UNITS])  r_out_count <= r_out_count + 1'b1;
  end

  /*
   * Gating de salida (igual que el R2SDF): con o_valid=0 lo que sale del
   * pipeline sin cebar es basura (X en simulacion, porque las lineas de retardo
   * arrancan indefinidas). Se fuerza a 0 para espejar el modelo C++ y para que
   * el vm no compare muestras invalidas. Es solo un mux; la muestra valida no
   * se altera.
   */
  assign o_valid       = w_valid[N_UNITS];
  assign o_cplx_sample = w_valid[N_UNITS] ? w_data[N_UNITS] : {(2*NB){1'b0}};
  assign o_last        = w_valid[N_UNITS] && (r_out_count == LOG2N'(N - 1));

endmodule
`endif
`default_nettype wire
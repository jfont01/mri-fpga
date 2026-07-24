`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R2SDF_V
`define FFT1D_R2SDF_V
`include "fft1d_r2sdf_stage.v"
// -----------------------------------------------------------------------------
// fft1d_r2sdf -- FFT radix-2 Single-path Delay Feedback, N puntos.
//
// Arquitectura STREAMING: una muestra por ciclo, entrada y salida continuas.
// Espeja bit a bit el golden C++ fft1d_r2sdf_model.
//
//   etapas          : log2(N), encadenadas
//   linea de la et.k: L_k = 2^(log2N-1-k)   -> N/2, N/4, ... , 1  (total N-1)
//   multiplicadores : log2(N) - 2  (las dos ultimas etapas son triviales)
//   latencia        : N-1 ciclos
//   orden de salida : BIT-REVERSED
//
// CONTROL
// -------
// Un unico contador libre de log2(N) bits alimenta todas las etapas:
//
//   etapa k:  ctrl = r_count[LOG2N-1-k]          (mariposa si/no)
//             addr = r_count[LOG2N-2-k : 0]      (puntero de linea Y de ROM)
//
// o sea: el bit mas significativo maneja la primera etapa y el menos
// significativo la ultima. No hay FSM.
//
// PRECONDICION: i_valid no puede tener huecos dentro de un frame (una muestra
// por ciclo). Si la fuente no lo sostiene, poner un FIFO antes de este bloque.
//
// PENDIENTE: la cadena es combinacional de punta a punta (log2(N) mariposas y
// multiplicadores en serie). Habra que insertar registros entre etapas para
// cerrar timing; eso suma log2(N) ciclos de latencia y hay que replicarlo en el
// modelo C++ para mantener la bit-exactitud del vm.
// -----------------------------------------------------------------------------
module fft1d_r2sdf #(
  parameter int    N   = 512,
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

  localparam int LOG2N = $clog2(N);

  // ---------------------------------------------------------------- contador
  reg [LOG2N-1:0] r_count;
  reg             r_primed;

  localparam [LOG2N-1:0] COUNT_MAX = LOG2N'(N - 1);

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_count  <= {LOG2N{1'b0}};
      r_primed <= 1'b0;
    end
    else begin
      r_count <= r_count + 1'b1;
      if (r_count == COUNT_MAX) begin
        r_primed <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------- cadena de etapas
  wire [2*NB-1:0] w_chain [0:LOG2N];

  assign w_chain[0] = i_valid ? i_cplx_sample : {(2*NB){1'b0}};

  genvar k;
  generate
    for (k = 0; k < LOG2N; k = k + 1) begin : gen_stage
      localparam int DEPTH    = 1 << (LOG2N - 1 - k);
      localparam int ADDR_W   = (DEPTH > 1) ? (LOG2N - 1 - k) : 1;
      // twiddles todos triviales cuando la linea mide 2 o menos
      localparam bit HAS_MULT = (DEPTH > 2);

      wire [ADDR_W-1:0] w_addr = (DEPTH > 1) ? r_count[ADDR_W-1:0] : {ADDR_W{1'b0}};
      wire              w_ctrl = r_count[LOG2N-1-k];

      fft1d_r2sdf_stage #(
        .NB           (NB),
        .NBF          (NBF),
        .DEPTH        (DEPTH),
        .ADDR_W       (ADDR_W),
        .HAS_MULT     (HAS_MULT),
        .TW_RE_FILE   ({TW_DIR, "/tw_s", 8'(48 + k), "_re.mem"}),
        .TW_IM_FILE   ({TW_DIR, "/tw_s", 8'(48 + k), "_im.mem"}),
        .TW_FROM_FILE (TW_FROM_FILE)
      ) u_stage (
        .i_clock (i_clock),
        .i_addr  (w_addr),
        .i_ctrl  (w_ctrl),
        .i_data  (w_chain[k]),
        .o_data  (w_chain[k+1])
      );
    end
  endgenerate

  // ------------------------------------------------------ registro de salida
  // Corta el camino combinacional y hace que la salida sea funcion solo del
  // estado, igual que en el modelo C++.
  reg [2*NB-1:0] r_out;
  reg            r_out_valid;
  reg            r_out_last;

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_out       <= {(2*NB){1'b0}};
      r_out_valid <= 1'b0;
      r_out_last  <= 1'b0;
    end
    else begin
      r_out       <= w_chain[LOG2N];
      r_out_valid <= r_primed || (r_count == COUNT_MAX);
      r_out_last  <= (r_primed || (r_count == COUNT_MAX)) && (r_count == COUNT_MAX);
    end
  end

  assign o_cplx_sample = r_out;
  assign o_valid       = r_out_valid;
  assign o_last        = r_out_last;

endmodule
`endif
`default_nettype wire
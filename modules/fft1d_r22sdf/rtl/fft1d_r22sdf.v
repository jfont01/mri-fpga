`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_V
`define FFT1D_R22SDF_V
`include "fft1d_r22sdf_stage.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf -- FFT radix-2^2 Single-path Delay Feedback, N puntos.
//
// Misma familia SDF que fft1d_r2sdf (una muestra por ciclo, salida bit-reversed,
// linea total N-1), pero con la organizacion radix-2^2: las etapas van de a
// PARES (BF2I, BF2II) y hay UN multiplicador por par, salvo el ultimo. Esto da
//
//   multiplicadores : log4(N) - 1     (contra log2(N)-1 del R2SDF)
//   memoria         : N - 1           (identica al R2SDF)
//   sumadores       : 2*log2(N)       (identica al R2SDF)
//   orden de salida : BIT-REVERSED
//
// La ventaja del radix-2^2 es puramente en multiplicadores/ROM de twiddles.
//
// PRECONDICION: N debe ser potencia de 4 (log2(N) par) para que las etapas se
// emparejen. Para N potencia de 2 no potencia de 4 (p.ej. 512), hace falta una
// etapa radix-2 suelta adicional -- no cubierta por este modulo (ver la tesis).
//
// CONTROL (un unico contador libre de log2(N) bits, estilo R2SDF; el mapeo de
// -j y twiddle se deriva de la referencia nanamake/r22sdf, MIT):
//
//   etapa k:  ctrl = r_count[LOG2N-1-k]                 (mariposa si/no)
//             addr = r_count[LOG2N-2-k : 0]             (linea y ROM)
//   BF2II (k impar): mj = (r_count[LOG2N-1-(k-1) : LOG2N-2-(k-1)] == 2'b11)
//             o sea, los dos bits altos del par valen 11 -> aplica -j.
//
// PENDIENTE (igual que el R2SDF): la cadena es combinacional de punta a punta;
// habra que registrar entre etapas para cerrar timing, replicando la latencia
// en el modelo C++ cuando se haga la verificacion funcional.
// -----------------------------------------------------------------------------
module fft1d_r22sdf #(
  parameter int    N   = 256,
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

      // Par al que pertenece la etapa: k=0,1 -> par 0 ; k=2,3 -> par 1 ; ...
      // BF2I = etapa par del par (k par) ; BF2II = etapa impar (k impar).
      localparam int STAGE_KIND = k[0];        // 0=BF2I, 1=BF2II

      // El multiplicador va tras cada BF2II (k impar), salvo el ULTIMO par.
      // Ultimo par: k == LOG2N-1 (la etapa BF2II final). No multiplica.
      localparam bit IS_LAST_PAIR = (k >= LOG2N - 2);
      localparam bit HAS_MULT     = (STAGE_KIND == 1) && !IS_LAST_PAIR && (DEPTH > 1);

      wire [ADDR_W-1:0] w_addr = (DEPTH > 1) ? r_count[ADDR_W-1:0] : {ADDR_W{1'b0}};
      wire              w_ctrl = r_count[LOG2N-1-k];

      // -j habilitado en BF2II cuando los dos bits altos del par valen 11.
      // Los dos bits altos del par (etapas k-1, k) son r_count[LOG2N-1-(k-1)]
      // y r_count[LOG2N-1-k].
      wire w_mj;
      if (STAGE_KIND == 1) begin : gen_mj_ctrl
        wire hi = r_count[LOG2N - 1 - (k-1)];
        wire lo = r_count[LOG2N - 1 - k];
        assign w_mj = hi & lo;               // == 2'b11
      end
      else begin : gen_no_mj_ctrl
        assign w_mj = 1'b0;
      end

      fft1d_r22sdf_stage #(
        .NB           (NB),
        .NBF          (NBF),
        .DEPTH        (DEPTH),
        .ADDR_W       (ADDR_W),
        .STAGE_KIND   (STAGE_KIND),
        .HAS_MULT     (HAS_MULT),
        .TW_RE_FILE   ({TW_DIR, "/tw_s", 8'(48 + k), "_re.mem"}),
        .TW_IM_FILE   ({TW_DIR, "/tw_s", 8'(48 + k), "_im.mem"}),
        .TW_FROM_FILE (TW_FROM_FILE)
      ) u_stage (
        .i_clock (i_clock),
        .i_addr  (w_addr),
        .i_ctrl  (w_ctrl),
        .i_mj    (w_mj),
        .i_data  (w_chain[k]),
        .o_data  (w_chain[k+1])
      );
    end
  endgenerate

  // ------------------------------------------------------ registro de salida
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

  assign o_cplx_sample = r_out_valid ? r_out : {(2*NB){1'b0}};
  assign o_valid       = r_out_valid;
  assign o_last        = r_out_last;

endmodule
`endif
`default_nettype wire
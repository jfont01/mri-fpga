`timescale 1ns/1ps
`ifndef FFT1D_R2_V
`define FFT1D_R2_V
`include "fft1d_r2_FSM.v"
`include "fft1d_r2_core.v"
// -----------------------------------------------------------------------------
// fft1d_r2 -- FFT radix-2 Cooley-Tukey de N puntos, arquitectura iterativa
//             basada en memoria (1 mariposa/ciclo). Espeja bit a bit el golden
//             C++ fft1d_r2_model.
//
// Este top solo cablea los dos planos:
//   fft1d_r2_ctrl : FSM, contadores y generacion de direcciones
//   fft1d_r2_dp   : memorias, ROM de twiddles, cmul y mariposa
//
// Protocolo:
//   i_rst=1 -> vuelve a LOADING, contadores en 0.
//   LOADING : con i_valid=1, N ciclos, una muestra/ciclo en orden NATURAL;
//             se guarda en direccion bit-reversed (permutacion "gratis").
//   COMPUTE : LOG2N etapas x N/2 mariposas, 1/ciclo. Cada rama escala *1/2
//             (round-half-up) -> factor total 1/N.
//   OUTPUT  : N ciclos, un resultado/ciclo en orden NATURAL. o_last en la ult.
// -----------------------------------------------------------------------------
module fft1d_r2 #(
  parameter int    N   = 512,
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_RE_FILE = "fft1d_r2_tw_re.mem",
  parameter        TW_IM_FILE = "fft1d_r2_tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                 i_clock,
  input  wire                 i_rst,      // sincrono, activo alto
  input  wire                 i_valid,
  input  wire signed [NB-1:0] i_re,
  input  wire signed [NB-1:0] i_im,
  output wire                 o_valid,
  output wire                 o_last,
  output wire signed [NB-1:0] o_re,
  output wire signed [NB-1:0] o_im
);

  localparam int LOG2N = $clog2(N);

  wire                 w_load_en;
  wire [LOG2N-1:0]     w_load_addr;
  wire                 w_btfly_en;
  wire [LOG2N-1:0]     w_idx_a;
  wire [LOG2N-1:0]     w_idx_b;
  wire [LOG2N-2:0]     w_idx_tw;
  wire [LOG2N-1:0]     w_out_addr;
  wire                 w_valid;

  fft1d_r2_FSM #(
    .N            (N)
  ) u_fsm (
    .i_clock      (i_clock),
    .i_rst        (i_rst),
    .i_valid      (i_valid),
    .o_load_en    (w_load_en),
    .o_load_addr  (w_load_addr),
    .o_btfly_en   (w_btfly_en),
    .o_idx_a      (w_idx_a),
    .o_idx_b      (w_idx_b),
    .o_idx_tw     (w_idx_tw),
    .o_out_addr   (w_out_addr),
    .o_valid      (w_valid),
    .o_last       (o_last)
  );

  fft1d_r2_core #(
    .N            (N),
    .NB           (NB),
    .NBF          (NBF),
    .TW_RE_FILE   (TW_RE_FILE),
    .TW_IM_FILE   (TW_IM_FILE),
    .TW_FROM_FILE (TW_FROM_FILE)
  ) u_core (
    .i_clock      (i_clock),
    .i_re         (i_re),
    .i_im         (i_im),
    .i_load_en    (w_load_en),
    .i_load_addr  (w_load_addr),
    .i_btfly_en   (w_btfly_en),
    .i_idx_a       (w_idx_a),
    .i_idx_b       (w_idx_b),
    .i_idx_tw     (w_idx_tw),
    .i_out_addr   (w_out_addr),
    .i_out_en     (w_valid),
    .o_re         (o_re),
    .o_im         (o_im)
  );

  assign o_valid = w_valid;

endmodule
`endif
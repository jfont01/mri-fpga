`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_TAIL_V
`define FFT1D_R22SDF_TAIL_V
`include "btfly_bf2i.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf_tail -- etapa radix-2 final para N = 2^impar (p. ej. 512).
//
// Con log2(N) impar, las unidades radix-2^2 (M = N, N/4, ..., 8) cubren
// log2(N) - 1 etapas. Falta una etapa radix-2 DIF de span 1 y SIN twiddle (es
// la ultima: despues de ella no hay nada que rotar):
//
//   x(n) -> [unidad M=N] -> ... -> [unidad M=8] -> [tail] -> X(k) bit-reversed
//
// Para N = 512 son 4 unidades, todas con multiplicador (la regla LOG_M != 2 de
// la unidad lo da solo), mas esta etapa. Verificado a nivel de arreglos, con el
// mismo direccionamiento de twiddles del RTL, para N = 8 ... 2048
// (model/fft2d_theory.py, check b).
//
// ESTRUCTURA
// ----------
// La misma mariposa BF2I de las unidades (btfly_bf2i, sin -j) con DEPTH = 1:
//
//   muestra par   (bf = 0): entra a la linea; sale lo que la linea guardaba
//   muestra impar (bf = 1): sale (d + x)[/2]; la linea guarda (d - x)[/2]
//
// asi que la salida es  x0+x1, x0-x1, x2+x3, x2-x3, ...  (span 1).
//
// CONTROL
// -------
// El mismo patron que la primera etapa de cada unidad: un contador que arranca
// con el valid que llega (di_count) y un habilitador de salida de N ciclos que
// arranca cuando di_count == DEPTH - 1. Diferencia: con DEPTH = 1 la condicion
// es di_count == 0, que tambien vale en reposo; por eso aca se exige i_valid.
// (En las unidades DEPTH >= 2 y di_count == DEPTH - 1 solo se alcanza con datos.)
//
// Un registro a la salida, como las demas etapas: latencia 2 (1 de la linea +
// 1 del registro). El camino critico de la etapa es una suma y su cast.
//
// SCALE: bit log2(N) - 1 de SCALE_MASK (la ultima etapa en orden de tiempo).
// -----------------------------------------------------------------------------
module fft1d_r22sdf_tail #(
  parameter int N     = 512,
  parameter int NB    = 16,
  parameter int NBF   = 15,
  parameter bit SCALE = 1'b1,
  parameter bit ROUND_CONV = 1'b0   // 1: redondeo convergente al escalar
)(
  input  wire                i_clock,
  input  wire                i_rst,       // sincrono, activo alto
  input  wire                i_valid,
  input  wire [2*NB-1:0]     i_din,
  output wire                o_valid,
  output wire [2*NB-1:0]     o_dout
);

  localparam int LOG_N = $clog2(N);

  // ------------------------------------------------------------ control
  reg [LOG_N-1:0] r_di_count;
  always @(posedge i_clock) begin
    if (i_rst) r_di_count <= {LOG_N{1'b0}};
    else       r_di_count <= i_valid ? (r_di_count + 1'b1) : {LOG_N{1'b0}};
  end

  wire w_bf = r_di_count[0];

  reg             r_sp_en;
  reg [LOG_N-1:0] r_count;

  wire w_start = i_valid && (r_di_count == {LOG_N{1'b0}});
  wire w_end   = (r_count == LOG_N'((1 << LOG_N) - 1));

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_sp_en <= 1'b0;
      r_count <= {LOG_N{1'b0}};
    end
    else begin
      r_sp_en <= w_start ? 1'b1 : (w_end ? 1'b0 : r_sp_en);
      r_count <= r_sp_en ? (r_count + 1'b1) : {LOG_N{1'b0}};
    end
  end

  // ----------------------------------------------------------- mariposa
  wire [2*NB-1:0] w_sp;

  btfly_bf2i #(
    .NB      (NB),
    .NBF     (NBF),
    .DEPTH   (1),
    .ADDR_W  (1),
    .SCALE   (SCALE),
    .ROUND_CONV (ROUND_CONV)
  ) u_bf2 (
    .i_clock (i_clock),
    .i_rst   (i_rst),
    .i_bf    (w_bf),
    .i_din   (i_din),
    .o_sp    (w_sp)
  );

  // ------------------------------------------------- registro de salida
  reg [2*NB-1:0] r_do;
  reg            r_do_en;
  always @(posedge i_clock) begin
    if (i_rst) begin
      r_do    <= {(2*NB){1'b0}};
      r_do_en <= 1'b0;
    end
    else begin
      r_do    <= w_sp;
      r_do_en <= r_sp_en;
    end
  end

  assign o_dout  = r_do;
  assign o_valid = r_do_en;

endmodule
`endif
`default_nettype wire
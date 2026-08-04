`timescale 1ns/1ps
`default_nettype none
`ifndef BTFLY_BF2II_V
`define BTFLY_BF2II_V
`include "cast.v"
`include "delay_line.v"
// -----------------------------------------------------------------------------
// btfly_bf2ii -- mariposa BF2II del algoritmo radix-2^2 (He & Torkelson, Fig 5-ii).
//
// Identica a btfly_bf2i (mariposa radix-2 SDF + linea de retardo) pero agrega el
// CONMUTADOR que realiza la multiplicacion trivial por -j:
//
//     -j * (a + jb) = b - ja      ->      (re, im) -> (im, -re)
//
// que es solo intercambio real/imaginario y una inversion de signo: no consume
// multiplicador. Es la operacion que da su ventaja al radix-2^2, ya que la mitad
// de las rotaciones del radix-2 se vuelven triviales.
//
//   i_bf = 0, i_mj = 0 : o_sp = d            (carga normal)
//   i_bf = 0, i_mj = 1 : o_sp = -j * d       (carga con twiddle trivial)
//   i_bf = 1           : o_sp = (d + x)/2    (mariposa; la linea guarda (d-x)/2)
//
// UBICACION DEL CONMUTADOR (importante)
// -------------------------------------
// En el grafo de flujo (Fig. 3 del paper) el factor -j aparece ENTRE las dos
// mariposas del par: es el twiddle trivial que multiplica una rama antes de
// entrar a la segunda mariposa. En la realizacion SDF ese dato es justamente el
// que sale de la linea de retardo de la PRIMERA etapa del par durante su fase de
// carga. Por eso este modulo -- el que lleva el conmutador -- ocupa la primera
// posicion del par (linea de M/2), y btfly_bf2i la segunda (linea de M/4).
// Aplicar el -j aqui equivale exactamente a aplicarlo a la entrada de la segunda
// mariposa, que es como lo dibuja la Fig. 5-(ii).
//
// El control i_mj se deriva de dos bits del contador de la etapa, como indica el
// paper ("requires two bit control signal from the synchronizing counter").
//
// Complejos empaquetados {imag, real}.
// -----------------------------------------------------------------------------
module btfly_bf2ii #(
  parameter int NB     = 16,
  parameter int NBF    = 15,
  parameter int DEPTH  = 32,
  parameter int ADDR_W = 5
)(
  input  wire                i_clock,
  input  wire                i_rst,     // sincrono, activo alto
  input  wire                i_bf,      // 0 = carga, 1 = mariposa
  input  wire                i_mj,      // 1 = aplicar -j (solo con i_bf = 0)
  input  wire [2*NB-1:0]     i_din,
  output wire [2*NB-1:0]     o_sp
);

  localparam int RE_LSB  = 0;
  localparam int IM_LSB  = NB;
  localparam int NB_SUM  = NB + 1;
  localparam int NBF_SUM = NBF + 1;

  // ------------------------------------------- direccion de la linea
  reg [ADDR_W-1:0] r_addr;
  always @(posedge i_clock) begin
    if (i_rst) r_addr <= {ADDR_W{1'b0}};
    else       r_addr <= r_addr + 1'b1;
  end

  // ------------------------------------------------- linea de retardo
  wire [2*NB-1:0] w_d;
  wire [2*NB-1:0] w_db_di;

  delay_line #(
    .DEPTH   (DEPTH),
    .WIDTH   (2*NB),
    .ADDR_W  (ADDR_W)
  ) u_delay (
    .i_clock (i_clock),
    .i_addr  (r_addr),
    .i_data  (w_db_di),
    .o_data  (w_d)
  );

  // ------------------------------------------------ mariposa radix-2
  wire signed [NB-1:0] x_re = $signed(i_din[RE_LSB +: NB]);
  wire signed [NB-1:0] x_im = $signed(i_din[IM_LSB +: NB]);
  wire signed [NB-1:0] d_re = $signed(w_d[RE_LSB +: NB]);
  wire signed [NB-1:0] d_im = $signed(w_d[IM_LSB +: NB]);

  wire signed [NB_SUM-1:0] sum_re = d_re + x_re;
  wire signed [NB_SUM-1:0] sum_im = d_im + x_im;
  wire signed [NB_SUM-1:0] dif_re = d_re - x_re;
  wire signed [NB_SUM-1:0] dif_im = d_im - x_im;

  wire signed [NB-1:0] sum_rnd_re, sum_rnd_im, dif_rnd_re, dif_rnd_im;

  cast #(.NB_IN(NB_SUM), .NBF_IN(NBF_SUM), .NB_OUT(NB), .NBF_OUT(NBF), .ROUND_MODE(1'b1))
    u_rnd_sum_re (.i_word(sum_re), .o_word(sum_rnd_re));
  cast #(.NB_IN(NB_SUM), .NBF_IN(NBF_SUM), .NB_OUT(NB), .NBF_OUT(NBF), .ROUND_MODE(1'b1))
    u_rnd_sum_im (.i_word(sum_im), .o_word(sum_rnd_im));
  cast #(.NB_IN(NB_SUM), .NBF_IN(NBF_SUM), .NB_OUT(NB), .NBF_OUT(NBF), .ROUND_MODE(1'b1))
    u_rnd_dif_re (.i_word(dif_re), .o_word(dif_rnd_re));
  cast #(.NB_IN(NB_SUM), .NBF_IN(NBF_SUM), .NB_OUT(NB), .NBF_OUT(NBF), .ROUND_MODE(1'b1))
    u_rnd_dif_im (.i_word(dif_im), .o_word(dif_rnd_im));

  wire [2*NB-1:0] w_y0 = {sum_rnd_im, sum_rnd_re};
  wire [2*NB-1:0] w_y1 = {dif_rnd_im, dif_rnd_re};

  // ---------------------------------------- conmutador -j (twiddle trivial)
  // -j * (d_re + j d_im) = d_im - j d_re
  // La negacion satura: -(-2^(NB-1)) no entra en NB bits.
  wire signed [NB-1:0] neg_d_re = (d_re == {1'b1, {(NB-1){1'b0}}})
                                ? {1'b0, {(NB-1){1'b1}}}
                                : -d_re;

  wire [2*NB-1:0] w_d_mj = {neg_d_re, d_im};   // {imag, real} = {-d_re, d_im}

  // --------------------------------------------- realimentacion SDF
  assign w_db_di = i_bf ? w_y1 : i_din;
  assign o_sp    = i_bf ? w_y0 : (i_mj ? w_d_mj : w_d);

endmodule
`endif
`default_nettype wire
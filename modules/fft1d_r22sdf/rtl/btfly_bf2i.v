`timescale 1ns/1ps
`default_nettype none
`ifndef BTFLY_BF2I_V
`define BTFLY_BF2I_V
`include "cast.v"
`include "delay_line.v"
// -----------------------------------------------------------------------------
// btfly_bf2i -- mariposa BF2I del algoritmo radix-2^2 (He & Torkelson, Fig 5-i).
//
// Es una etapa SDF (single-path delay feedback) completa: mariposa radix-2 +
// linea de retardo + los dos multiplexores de realimentacion.
//
//   i_bf = 0 (carga)    : la muestra del stream entra a la linea,
//                         y sale por o_sp lo que la linea guardaba.
//   i_bf = 1 (mariposa) : o_sp = (d + x)/2   y la linea guarda (d - x)/2
//
// donde x = i_din (stream) y d = salida de la linea de retardo.
//
// El x1/2 no es un divisor: la suma d±x vive en Q((NBI+1).NBF) con NB+1 bits;
// reinterpretada como Q(NBI.(NBF+1)) su valor YA es la mitad, asi que un cast
// que baja un bit fraccional hace el escalado y el redondeo de una vez. Es el
// mismo criterio que btfly_sdf.
//
// La direccion de la linea la genera un contador libre interno: escribir y leer
// en la misma direccion hace que lo leido sea lo escrito DEPTH ciclos antes, o
// sea un shift register de DEPTH posiciones.
//
// Complejos empaquetados {imag, real}.
// -----------------------------------------------------------------------------
module btfly_bf2i #(
  parameter int NB     = 16,
  parameter int NBF    = 15,
  parameter int DEPTH  = 32,
  parameter int ADDR_W = 5
)(
  input  wire                i_clock,
  input  wire                i_rst,     // sincrono, activo alto
  input  wire                i_bf,      // 0 = carga, 1 = mariposa
  input  wire [2*NB-1:0]     i_din,
  output wire [2*NB-1:0]     o_sp       // single-path: hacia la etapa siguiente
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
  wire [2*NB-1:0] w_d;      // dato que sale de la linea
  wire [2*NB-1:0] w_db_di;  // dato que entra a la linea

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

  wire [2*NB-1:0] w_y0 = {sum_rnd_im, sum_rnd_re};   // suma  -> sale
  wire [2*NB-1:0] w_y1 = {dif_rnd_im, dif_rnd_re};   // resta -> a la linea

  // --------------------------------------------- realimentacion SDF
  assign w_db_di = i_bf ? w_y1  : i_din;
  assign o_sp    = i_bf ? w_y0  : w_d;

endmodule
`endif
`default_nettype wire
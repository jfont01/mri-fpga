`timescale 1ns/1ps
`default_nettype none
`ifndef BTFLY_SDF_V
`define BTFLY_SDF_V
`include "cast.v"
// -----------------------------------------------------------------------------
// btfly_sdf -- mariposa de una etapa SDF, combinacional.
//
//   i_ctrl = 0 (carga)    : o_y = d            o_z = x
//   i_ctrl = 1 (mariposa) : o_y = (d + x)/2    o_z = (d - x)/2
//
// donde x es la muestra que entra del stream y d la que sale de la linea de
// retardo. o_y va a la etapa siguiente y o_z al twiddle y de vuelta a la linea.
//
// El x1/2 no es un divisor: la suma d±x vive en Q((NBI+1).NBF) con NB+1 bits;
// reinterpretada como Q(NBI.(NBF+1)) su valor YA es la mitad, asi que un cast
// que baja un bit fraccional hace el escalado y el redondeo de una vez.
//
// Complejos empaquetados {imag, real}, misma convencion que el resto.
// -----------------------------------------------------------------------------
module btfly_sdf #(
  parameter int NB  = 16,
  parameter int NBF = 15
)(
  input  wire                i_ctrl,
  input  wire [2*NB-1:0]     i_x,
  input  wire [2*NB-1:0]     i_d,
  output wire [2*NB-1:0]     o_y,
  output wire [2*NB-1:0]     o_z
);

  localparam int RE_LSB  = 0;
  localparam int IM_LSB  = NB;
  localparam int NB_SUM  = NB + 1;
  localparam int NBF_SUM = NBF + 1;

  wire signed [NB-1:0] x_re = $signed(i_x[RE_LSB +: NB]);
  wire signed [NB-1:0] x_im = $signed(i_x[IM_LSB +: NB]);
  wire signed [NB-1:0] d_re = $signed(i_d[RE_LSB +: NB]);
  wire signed [NB-1:0] d_im = $signed(i_d[IM_LSB +: NB]);

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

  assign o_y = i_ctrl ? {sum_rnd_im, sum_rnd_re} : i_d;
  assign o_z = i_ctrl ? {dif_rnd_im, dif_rnd_re} : i_x;

endmodule
`endif
`default_nettype wire
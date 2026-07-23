`timescale 1ns/1ps
`ifndef BTFLY_R2_V
`define BTFLY_R2_V
`include "cast.v"
`ifdef USE_CMUL_KARATSUBA
`include "cmul_karatsuba.v"
`else
`include "cmul.v"
`endif
// -----------------------------------------------------------------------------
// btfly_r2 -- mariposa radix-2 con twiddle y escalado x1/2, combinacional.
//
//   t = W * b                         (cmul: producto exacto + 1 cuantizacion)
//   c = (a + t)/2      d = (a - t)/2  (cast: escalado + redondeo)
//
// CONVENCION DE EMPAQUETADO
// -------------------------
// Cada numero complejo viaja en UN puerto de 2*NB bits:
//
//        bits [2*NB-1 : NB]   parte imaginaria
//        bits [  NB-1 :  0]   parte real
//
// o sea {imag, real}, con la parte real en los bits bajos. Es la misma
// convencion que usa AXI4-Stream para datos I/Q y que std::complex en memoria,
// asi que se mantiene al cruzar hacia el PS o hacia un DMA.
//
// Los part-selects son SIN SIGNO, por eso hay que envolverlos en $signed() al
// desempaquetar: sin eso la aritmetica trataria los operandos como positivos y
// los valores negativos darian resultados incorrectos.
// -----------------------------------------------------------------------------
module btfly_r2 #(
  parameter int NB  = 16,
  parameter int NBF = 15
)(
  input  wire [2*NB-1:0] i_cplx_a,
  input  wire [2*NB-1:0] i_cplx_b,
  input  wire [2*NB-1:0] i_cplx_w,

  output wire [2*NB-1:0] o_cplx_c,
  output wire [2*NB-1:0] o_cplx_d
);

  localparam int RE_LSB = 0;    // posicion de la parte real
  localparam int IM_LSB = NB;   // posicion de la parte imaginaria

  localparam int NB_SUM = NB + 1;   // a±t gana un bit entero
  localparam int NBF_SUM = NBF + 1;

  // ------------------------------------------------------------ desempaquetado
  // Se les da nombre propio a proposito: asi aparecen en el VCD y el diseno
  // sigue siendo legible en gtkwave pese a que los puertos son vectores.
  wire signed [NB-1:0] a_re = $signed(i_cplx_a[RE_LSB +: NB]);
  wire signed [NB-1:0] a_im = $signed(i_cplx_a[IM_LSB +: NB]);
  wire signed [NB-1:0] b_re = $signed(i_cplx_b[RE_LSB +: NB]);
  wire signed [NB-1:0] b_im = $signed(i_cplx_b[IM_LSB +: NB]);
  wire signed [NB-1:0] w_re = $signed(i_cplx_w[RE_LSB +: NB]);
  wire signed [NB-1:0] w_im = $signed(i_cplx_w[IM_LSB +: NB]);

  // ------------------------------------------------------- producto complejo
  wire signed [NB-1:0] t_re ;
  wire signed [NB-1:0] t_im; 

`ifdef USE_CMUL_KARATSUBA
  cmul_karatsuba #(
`else
  cmul #(
`endif
    .NB_IN   (NB),
    .NBF_IN  (NBF),
    .NB_OUT  (NB),
    .NBF_OUT (NBF)
  ) u_cmul (
    .i_1_re  (w_re),
    .i_1_im  (w_im),
    .i_2_re  (b_re),
    .i_2_im  (b_im),
    .o_re    (t_re),
    .o_im    (t_im)
  );

  // ------------------------------------------------- suma/resta y escalado
  wire signed [NB_SUM - 1 : 0]  c_re      ;
  wire signed [NB_SUM - 1 : 0]  c_im      ;
  wire signed [NB_SUM - 1 : 0]  d_re      ;
  wire signed [NB_SUM - 1 : 0]  d_im      ;
  wire signed [NB - 1 : 0]      c_rnd_re  ; 
  wire signed [NB - 1 : 0]      c_rnd_im  ;
  wire signed [NB - 1 : 0]      d_rnd_re  ;
  wire signed [NB - 1 : 0]      d_rnd_im  ; 

  assign c_re = a_re + t_re;
  assign c_im = a_im + t_im;
  assign d_re = a_re - t_re;
  assign d_im = a_im - t_im;


    cast #(
      .NB_IN      (NB_SUM),
      .NBF_IN     (NBF_SUM),
      .NB_OUT     (NB),
      .NBF_OUT    (NBF),
      .ROUND_MODE (1'b1)
    ) u_rnd_c_re (
      .i_word     (c_re),
      .o_word     (c_rnd_re)
    );

    cast #(
      .NB_IN      (NB_SUM),
      .NBF_IN     (NBF_SUM),
      .NB_OUT     (NB),
      .NBF_OUT    (NBF),
      .ROUND_MODE (1'b1)
    ) u_rnd_c_im (
      .i_word     (c_im),
      .o_word     (c_rnd_im)
    );

    cast #(
      .NB_IN      (NB_SUM),
      .NBF_IN     (NBF_SUM),
      .NB_OUT     (NB),
      .NBF_OUT    (NBF),
      .ROUND_MODE (1'b1)
    ) u_rnd_d_re (
      .i_word     (d_re),
      .o_word     (d_rnd_re)
    );

    cast #(
      .NB_IN      (NB_SUM),
      .NBF_IN     (NBF_SUM),
      .NB_OUT     (NB),
      .NBF_OUT    (NBF),
      .ROUND_MODE (1'b1)
    ) u_rnd_d_im (
      .i_word     (d_im),
      .o_word     (d_rnd_im)
    );


  // -------------------------------------------------------------- empaquetado
  assign o_cplx_c = { c_rnd_im, c_rnd_re };
  assign o_cplx_d = { d_rnd_im, d_rnd_re };

endmodule
`endif
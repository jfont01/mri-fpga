  
`timescale 1ns/1ps
`ifndef BTFLY_R2_CORE_V
`define BTFLY_R2_CORE_V
`include "cast.v"
`ifdef USE_CMUL_KARATSUBA
`include "cmul_karatsuba.v"
`else
`include "cmul.v"
`endif

module btfly_r2 #(
  parameter int    NB  = 16,
  parameter int    NBF = 15
)(
  input  wire signed [NB-1:0] i_a_re,
  input  wire signed [NB-1:0] i_a_im,
  input  wire signed [NB-1:0] i_b_re,
  input  wire signed [NB-1:0] i_b_im,
  input  wire signed [NB-1:0] i_w_re,
  input  wire signed [NB-1:0] i_w_im,

  output wire signed [NB-1:0] o_c_re,
  output wire signed [NB-1:0] o_c_im,
  output wire signed [NB-1:0] o_d_re,
  output wire signed [NB-1:0] o_d_im
);

  localparam RE = 0;
  localparam IM = 1;

  localparam int SUM_W = NB + 1;   // a±t en Q((NBI+1).NBF)
  
  wire signed [NB - 1 : 0] t [1 : 0];

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
    .i_1_re  (i_w_re),
    .i_1_im  (i_w_im),
    .i_2_re  (i_b_re),
    .i_2_im  (i_b_im),
    .o_re    (t[RE]),
    .o_im    (t[IM])
  );


  wire signed [SUM_W-1:0] c [1 : 0] ;
  wire signed [SUM_W-1:0] d [1 : 0] ;

  wire signed [NB-1:0] c_rnd [1 : 0] ;
  wire signed [NB-1:0] d_rnd [1 : 0] ;

    assign c[RE] = i_a_re + t[RE];
    assign c[IM] = i_a_im + t[IM];
    assign d[RE] = i_a_re - t[RE];
    assign d[IM] = i_a_im - t[IM];


    genvar z;
    generate
        for (z = 0; z < 2; z = z + 1) begin : gen_rnd
            cast #(
                .NB_IN(SUM_W), 
                .NBF_IN(NBF+1), 
                .NB_OUT(NB), 
                .NBF_OUT(NBF), 
                .ROUND_MODE(1'b1)
            ) u_rnd_c (
                .i_word(c[z]), 
                .o_word(c_rnd[z])
            );

            cast #(
                .NB_IN(SUM_W), 
                .NBF_IN(NBF+1), 
                .NB_OUT(NB), 
                .NBF_OUT(NBF), 
                .ROUND_MODE(1'b1)
            ) u_rnd_d (
                .i_word(d[z]), 
                .o_word(d_rnd[z])
            );
        end
    endgenerate
    
    assign o_c_re = c_rnd[RE];
    assign o_c_im = c_rnd[IM];
    assign o_d_re = d_rnd[RE];
    assign o_d_im = d_rnd[IM];

endmodule
`endif
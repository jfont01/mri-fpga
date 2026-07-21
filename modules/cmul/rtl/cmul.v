`timescale 1ns/1ps
`ifndef CMUL_V
`define CMUL_V
`include "cast.v"

module cmul
#(
    parameter NB_IN  = 8,
    parameter NBF_IN = 5,
    parameter NB_OUT = 6,
    parameter NBF_OUT = 4
)
(
    input  wire signed [NB_IN  - 1 : 0] i_1_re,
    input  wire signed [NB_IN  - 1 : 0] i_1_im,
    input  wire signed [NB_IN  - 1 : 0] i_2_re,
    input  wire signed [NB_IN  - 1 : 0] i_2_im,

    output  wire signed [NB_OUT  - 1 : 0] o_re,
    output  wire signed [NB_OUT  - 1 : 0] o_im
);

    localparam NB_PROD = NB_IN * 2          ;
    localparam NBF_PROD = NBF_IN * 2        ;
    localparam NB_FULL = NB_PROD + 1        ;
    localparam NBF_FULL = NBF_PROD          ;

    wire signed [NB_IN - 1 : 0]         w_a                 ;
    wire signed [NB_IN - 1 : 0]         w_b                 ;
    wire signed [NB_IN - 1 : 0]         w_c                 ;
    wire signed [NB_IN - 1 : 0]         w_d                 ;

    assign w_a = i_1_re;
    assign w_b = i_1_im;
    assign w_c = i_2_re;
    assign w_d = i_2_im;

    wire signed [NB_PROD - 1 : 0]       w_ac                ;
    wire signed [NB_PROD - 1 : 0]       w_bd                ;
    wire signed [NB_PROD - 1 : 0]       w_ad                ;
    wire signed [NB_PROD - 1 : 0]       w_bc                ;

    wire signed [NB_FULL - 1 : 0]       w_ac_minus_bd       ;
    wire signed [NB_FULL - 1 : 0]       w_ad_plus_bc        ;

    wire signed [NB_OUT - 1 : 0 ]       w_rnd_ac_minus_bd   ;
    wire signed [NB_OUT - 1 : 0 ]       w_rnd_ad_plus_bc    ;

    assign w_ac = w_a * w_c;
    assign w_bd = w_b * w_d;
    assign w_ad = w_a * w_d;
    assign w_bc = w_b * w_c;

    assign w_ac_minus_bd = $signed({w_ac[NB_PROD-1], w_ac}) - $signed({w_bd[NB_PROD-1], w_bd});
    assign w_ad_plus_bc = $signed({w_ad[NB_PROD-1], w_ad}) + $signed({w_bc[NB_PROD-1], w_bc});


    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_re (
        .i_word         (w_ac_minus_bd) ,
        .o_word         (w_rnd_ac_minus_bd)
    );

    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_im (
        .i_word         (w_ad_plus_bc)  ,
        .o_word         (w_rnd_ad_plus_bc)
    );

    assign o_re = w_rnd_ac_minus_bd;
    assign o_im = w_rnd_ad_plus_bc;


endmodule
`endif
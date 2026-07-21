`timescale 1ns/1ps
`include "cast.v"

module cmul_karatsuba
#(
    parameter NB_IN  = 8,
    parameter NBF_IN = 5,
    parameter NB_OUT = 6,
    parameter NBF_OUT = 4
)(
    input  wire signed [NB_IN  - 1 : 0] i_1_re,
    input  wire signed [NB_IN  - 1 : 0] i_1_im,
    input  wire signed [NB_IN  - 1 : 0] i_2_re,
    input  wire signed [NB_IN  - 1 : 0] i_2_im,

    output  wire signed [NB_OUT  - 1 : 0] o_re,
    output  wire signed [NB_OUT  - 1 : 0] o_im
);

    localparam NBI_IN   = NB_IN  - NBF_IN               ;
    localparam NBI_OUT  = NB_OUT - NBF_OUT              ;

    localparam NB_SUM   = NB_IN + 1                     ;
    localparam NBF_SUM  = NBF_IN                        ;
    localparam NBI_SUM  = NB_SUM - NBF_SUM              ;

    localparam NB_PROD  = NB_IN + NB_SUM                ;
    localparam NBF_PROD = NBF_IN + NBF_SUM              ;
    localparam NBI_PROD = NB_PROD - NBF_PROD            ;

    localparam NB_FULL  = NB_PROD + 1                   ;
    localparam NBF_FULL = NBF_PROD                      ;
    localparam NBI_FULL = NB_FULL - NBF_FULL            ;


    wire signed [NB_IN - 1 : 0]    w_a                         ;
    wire signed [NB_IN - 1 : 0]    w_b                         ;
    wire signed [NB_IN - 1 : 0]    w_c                         ;
    wire signed [NB_IN - 1 : 0]    w_d                         ;

    wire signed [NB_SUM - 1 : 0]   w_a_plus_b                  ;
    wire signed [NB_SUM - 1 : 0]   w_d_minus_c                 ;
    wire signed [NB_SUM - 1 : 0]   w_c_plus_d                  ;

    wire signed [NB_PROD - 1 : 0]  w_k1                        ;
    wire signed [NB_PROD - 1 : 0]  w_k2                        ;
    wire signed [NB_PROD - 1 : 0]  w_k3                        ;

    wire signed [NB_FULL - 1 : 0]  w_k1_minus_k3               ;
    wire signed [NB_FULL - 1 : 0]  w_k1_plus_k2                ;

    wire signed [NB_OUT - 1 : 0]   w_k1_minus_k3_rnd           ;
    wire signed [NB_OUT - 1 : 0]   w_k1_plus_k2_rnd            ;

    assign w_a              =  i_1_re                   ;
    assign w_b              =  i_1_im                   ;
    assign w_c              =  i_2_re                   ;
    assign w_d              =  i_2_im                   ;

    assign w_a_plus_b       =   $signed({w_a[NB_IN-1], w_a}) + $signed({w_b[NB_IN-1], w_b});
    assign w_d_minus_c      =   $signed({w_d[NB_IN-1], w_d}) - $signed({w_c[NB_IN-1], w_c});
    assign w_c_plus_d       =   $signed({w_c[NB_IN-1], w_c}) + $signed({w_d[NB_IN-1], w_d});

    assign w_k1             =   $signed({w_c[NB_IN-1], w_c}) * w_a_plus_b    ;
    assign w_k2             =   $signed({w_a[NB_IN-1], w_a}) * w_d_minus_c   ;
    assign w_k3             =   $signed({w_b[NB_IN-1], w_b}) * w_c_plus_d    ;

    assign w_k1_minus_k3    =   $signed({w_k1[NB_PROD-1], w_k1}) - $signed({w_k3[NB_PROD-1], w_k3});    
    assign w_k1_plus_k2     =   $signed({w_k1[NB_PROD-1], w_k1}) + $signed({w_k2[NB_PROD-1], w_k2});    


    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_re (
        .i_word         (w_k1_minus_k3) ,
        .o_word         (w_k1_minus_k3_rnd)
    );

    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_im (
        .i_word         (w_k1_plus_k2)  ,
        .o_word         (w_k1_plus_k2_rnd)
    );

    assign o_re = w_k1_minus_k3_rnd;
    assign o_im = w_k1_plus_k2_rnd;

endmodule
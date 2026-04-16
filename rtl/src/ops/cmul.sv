module cmul #(
    parameter NB0_IN = 16,
    parameter NBF0_IN = 15,
    parameter NB1_IN = 16,
    parameter NBF1_IN = 15,
    parameter NB_OUT = 12,
    parameter NBF_OUT = 10
)
(
    input logic signed [NB0_IN - 1 : 0] i_re_0,       //a
    input logic signed [NB0_IN - 1 : 0] i_im_0,       //b
    input logic signed [NB1_IN - 1 : 0] i_re_1,       //c
    input logic signed [NB1_IN - 1 : 0] i_im_1,       //d

    output logic signed  [NB_OUT - 1 : 0] o_re,
    output logic signed  [NB_OUT - 1 : 0] o_im
);

    localparam NB_PROD = NB0_IN + NB1_IN;
    localparam NBF_PROD = NBF0_IN + NBF1_IN;
    localparam NB_SUM = NB_PROD + 1;
    localparam NBF_SUM = NBF_PROD;


    logic signed [NB_PROD - 1 : 0] ac;
    logic signed [NB_PROD - 1 : 0] bd;
    logic signed [NB_PROD - 1 : 0] ad;
    logic signed [NB_PROD - 1 : 0] bc;

    assign ac = i_re_0*i_re_1;
    assign bd = i_im_0*i_im_1;
    assign ad = i_re_0*i_im_1;
    assign bc = i_im_0*i_re_1;

    logic signed [NB_SUM - 1 : 0] ac_minus_bd;
    logic signed [NB_SUM - 1 : 0] ad_plus_bc;

    assign ac_minus_bd = ac - bd;
    assign ad_plus_bc = ad + bc;

    cast #(
        .NB_IN      (NB_SUM),
        .NBF_IN     (NBF_SUM),
        .NB_OUT     (NB_OUT),
        .NBF_OUT    (NBF_OUT),
        .ROUND_MODE (1'b1)

    ) u_cast_cmul_re (
        .i_word(ac_minus_bd),
        .o_word(o_re)
    );

    cast #(
        .NB_IN      (NB_SUM),
        .NBF_IN     (NBF_SUM),
        .NB_OUT     (NB_OUT),
        .NBF_OUT    (NBF_OUT),
        .ROUND_MODE (1'b1)

    ) u_cast_cmul_im (
        .i_word(ad_plus_bc),
        .o_word(o_im)
    );


    
endmodule


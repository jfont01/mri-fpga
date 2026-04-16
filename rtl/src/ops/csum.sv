module csum #(
    parameter NB_IN = 16,
    parameter NBF_IN = 15,
    parameter NB_OUT = 16,
    parameter NBF_OUT = 16
)
(
    input logic signed [NB_IN - 1 : 0] i_re_0,       //a
    input logic signed [NB_IN - 1 : 0] i_im_0,       //b
    input logic signed [NB_IN - 1 : 0] i_re_1,       //c
    input logic signed [NB_IN - 1 : 0] i_im_1,       //d

    output logic signed  [NB_OUT - 1 : 0] o_re,
    output logic signed  [NB_OUT - 1 : 0] o_im
);

    localparam NB_SUM = NB_IN + 1;
    localparam NBF_SUM = NBF_IN;

    logic signed [NB_SUM - 1 : 0] a_plus_c;
    logic signed [NB_SUM - 1 : 0] b_plus_d;
    
    assign a_plus_c = i_re_0 + i_re_1;
    assign b_plus_d = i_im_0 + i_im_1;


    cast #(
        .NB_IN      (NB_SUM),
        .NBF_IN     (NBF_SUM),
        .NB_OUT     (NB_OUT),
        .NBF_OUT    (NBF_OUT),
        .ROUND_MODE (1'b1)

    ) u_cast_csum_re (
        .i_word(a_plus_c),
        .o_word(o_re)
    );

    cast #(
        .NB_IN      (NB_SUM),
        .NBF_IN     (NBF_SUM),
        .NB_OUT     (NB_OUT),
        .NBF_OUT    (NBF_OUT),
        .ROUND_MODE (1'b1)

    ) u_cast_csum_im (
        .i_word(b_plus_d),
        .o_word(o_im)
    );


    
endmodule


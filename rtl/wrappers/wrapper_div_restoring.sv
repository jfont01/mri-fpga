import track_params_pkg::*;

module wrapper_div_restoring (
    input  logic i_clock,
    input  logic i_rst,
    input  logic i_start,

    input  logic signed [NB_DIV_NUM-1:0] i_num,
    input  logic signed [NB_DIV_DEN-1:0] i_den,

    output logic signed [NB_DIV_QUOTIENT-1:0] o_quotient,
    output logic o_ready,
    output logic o_busy
);

    div_restoring #(
        .NB_NUM       (NB_DIV_NUM),
        .NBF_NUM      (NBF_DIV_NUM),
        .NB_DEN       (NB_DIV_DEN),
        .NBF_DEN      (NBF_DIV_DEN),
        .NB_QUOTIENT  (NB_DIV_QUOTIENT),
        .NBF_QUOTIENT (NBF_DIV_QUOTIENT)
    ) u_div_restoring (
        .i_clock    (i_clock),
        .i_rst      (i_rst),
        .i_start    (i_start),
        .i_num      (i_num),
        .i_den      (i_den),
        .o_quotient (o_quotient),
        .o_ready    (o_ready),
        .o_busy     (o_busy)
    );

endmodule
import track_params_pkg::*;

module wrapper_compute_Aij (
    input  logic i_clock,
    input  logic i_rst,
    input  logic i_start,
    input  logic signed [NB_S-1:0] s0_re [L-1:0],
    input  logic signed [NB_S-1:0] s0_im [L-1:0],
    input  logic signed [NB_S-1:0] s1_re [L-1:0],
    input  logic signed [NB_S-1:0] s1_im [L-1:0],
    output logic signed [NB_A-1:0] A00_re,
    output logic signed [NB_A-1:0] A11_re,
    output logic signed [NB_A-1:0] A01_re,
    output logic signed [NB_A-1:0] A01_im,
    output logic signed [NB_A-1:0] A10_re,
    output logic signed [NB_A-1:0] A10_im,
    output logic o_valid
);

    compute_Aij #(
        .NB_S  (NB_S),
        .NBF_S (NBF_S),
        .NB_A  (NB_A),
        .NBF_A (NBF_A),
        .L     (L)
    ) u_compute_Aij (
        .i_clock(i_clock),
        .i_rst  (i_rst),
        .i_start(i_start),
        .s0_re  (s0_re),
        .s0_im  (s0_im),
        .s1_re  (s1_re),
        .s1_im  (s1_im),
        .A00_re (A00_re),
        .A11_re (A11_re),
        .A01_re (A01_re),
        .A01_im (A01_im),
        .A10_re (A10_re),
        .A10_im (A10_im),
        .o_valid(o_valid)
    );

endmodule
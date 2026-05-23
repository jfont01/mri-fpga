import track_params_pkg::*;

module wrapper_compute_bi (
    input  logic i_clock,
    input  logic i_rst,
    input  logic i_start,

    input  logic signed [track_params_pkg::NB_S-1:0] s0_re [track_params_pkg::L-1:0],
    input  logic signed [track_params_pkg::NB_S-1:0] s0_im [track_params_pkg::L-1:0],
    input  logic signed [track_params_pkg::NB_S-1:0] s1_re [track_params_pkg::L-1:0],
    input  logic signed [track_params_pkg::NB_S-1:0] s1_im [track_params_pkg::L-1:0],

    input  logic signed [track_params_pkg::NB_Y-1:0] y_re [track_params_pkg::L-1:0],
    input  logic signed [track_params_pkg::NB_Y-1:0] y_im [track_params_pkg::L-1:0],

    output logic signed [track_params_pkg::NB_B-1:0] b0_re,
    output logic signed [track_params_pkg::NB_B-1:0] b0_im,
    output logic signed [track_params_pkg::NB_B-1:0] b1_re,
    output logic signed [track_params_pkg::NB_B-1:0] b1_im,

    output logic o_valid
);

    compute_bi #(
        .NB_S  (NB_S),
        .NBF_S (NBF_S),
        .NB_B  (NB_B),
        .NBF_B (NBF_B),
        .NB_Y  (NB_Y),
        .NBF_Y (NBF_Y),
        .L     (L)
    ) u_compute_bi (
        .i_clock(i_clock),
        .i_rst  (i_rst),
        .i_start(i_start),

        .s0_re  (s0_re),
        .s0_im  (s0_im),
        .s1_re  (s1_re),
        .s1_im  (s1_im),
        .y_re   (y_re),
        .y_im   (y_im),

        .b0_re  (b0_re),
        .b0_im  (b0_im),
        .b1_re  (b1_re),
        .b1_im  (b1_im),

        .o_valid(o_valid)
    );

endmodule
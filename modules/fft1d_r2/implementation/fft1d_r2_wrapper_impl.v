`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R2_WRAPPER_IMPL_V
`define FFT1D_R2_WRAPPER_IMPL_V
`include "cmul.v"

`ifndef FFT1D_R2_N
`define FFT1D_R2_N 16
`endif
`ifndef FFT1D_R2_NB
`define FFT1D_R2_NB 16
`endif
`ifndef FFT1D_R2_NBF
`define FFT1D_R2_NBF 14
`endif
`ifndef FFT1D_R2_TW_RE_FILE
`define FFT1D_R2_TW_RE_FILE "fft1d_r2_tw_re.mem"
`endif
`ifndef FFT1D_R2_TW_IM_FILE
`define FFT1D_R2_TW_IM_FILE "fft1d_r2_tw_im.mem"
`endif
`ifndef FFT1D_R2_TW_FROM_FILE
`define FFT1D_R2_TW_FROM_FILE 1'b1
`endif


module fft1d_r2_wrapper_impl (
    input wire i_clock
);

    // Implement implementation wrapper

endmodule

`default_nettype wire
`endif



// -----------------------------------------------------------------------------
module cmul_wrapper_impl #(
    parameter int NB_IN   = `CMUL_NB_IN,
    parameter int NBF_IN  = `CMUL_NBF_IN,
    parameter int NB_OUT  = `CMUL_NB_OUT,
    parameter int NBF_OUT = `CMUL_NBF_OUT
)(
    input  wire                     i_clock,
    input  wire signed [NB_IN-1:0]  i_1_re,
    input  wire signed [NB_IN-1:0]  i_1_im,
    input  wire signed [NB_IN-1:0]  i_2_re,
    input  wire signed [NB_IN-1:0]  i_2_im,
    output wire signed [NB_OUT-1:0] o_re,
    output wire signed [NB_OUT-1:0] o_im
);

    // -------------------------------------------------- registros de entrada
    reg signed [NB_IN-1:0] r_1_re;
    reg signed [NB_IN-1:0] r_1_im;
    reg signed [NB_IN-1:0] r_2_re;
    reg signed [NB_IN-1:0] r_2_im;

    // --------------------------------------------------- DUT (combinacional)
    wire signed [NB_OUT-1:0] w_o_re;
    wire signed [NB_OUT-1:0] w_o_im;

    cmul #(
        .NB_IN   (NB_IN),
        .NBF_IN  (NBF_IN),
        .NB_OUT  (NB_OUT),
        .NBF_OUT (NBF_OUT)
    ) u_dut (
        .i_1_re  (r_1_re),
        .i_1_im  (r_1_im),
        .i_2_re  (r_2_re),
        .i_2_im  (r_2_im),
        .o_re    (w_o_re),
        .o_im    (w_o_im)
    );

    // --------------------------------------------------- registros de salida
    reg signed [NB_OUT-1:0] r_o_re;
    reg signed [NB_OUT-1:0] r_o_im;

    always @(posedge i_clock) begin
        r_1_re <= i_1_re;
        r_1_im <= i_1_im;
        r_2_re <= i_2_re;
        r_2_im <= i_2_im;

        r_o_re <= w_o_re;
        r_o_im <= w_o_im;
    end

    assign o_re = r_o_re;
    assign o_im = r_o_im;

endmodule
`default_nettype wire
`endif
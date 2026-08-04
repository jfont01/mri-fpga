`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R2_WRAPPER_SYNTH_V
`define FFT1D_R2_WRAPPER_SYNTH_V
`include "fft1d_r2.v"
`ifndef FFT1D_R2_N
`define FFT1D_R2_N 1024
`endif
`ifndef FFT1D_R2_NB
`define FFT1D_R2_NB 16
`endif
`ifndef FFT1D_R2_NBF
`define FFT1D_R2_NBF 15
`endif
`ifndef FFT1D_R2_TW_RE_FILE
`define FFT1D_R2_TW_RE_FILE "twiddles/fft1d_r2_tw_1024_re.mem"
`endif
`ifndef FFT1D_R2_TW_IM_FILE
`define FFT1D_R2_TW_IM_FILE "twiddles/fft1d_r2_tw_1024_im.mem"
`endif
// -----------------------------------------------------------------------------
// fft1d_r2_wrapper_synth
//
// Igual criterio que los otros arneses de FFT: el R2 iterativo ya es secuencial
// (FSM + memoria), pero se registran entrada y salida para aislar los tramos
// IBUF->FF y FF->OBUF del WNS, de modo que Vivado mida los caminos internos
// (FSM/core/BRAM) y no el ruteo hacia pines. Asi la comparacion contra r2sdf y
// r22sdf es sobre el mismo arnes.
//
// A DIFERENCIA de r2sdf/r22sdf, el R2 recibe los twiddles como DOS archivos
// sueltos (TW_RE_FILE / TW_IM_FILE), no como un directorio TW_DIR. El wrapper
// respeta esa convension.
// -----------------------------------------------------------------------------
`define FFT1D_R2_STRINGIFY(x) `"x`"

module fft1d_r2_wrapper_synth #(
    parameter int N   = `FFT1D_R2_N,
    parameter int NB  = `FFT1D_R2_NB,
    parameter int NBF = `FFT1D_R2_NBF
)(
    input  wire                i_clock,
    input  wire                i_rst,
    input  wire                i_valid,
    input  wire [2*NB-1:0]     i_cplx_sample,
    output wire                o_valid,
    output wire                o_last,
    output wire [2*NB-1:0]     o_cplx_sample
);
    localparam string TW_RE_FILE = `FFT1D_R2_STRINGIFY(`FFT1D_R2_TW_RE_FILE);
    localparam string TW_IM_FILE = `FFT1D_R2_STRINGIFY(`FFT1D_R2_TW_IM_FILE);

    // -------------------------------------------------- registros de entrada
    reg                r_rst;
    reg                r_valid;
    reg [2*NB-1:0]     r_cplx_sample;

    // --------------------------------------------------------------- DUT
    wire               w_o_valid;
    wire               w_o_last;
    wire [2*NB-1:0]    w_o_cplx_sample;

    fft1d_r2 #(
        .N            (N),
        .NB           (NB),
        .NBF          (NBF),
        .TW_RE_FILE   (TW_RE_FILE),
        .TW_IM_FILE   (TW_IM_FILE),
        .TW_FROM_FILE (1'b1)
    ) u_dut (
        .i_clock       (i_clock),
        .i_rst         (r_rst),
        .i_valid       (r_valid),
        .i_cplx_sample (r_cplx_sample),
        .o_valid       (w_o_valid),
        .o_last        (w_o_last),
        .o_cplx_sample (w_o_cplx_sample)
    );

    // -------------------------------------------------- registros de salida
    reg                r_o_valid;
    reg                r_o_last;
    reg [2*NB-1:0]     r_o_cplx_sample;

    always @(posedge i_clock) begin
        r_rst         <= i_rst;
        r_valid       <= i_valid;
        r_cplx_sample <= i_cplx_sample;
        r_o_valid       <= w_o_valid;
        r_o_last        <= w_o_last;
        r_o_cplx_sample <= w_o_cplx_sample;
    end

    assign o_valid       = r_o_valid;
    assign o_last        = r_o_last;
    assign o_cplx_sample = r_o_cplx_sample;
endmodule
`default_nettype wire
`endif
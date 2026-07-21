`timescale 1ns/1ps
`default_nettype none
`ifndef CMUL_WRAPPER_IMPL_V
`define CMUL_WRAPPER_IMPL_V
`include "cmul.v"

`ifndef CMUL_NB_IN
`define CMUL_NB_IN 16
`endif
`ifndef CMUL_NBF_IN
`define CMUL_NBF_IN 14
`endif
`ifndef CMUL_NB_OUT
`define CMUL_NB_OUT 16
`endif
`ifndef CMUL_NBF_OUT
`define CMUL_NBF_OUT 14
`endif

// -----------------------------------------------------------------------------
// cmul_wrapper_impl
//
// cmul es COMBINACIONAL: no tiene reloj propio. Para que la sintesis mida algo
// con sentido hay que encerrarlo entre registros. Sin ellos el unico camino del
// diseno va de pin a pin, no hay nada que restringir con create_clock, y el WNS
// que reporte Vivado no representaria al modulo.
//
//   puertos -> registros de entrada -> cmul -> registros de salida -> puertos
//
// Con esa estructura el UNICO camino sincronico es
//
//   FF -> cmul (logica combinacional) -> FF
//
// que es exactamente el retardo del modulo bajo prueba. Los tramos IBUF->FF y
// FF->OBUF quedan sin restringir (no hay set_input_delay ni set_output_delay),
// asi que no contaminan el WNS.
//
// Para comparar arquitecturas (cmul vs cmul_karatsuba) los dos wrappers son
// identicos salvo el DUT: la diferencia en LUT/FF/DSP y en WNS es atribuible al
// modulo y no al arnes. Los registros aportan 4*NB_IN + 2*NB_OUT flip-flops a
// AMBOS por igual, asi que se cancelan al restar.
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
`timescale 1ns/1ps
`default_nettype none
`ifndef CORDIC_MAG_WRAPPER_SYNTH_V
`define CORDIC_MAG_WRAPPER_SYNTH_V
`include "cordic_mag.v"

/*
 * Parametros por define desde el JSON (CORDIC_MAG_<CLAVE>). Mismos defaults que
 * el RTL, el modelo y el tb.sv: un caso que no los menciona sintetiza los
 * defaults del modulo (n = 12, G = 5, NB_M = 22, half-up).
 */
`ifndef CORDIC_MAG_NB_OUT
`define CORDIC_MAG_NB_OUT 16
`endif
`ifndef CORDIC_MAG_G
`define CORDIC_MAG_G 5
`endif
`ifndef CORDIC_MAG_N_IT
`define CORDIC_MAG_N_IT 12
`endif
`ifndef CORDIC_MAG_NB_M
`define CORDIC_MAG_NB_M (`CORDIC_MAG_NB_OUT + 1 + `CORDIC_MAG_G)
`endif
`ifndef CORDIC_MAG_ROUND_CONV
`define CORDIC_MAG_ROUND_CONV 0
`endif

// -----------------------------------------------------------------------------
// cordic_mag_wrapper_synth
//
// Arnes de sintesis, como el del nucleo FFT: registra las entradas y las
// salidas para que el WNS que reporta Vivado sea el de los caminos INTERNOS de
// cordic_mag (registro -> sumador -> registro) y no el de IBUF -> primer
// registro o ultimo registro -> OBUF, que no estan restringidos (no hay
// set_input_delay / set_output_delay).
//
// Adentro de cordic_mag todos los caminos son de un registro a otro: |.| desde
// los registros de entrada del arnes, una suma/resta por iteracion, y
// redondeo + saturacion antes de r_mag. o_mag y o_valid salen de registros, asi
// que el registro de salida del arnes solo agrega un tramo trivial.
//
// Los registros del arnes aportan 1 + 2 * NB_M + 1 + NB_OUT = 62 FF con los
// defaults; restarlos al leer el reporte de utilizacion.
// -----------------------------------------------------------------------------
module cordic_mag_wrapper_synth #(
    parameter int NB_OUT     = `CORDIC_MAG_NB_OUT,
    parameter int G          = `CORDIC_MAG_G,
    parameter int N_IT       = `CORDIC_MAG_N_IT,
    parameter int NB_M       = `CORDIC_MAG_NB_M,
    parameter int ROUND_CONV = `CORDIC_MAG_ROUND_CONV
)(
    input  wire              i_clock,
    input  wire              i_valid,
    input  wire [NB_M-1:0]   i_m_re,
    input  wire [NB_M-1:0]   i_m_im,
    output wire              o_valid,
    output wire [NB_OUT-1:0] o_mag
);

    // -------------------------------------------------- registros de entrada
    reg                   r_valid = 1'b0;
    reg signed [NB_M-1:0] r_m_re  = {NB_M{1'b0}};
    reg signed [NB_M-1:0] r_m_im  = {NB_M{1'b0}};

    // --------------------------------------------------------------- DUT
    wire              w_o_valid;
    wire [NB_OUT-1:0] w_o_mag;

    cordic_mag #(
        .NB_OUT     (NB_OUT),
        .G          (G),
        .N_IT       (N_IT),
        .NB_M       (NB_M),
        .ROUND_CONV (ROUND_CONV)
    ) u_dut (
        .i_clock (i_clock),
        .i_valid (r_valid),
        .i_m_re  (r_m_re),
        .i_m_im  (r_m_im),
        .o_valid (w_o_valid),
        .o_mag   (w_o_mag)
    );

    // -------------------------------------------------- registros de salida
    reg              r_o_valid = 1'b0;
    reg [NB_OUT-1:0] r_o_mag   = {NB_OUT{1'b0}};

    always @(posedge i_clock) begin
        r_valid   <= i_valid;
        r_m_re    <= i_m_re;
        r_m_im    <= i_m_im;
        r_o_valid <= w_o_valid;
        r_o_mag   <= w_o_mag;
    end

    assign o_valid = r_o_valid;
    assign o_mag   = r_o_mag;
endmodule
`endif // CORDIC_MAG_WRAPPER_SYNTH_V
`default_nettype wire

`timescale 1ns/1ps
`default_nettype none
`ifndef UNFOLD_WRAPPER_SYNTH_V
`define UNFOLD_WRAPPER_SYNTH_V
`include "unfold.v"

/*
 * Parametros: mismos defines y defaults que el RTL, el modelo y el tb.sv. Un
 * caso del JSON que no los menciona sintetiza los defaults de D2.
 */
`ifndef UNFOLD_L
`define UNFOLD_L 8
`endif
`ifndef UNFOLD_AF
`define UNFOLD_AF 2
`endif
`ifndef UNFOLD_NB_A
`define UNFOLD_NB_A 16
`endif
`ifndef UNFOLD_NB_U
`define UNFOLD_NB_U 16
`endif
`ifndef UNFOLD_G
`define UNFOLD_G 5
`endif
`ifndef UNFOLD_NB_M
`define UNFOLD_NB_M (17 + `UNFOLD_G)
`endif
`ifndef UNFOLD_ROUND_CONV
`define UNFOLD_ROUND_CONV 0
`endif
`ifndef UNFOLD_NB_D
`define UNFOLD_NB_D 6
`endif

// -----------------------------------------------------------------------------
// unfold_wrapper_synth
//
// Igual que los arneses del nucleo FFT: se registran las entradas y las salidas
// para aislar los tramos IBUF -> primer registro y ultimo registro -> OBUF (sin
// set_input_delay / set_output_delay no estan restringidos). Asi el WNS que
// reporta Vivado es el de los caminos internos de unfold, en particular el de la
// etapa 4 (registro P -> corrimiento variable -> redondeo -> saturacion ->
// registro M), que es el que hay que mirar.
//
// DIFERENCIA CON EL ARNES DE LA FFT: los puertos de datos no salen enteros a
// pines. Con los defaults, unfold tiene 32 + 64 bits de datos de entrada y 88 de
// salida: con el control y el reloj son 195 pines, y el XCK26 en el SFVC784 tiene
// 186 de I/O de la PL (bancos 43, 44 y 45 HD: 24 + 24 + 22; 64, 65 y 66 HP:
// 42 + 42 + 32; contados en xck26_sfvc784.pkg de Vivado 2024.2), asi que la
// implementacion no podria ubicarlos. Entonces:
//
//   * entrada: los NB_DATA = 2*NB_A + AF*2*NB_U bits de i_a e i_u salen de un
//     registro de desplazamiento que carga NB_PIN bits por ciclo desde i_data.
//     Cada bit del registro es un FF que va al DUT (con fanout al siguiente: no
//     se puede absorber en el DSP ni en un SRL), igual que un registro de
//     entrada. i_valid, i_last e i_d se registran tal cual.
//   * salida: o_m se registra entero (AF*2*NB_M FF) y despues se pliega con XOR
//     en NB_PIN bits, que se registran otra vez antes del pin. Cada bit de o_m
//     llega a algun pin, asi que Vivado no puede podar nada; el XOR es un nivel
//     de LUT entre registros, lejos del camino critico.
//
// Con los defaults: 1 + 2 + 6 + 16 entradas y 2 + 16 salidas = 43 pines. Los
// registros del arnes suman NB_DATA + 2 + NB_D + AF*2*NB_M + 2 + NB_PIN FF.
// -----------------------------------------------------------------------------
module unfold_wrapper_synth #(
    parameter int L          = `UNFOLD_L,
    parameter int AF         = `UNFOLD_AF,
    parameter int NB_A       = `UNFOLD_NB_A,
    parameter int NB_U       = `UNFOLD_NB_U,
    parameter int G          = `UNFOLD_G,
    parameter int NB_M       = `UNFOLD_NB_M,
    parameter int ROUND_CONV = `UNFOLD_ROUND_CONV,
    parameter int NB_D       = `UNFOLD_NB_D,
    parameter int NB_PIN     = 16                 // pines de datos de entrada y de salida
)(
    input  wire               i_clock,
    input  wire               i_valid,
    input  wire               i_last,
    input  wire [NB_D-1:0]    i_d,
    input  wire [NB_PIN-1:0]  i_data,             // i_a e i_u, de a NB_PIN bits por ciclo
    output wire               o_valid,
    output wire               o_err,
    output wire [NB_PIN-1:0]  o_fold              // o_m plegado con XOR
);
    localparam int NB_IA   = 2*NB_A;
    localparam int NB_IU   = AF*2*NB_U;
    localparam int NB_DATA = NB_IA + NB_IU;
    localparam int NB_OM   = AF*2*NB_M;
    localparam int N_FOLD  = (NB_OM + NB_PIN - 1) / NB_PIN;    // tramos de NB_PIN bits

    // -------------------------------------------------- registros de entrada
    reg               r_valid = 1'b0;
    reg               r_last  = 1'b0;
    reg [NB_D-1:0]    r_d     = '0;
    reg [NB_DATA-1:0] r_data  = '0;

    always @(posedge i_clock) begin
        r_valid <= i_valid;
        r_last  <= i_last;
        r_d     <= i_d;
        r_data  <= (NB_DATA > NB_PIN) ? {r_data, i_data} : NB_DATA'(i_data);
    end

    // --------------------------------------------------------------- DUT
    wire              w_o_valid;
    wire              w_o_err;
    wire [NB_OM-1:0]  w_o_m;

    unfold #(
        .L          (L),
        .AF         (AF),
        .NB_A       (NB_A),
        .NB_U       (NB_U),
        .G          (G),
        .NB_M       (NB_M),
        .ROUND_CONV (ROUND_CONV),
        .NB_D       (NB_D)
    ) u_dut (
        .i_clock    (i_clock),
        .i_valid    (r_valid),
        .i_last     (r_last),
        .i_a        (r_data[0     +: NB_IA]),
        .i_u        (r_data[NB_IA +: NB_IU]),
        .i_d        (r_d),
        .o_valid    (w_o_valid),
        .o_err      (w_o_err),
        .o_m        (w_o_m)
    );

    // -------------------------------------------------- registros de salida
    reg                      r_o_valid = 1'b0;
    reg                      r_o_err   = 1'b0;
    reg [NB_OM-1:0]          r_o_m     = '0;
    reg [NB_PIN-1:0]         r_o_fold  = '0;

    // o_m registrado, rellenado con ceros hasta N_FOLD tramos y plegado con XOR.
    wire [N_FOLD*NB_PIN-1:0] w_o_m_pad = (N_FOLD*NB_PIN)'(r_o_m);
    reg  [NB_PIN-1:0]        w_fold;

    always @* begin
        w_fold = '0;
        for (int j = 0; j < N_FOLD; j++)
            w_fold = w_fold ^ w_o_m_pad[j*NB_PIN +: NB_PIN];
    end

    always @(posedge i_clock) begin
        r_o_valid <= w_o_valid;
        r_o_err   <= w_o_err;
        r_o_m     <= w_o_m;
        r_o_fold  <= w_fold;
    end

    assign o_valid = r_o_valid;
    assign o_err   = r_o_err;
    assign o_fold  = r_o_fold;
endmodule
`endif
`default_nettype wire

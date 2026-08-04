`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R2SDF_WRAPPER_SYNTH_V
`define FFT1D_R2SDF_WRAPPER_SYNTH_V
`include "fft1d_r2sdf.v"
`ifndef FFT1D_R2SDF_N
`define FFT1D_R2SDF_N 1024
`endif
`ifndef FFT1D_R2SDF_NB
`define FFT1D_R2SDF_NB 16
`endif
`ifndef FFT1D_R2SDF_NBF
`define FFT1D_R2SDF_NBF 15
`endif
`ifndef FFT1D_R2SDF_TW_DIR
`define FFT1D_R2SDF_TW_DIR "twiddles/n1024"
`endif
// -----------------------------------------------------------------------------
// fft1d_r2sdf_wrapper_synth
//
// El FFT ya es SECUENCIAL (tiene i_clock, contador, delay lines), asi que a
// diferencia del cmul no hace falta registrar para "crear" un camino
// sincronico: ya existen caminos registro->logica->registro adentro. Se
// registran igual las entradas y salidas por la MISMA razon del arnes de cmul:
// aislar los tramos IBUF->primer_registro y ultimo_registro->OBUF del analisis.
// Sin set_input_delay/set_output_delay esos tramos no estan restringidos; al
// registrarlos en el arnes, el WNS que reporta Vivado corresponde a los caminos
// INTERNOS del FFT (que es lo que se quiere medir) y no al ruteo hacia pines.
//
// Para comparar las tres arquitecturas (r2, r2sdf, r2sdf) los tres wrappers
// son identicos salvo el DUT y como recibe los twiddles: la diferencia en
// LUT/FF/DSP/BRAM y en WNS es atribuible al modulo y no al arnes. Los registros
// de I/O aportan 2*(2*NB) + control flip-flops a los tres por igual.
//
// STRINGIFY: TW_DIR se pasa como -D...=algo por linea de comando; para que
// llegue como string de Verilog con comillas se expande con esta macro.
// -----------------------------------------------------------------------------
`define FFT1D_R2SDF_STRINGIFY(x) `"x`"

module fft1d_r2sdf_wrapper_synth #(
    parameter int N   = `FFT1D_R2SDF_N,
    parameter int NB  = `FFT1D_R2SDF_NB,
    parameter int NBF = `FFT1D_R2SDF_NBF
)(
    input  wire                i_clock,
    input  wire                i_rst,
    input  wire                i_valid,
    input  wire [2*NB-1:0]     i_cplx_sample,
    output wire                o_valid,
    output wire                o_last,
    output wire [2*NB-1:0]     o_cplx_sample
);
    localparam string TW_DIR = `FFT1D_R2SDF_STRINGIFY(`FFT1D_R2SDF_TW_DIR);

    // -------------------------------------------------- registros de entrada
    reg                r_valid;
    reg [2*NB-1:0]     r_cplx_sample;
    reg                r_rst;

    // --------------------------------------------------------------- DUT
    wire               w_o_valid;
    wire               w_o_last;
    wire [2*NB-1:0]    w_o_cplx_sample;

    fft1d_r2sdf #(
        .N            (N),
        .NB           (NB),
        .NBF          (NBF),
        .TW_DIR       (TW_DIR),
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
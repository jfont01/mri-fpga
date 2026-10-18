`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_WRAPPER_SYNTH_V
`define FFT1D_R22SDF_WRAPPER_SYNTH_V
`include "fft1d_r22sdf.v"
`ifndef FFT1D_R22SDF_N
`define FFT1D_R22SDF_N 1024
`endif
`ifndef FFT1D_R22SDF_NB
`define FFT1D_R22SDF_NB 16
`endif
`ifndef FFT1D_R22SDF_NBF
`define FFT1D_R22SDF_NBF 15
`endif
// Sin comillas: STRINGIFY las agrega. Con comillas en el default, la macro
// producia ""twiddles/n1024"" y el wrapper no elaboraba si el JSON no pasaba
// TW_DIR.
`ifndef FFT1D_R22SDF_TW_DIR
`define FFT1D_R22SDF_TW_DIR twiddles/n1024
`endif

/*
 * Formatos independientes del datapath y mascara de escalado. Mismos defaults
 * que el RTL, el modelo y el tb.sv: si no se fija el ancho, es el del datapath;
 * si no se fijan los fraccionales, se heredan los bits enteros del datapath.
 * Un caso del JSON que no los menciona sintetiza exactamente lo mismo que antes.
 */
`ifndef FFT1D_R22SDF_NB_TW
`define FFT1D_R22SDF_NB_TW `FFT1D_R22SDF_NB
`endif
`ifndef FFT1D_R22SDF_NBF_TW
`define FFT1D_R22SDF_NBF_TW (`FFT1D_R22SDF_NB_TW - (`FFT1D_R22SDF_NB - `FFT1D_R22SDF_NBF))
`endif
`ifndef FFT1D_R22SDF_NB_IN
`define FFT1D_R22SDF_NB_IN `FFT1D_R22SDF_NB
`endif
`ifndef FFT1D_R22SDF_NBF_IN
`define FFT1D_R22SDF_NBF_IN (`FFT1D_R22SDF_NB_IN - (`FFT1D_R22SDF_NB - `FFT1D_R22SDF_NBF))
`endif
`ifndef FFT1D_R22SDF_NB_OUT
`define FFT1D_R22SDF_NB_OUT `FFT1D_R22SDF_NB
`endif
`ifndef FFT1D_R22SDF_NBF_OUT
`define FFT1D_R22SDF_NBF_OUT (`FFT1D_R22SDF_NB_OUT - (`FFT1D_R22SDF_NB - `FFT1D_R22SDF_NBF))
`endif
// En el JSON, en DECIMAL (784, no 0x310): el mismo literal vale en C++ y Verilog.
`ifndef FFT1D_R22SDF_SCALE_MASK
`define FFT1D_R22SDF_SCALE_MASK 32'hFFFF_FFFF
`endif
// Registro antes del cast de cada cmul (PREG del DSP). Default 1, igual que el
// RTL; 0 = sin registro (referencia).
`ifndef FFT1D_R22SDF_CMUL_PIPE
`define FFT1D_R22SDF_CMUL_PIPE 1
`endif
// -----------------------------------------------------------------------------
// fft1d_r22sdf_wrapper_synth
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
// Ojo: el cast de salida del FFT (Q(NB,NBF) -> Q(NB_OUT,NBF_OUT)) es
// combinacional, asi que el camino ultimo_registro_del_FFT -> cast -> registro
// de salida del arnes SI se mide. Es justamente el que hay que mirar al pasar
// de 16 a 20 bits.
//
// Para comparar las tres arquitecturas (r2, r2sdf, r22sdf) los tres wrappers
// son identicos salvo el DUT y como recibe los twiddles: la diferencia en
// LUT/FF/DSP/BRAM y en WNS es atribuible al modulo y no al arnes. Los registros
// de I/O aportan 2*NB_IN + 2*NB_OUT + control flip-flops.
//
// STRINGIFY: TW_DIR se pasa como -D...=algo por linea de comando; para que
// llegue como string de Verilog con comillas se expande con esta macro.
// -----------------------------------------------------------------------------
`define FFT1D_R22SDF_STRINGIFY(x) `"x`"

module fft1d_r22sdf_wrapper_synth #(
    parameter int          N          = `FFT1D_R22SDF_N,
    parameter int          NB         = `FFT1D_R22SDF_NB,
    parameter int          NBF        = `FFT1D_R22SDF_NBF,
    parameter logic [31:0] SCALE_MASK = `FFT1D_R22SDF_SCALE_MASK,
    parameter int          NB_TW      = `FFT1D_R22SDF_NB_TW,
    parameter int          NBF_TW     = `FFT1D_R22SDF_NBF_TW,
    parameter int          NB_IN      = `FFT1D_R22SDF_NB_IN,
    parameter int          NBF_IN     = `FFT1D_R22SDF_NBF_IN,
    parameter int          NB_OUT     = `FFT1D_R22SDF_NB_OUT,
    parameter int          NBF_OUT    = `FFT1D_R22SDF_NBF_OUT,
    parameter bit          CMUL_PIPE  = `FFT1D_R22SDF_CMUL_PIPE
)(
    input  wire                  i_clock,
    input  wire                  i_rst,
    input  wire                  i_valid,
    input  wire [2*NB_IN-1:0]    i_cplx_sample,
    output wire                  o_valid,
    output wire                  o_last,
    output wire [2*NB_OUT-1:0]   o_cplx_sample
);
    localparam string TW_DIR = `FFT1D_R22SDF_STRINGIFY(`FFT1D_R22SDF_TW_DIR);

    // -------------------------------------------------- registros de entrada
    reg                  r_valid;
    reg [2*NB_IN-1:0]    r_cplx_sample;
    reg                  r_rst;

    // --------------------------------------------------------------- DUT
    wire                 w_o_valid;
    wire                 w_o_last;
    wire [2*NB_OUT-1:0]  w_o_cplx_sample;

    fft1d_r22sdf #(
        .N            (N),
        .NB           (NB),
        .NBF          (NBF),
        .TW_DIR       (TW_DIR),
        .TW_FROM_FILE (1'b1),
        .SCALE_MASK   (SCALE_MASK),
        .NB_TW        (NB_TW),
        .NBF_TW       (NBF_TW),
        .NB_IN        (NB_IN),
        .NBF_IN       (NBF_IN),
        .NB_OUT       (NB_OUT),
        .NBF_OUT      (NBF_OUT),
        .CMUL_PIPE    (CMUL_PIPE)
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
    reg                  r_o_valid;
    reg                  r_o_last;
    reg [2*NB_OUT-1:0]   r_o_cplx_sample;

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

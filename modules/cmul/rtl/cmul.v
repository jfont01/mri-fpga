`timescale 1ns/1ps
`ifndef CMUL_V
`define CMUL_V
`include "cast.v"

// -----------------------------------------------------------------------------
// cmul -- multiplicador complejo directo (4 productos reales), con redondeo y
// saturacion a la salida.
//
//   o = i_1 * i_2 = (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
//
// ANCHOS DE LOS OPERANDOS
// -----------------------
// Cada operando tiene su propio formato: i_1 en Q(NB_IN1, NBF_IN1) e i_2 en
// Q(NB_IN2, NBF_IN2). El producto completo se arma en precision exacta:
//
//   NB_PROD  = NB_IN1  + NB_IN2         NBF_PROD = NBF_IN1 + NBF_IN2
//   NB_FULL  = NB_PROD + 1  (bit de la suma/resta)
//
// y un unico cast lo lleva a Q(NB_OUT, NBF_OUT) con redondeo y saturacion.
//
// Por defecto NB_IN1/NBF_IN1/NB_IN2/NBF_IN2 heredan NB_IN/NBF_IN, asi que una
// instancia que solo fija NB_IN (por nombre o por posicion) se comporta
// exactamente como la version de ancho comun. Por eso los parametros nuevos
// van al final de la lista.
//
// En el DSP48E2 (multiplicador 27 x 18 con signo) cada producto real ocupa un
// solo DSP mientras un operando quepa en 27 bits y el otro en 18.
//
// SEGMENTACION (PIPE)
// -------------------
//   PIPE = 0 : combinacional, como siempre. i_clock e i_rst no se usan.
//   PIPE = 1 : registra la suma completa (NB_FULL bits) ANTES del cast. La
//              salida llega un ciclo despues; el valor es el mismo bit a bit.
//
// Sin el registro, el camino es  registro de entrada -> multiplicador -> suma
// en cascada -> salida P sin registrar -> cast en fabrica -> registro del que
// instancia. Es la configuracion mas lenta del DSP48E2 (DS925, tabla 83) y es
// el camino critico de la FFT R2^2SDF despues del ruteo (2026-10-17).
//
// Con PIPE = 1 Vivado absorbe el registro como PREG del segundo DSP de la
// cascada (no gasta LUT ni FF) y el camino queda partido en dos:
//   1) registro de entrada -> multiplicador -> cascada -> PREG  (dentro del DSP)
//   2) PREG -> cast -> registro del que instancia               (en fabrica)
// Para que lo absorba, el registro tiene reset SINCRONO (RSTP) y no lleva
// enable.
//
// Los puertos i_clock e i_rst van al final para no romper instancias
// posicionales existentes. Con PIPE = 0 pueden quedar sin conectar.
// -----------------------------------------------------------------------------
module cmul
#(
    parameter NB_IN   = 8,
    parameter NBF_IN  = 5,
    parameter NB_OUT  = 6,
    parameter NBF_OUT = 4,
    parameter NB_IN1  = NB_IN,      // ancho del operando 1
    parameter NBF_IN1 = NBF_IN,
    parameter NB_IN2  = NB_IN,      // ancho del operando 2
    parameter NBF_IN2 = NBF_IN,
    parameter PIPE    = 0           // 1: registro antes del cast (latencia 1)
)
(
    input  wire signed [NB_IN1 - 1 : 0] i_1_re,
    input  wire signed [NB_IN1 - 1 : 0] i_1_im,
    input  wire signed [NB_IN2 - 1 : 0] i_2_re,
    input  wire signed [NB_IN2 - 1 : 0] i_2_im,

    output  wire signed [NB_OUT  - 1 : 0] o_re,
    output  wire signed [NB_OUT  - 1 : 0] o_im,

    input  wire                         i_clock,    // solo con PIPE = 1
    input  wire                         i_rst       // sincrono, activo alto; solo con PIPE = 1
);

    // --------------------------------------------- chequeo de parametros
    // Doble $fatal: el de elaboracion lo respetan Vivado y XSim; iverilog lo
    // ignora en silencio, asi que el del initial lo hace cortar en tiempo 0.
    `ifndef __SYNTHESIS__
    generate
        if (PIPE != 0 && PIPE != 1) begin : g_err_pipe
            $fatal(1, "cmul: PIPE=%0d invalido (0 o 1)", PIPE);
            initial $fatal(1, "cmul: PIPE=%0d invalido (0 o 1)", PIPE);
        end
    endgenerate
    `endif // __SYNTHESIS__

    localparam NB_PROD  = NB_IN1  + NB_IN2  ;
    localparam NBF_PROD = NBF_IN1 + NBF_IN2 ;
    localparam NB_FULL  = NB_PROD + 1       ;
    localparam NBF_FULL = NBF_PROD          ;

    wire signed [NB_IN1 - 1 : 0]        w_a                 ;
    wire signed [NB_IN1 - 1 : 0]        w_b                 ;
    wire signed [NB_IN2 - 1 : 0]        w_c                 ;
    wire signed [NB_IN2 - 1 : 0]        w_d                 ;

    assign w_a = i_1_re;
    assign w_b = i_1_im;
    assign w_c = i_2_re;
    assign w_d = i_2_im;

    wire signed [NB_PROD - 1 : 0]       w_ac                ;
    wire signed [NB_PROD - 1 : 0]       w_bd                ;
    wire signed [NB_PROD - 1 : 0]       w_ad                ;
    wire signed [NB_PROD - 1 : 0]       w_bc                ;

    wire signed [NB_FULL - 1 : 0]       w_ac_minus_bd       ;
    wire signed [NB_FULL - 1 : 0]       w_ad_plus_bc        ;

    wire signed [NB_OUT - 1 : 0 ]       w_rnd_ac_minus_bd   ;
    wire signed [NB_OUT - 1 : 0 ]       w_rnd_ad_plus_bc    ;

    assign w_ac = w_a * w_c;
    assign w_bd = w_b * w_d;
    assign w_ad = w_a * w_d;
    assign w_bc = w_b * w_c;

    assign w_ac_minus_bd = $signed({w_ac[NB_PROD-1], w_ac}) - $signed({w_bd[NB_PROD-1], w_bd});
    assign w_ad_plus_bc = $signed({w_ad[NB_PROD-1], w_ad}) + $signed({w_bc[NB_PROD-1], w_bc});

    // Suma completa que entra al cast: directa (PIPE = 0) o registrada (PIPE = 1).
    wire signed [NB_FULL - 1 : 0]       w_re_full           ;
    wire signed [NB_FULL - 1 : 0]       w_im_full           ;

    generate
        if (PIPE == 1) begin : g_pipe
            reg signed [NB_FULL - 1 : 0] r_ac_minus_bd;
            reg signed [NB_FULL - 1 : 0] r_ad_plus_bc;

            always @(posedge i_clock) begin
                if (i_rst) begin
                    r_ac_minus_bd <= {NB_FULL{1'b0}};
                    r_ad_plus_bc  <= {NB_FULL{1'b0}};
                end
                else begin
                    r_ac_minus_bd <= w_ac_minus_bd;
                    r_ad_plus_bc  <= w_ad_plus_bc;
                end
            end

            assign w_re_full = r_ac_minus_bd;
            assign w_im_full = r_ad_plus_bc;
        end
        else begin : g_comb
            assign w_re_full = w_ac_minus_bd;
            assign w_im_full = w_ad_plus_bc;
        end
    endgenerate

    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_re (
        .i_word         (w_re_full)     ,
        .o_word         (w_rnd_ac_minus_bd)
    );

    cast #(
        .NB_IN          (NB_FULL)       ,
        .NBF_IN         (NBF_FULL)      ,
        .NB_OUT         (NB_OUT)        ,
        .NBF_OUT        (NBF_OUT)       ,
        .ROUND_MODE     (1'b1)
    ) u_round_im (
        .i_word         (w_im_full)     ,
        .o_word         (w_rnd_ad_plus_bc)
    );

    assign o_re = w_rnd_ac_minus_bd;
    assign o_im = w_rnd_ad_plus_bc;


endmodule
`endif

`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_STAGE_V
`define FFT1D_R22SDF_STAGE_V
`include "delay_line.v"
`include "btfly_sdf.v"
`include "cmul.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf_stage -- una etapa del pipeline radix-2^2 SDF.
//
// A diferencia del R2SDF (un multiplicador por etapa), el radix-2^2 agrupa las
// etapas de a PARES. Dentro de cada par:
//
//   - etapa BF2I  (STAGE_KIND=0): mariposa SDF simple, sin twiddle.
//   - etapa BF2II (STAGE_KIND=1): mariposa SDF + multiplicacion TRIVIAL por -j
//                                 en la mitad correspondiente del bloque, y
//                                 luego el MULTIPLICADOR no trivial (twiddle de
//                                 ROM) que cierra el par -- salvo el ultimo par.
//
// Esto reduce los multiplicadores a log4(N)-1 (contra log2(N)-1 del R2SDF),
// que es la ventaja del radix-2^2. La linea de retardo y la mariposa son las
// MISMAS piezas comunes que usa el R2SDF (delay_line, btfly_sdf, cmul, cast).
//
// CONTROL (derivado de la implementacion de referencia nanamake/r22sdf, MIT):
//   - i_ctrl : fase de mariposa (butterfly si/no), = bit del contador de esta et.
//   - i_mj   : habilita el -j trivial (solo en BF2II), = (2 bits altos == 11).
//   - i_addr : puntero de linea de retardo y direccion de ROM de twiddle.
//
// El -j trivial: -j*(x+jy) = y - jx  ->  (re,im) -> (im, -re).
// -----------------------------------------------------------------------------
module fft1d_r22sdf_stage #(
  parameter int    NB         = 16,
  parameter int    NBF        = 15,
  parameter int    DEPTH      = 256,     // largo de la linea de retardo
  parameter int    ADDR_W     = 8,       // $clog2(DEPTH), o 1 si DEPTH==1
  parameter int    STAGE_KIND = 0,       // 0 = BF2I, 1 = BF2II
  parameter bit    HAS_MULT   = 1'b1,    // multiplicador tras el par (solo BF2II)
  parameter        TW_RE_FILE = "tw_re.mem",
  parameter        TW_IM_FILE = "tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                i_clock,
  input  wire [ADDR_W-1:0]   i_addr,     // p = contador mod DEPTH
  input  wire                i_ctrl,     // fase de mariposa
  input  wire                i_mj,       // habilita -j (solo BF2II)
  input  wire [2*NB-1:0]     i_data,
  output wire [2*NB-1:0]     o_data
);

  localparam int RE_LSB = 0;
  localparam int IM_LSB = NB;

  wire [2*NB-1:0] w_d;     // sale de la linea de retardo
  wire [2*NB-1:0] w_y;     // hacia la etapa siguiente
  wire [2*NB-1:0] w_z;     // salida de la mariposa (rama a la linea)
  wire [2*NB-1:0] w_zj;    // tras el -j trivial (si BF2II)
  wire [2*NB-1:0] w_tw;    // tras el multiplicador (si HAS_MULT)
  wire [2*NB-1:0] w_din;   // lo que entra a la linea

  // -------------------------------------------------------------- mariposa
  btfly_sdf #(
    .NB      (NB),
    .NBF     (NBF)
  ) u_btfly (
    .i_ctrl  (i_ctrl),
    .i_x     (i_data),
    .i_d     (w_d),
    .o_y     (w_y),
    .o_z     (w_z)
  );

  // ------------------------------------------------ -j trivial (solo BF2II)
  generate
    if (STAGE_KIND == 1) begin : gen_mj
      wire signed [NB-1:0] z_re = $signed(w_z[RE_LSB +: NB]);
      wire signed [NB-1:0] z_im = $signed(w_z[IM_LSB +: NB]);

      // -re saturado: -(-2^(NB-1)) no entra en NB bits
      wire signed [NB-1:0] neg_re = (z_re == {1'b1, {(NB-1){1'b0}}})
                                  ? {1'b0, {(NB-1){1'b1}}}
                                  : -z_re;

      // -j: (re,im) -> (im, -re)
      assign w_zj = i_mj ? {neg_re, z_im} : w_z;
    end
    else begin : gen_no_mj
      assign w_zj = w_z;   // BF2I no aplica -j
    end
  endgenerate

  // ------------------------------------------------ multiplicador (fin de par)
  generate
    if (HAS_MULT) begin : gen_mult
      reg signed [NB-1:0] tw_re [0:DEPTH-1];
      reg signed [NB-1:0] tw_im [0:DEPTH-1];

      if (TW_FROM_FILE) begin : gen_tw_init
        initial begin
          $readmemh(TW_RE_FILE, tw_re);
          $readmemh(TW_IM_FILE, tw_im);
        end
      end

      wire signed [NB-1:0] w_tw_re = tw_re[i_addr];
      wire signed [NB-1:0] w_tw_im = tw_im[i_addr];
      wire signed [NB-1:0] w_out_re, w_out_im;

      cmul #(
        .NB_IN   (NB),
        .NBF_IN  (NBF),
        .NB_OUT  (NB),
        .NBF_OUT (NBF)
      ) u_cmul (
        .i_1_re  (w_tw_re),
        .i_1_im  (w_tw_im),
        .i_2_re  ($signed(w_zj[RE_LSB +: NB])),
        .i_2_im  ($signed(w_zj[IM_LSB +: NB])),
        .o_re    (w_out_re),
        .o_im    (w_out_im)
      );

      assign w_tw = {w_out_im, w_out_re};
    end
    else begin : gen_no_mult
      assign w_tw = w_zj;   // sin multiplicador (BF2I, o ultimo par)
    end
  endgenerate

  /*
   * El twiddle/-j SOLO actua en la fase de mariposa. Con i_ctrl=0 lo que entra
   * a la linea es la muestra cruda del stream (aun sin procesar); tocarla aca
   * la corromperia.
   */
  assign w_din = i_ctrl ? w_tw : w_z;

  // ------------------------------------------------------------- linea
  delay_line #(
    .DEPTH   (DEPTH),
    .WIDTH   (2*NB),
    .ADDR_W  (ADDR_W)
  ) u_delay (
    .i_clock (i_clock),
    .i_addr  (i_addr),
    .i_data  (w_din),
    .o_data  (w_d)
  );

  assign o_data = w_y;

endmodule
`endif
`default_nettype wire
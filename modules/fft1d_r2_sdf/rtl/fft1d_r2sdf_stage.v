`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R2SDF_STAGE_V
`define FFT1D_R2SDF_STAGE_V
`include "delay_line.v"
`include "btfly_sdf.v"
`include "cmul.v"
// -----------------------------------------------------------------------------
// fft1d_r2sdf_stage -- una etapa del pipeline R2SDF.
//
//   entrada -> btfly_sdf -> (twiddle) -> delay_line -> realimenta a btfly_sdf
//                      |
//                      +-> salida hacia la etapa siguiente
//
// El twiddle depende de la etapa:
//
//   HAS_MULT = 1 : multiplicador complejo real. La ROM tiene DEPTH entradas y
//                  se indexa DIRECTAMENTE con i_addr, porque ya viene horneada
//                  con el factor 2^k de esta etapa.
//
//   HAS_MULT = 0 : todos los twiddles de la etapa son triviales (1 o -j), lo
//                  que pasa en las dos ultimas. Se resuelve con cableado:
//                      addr = 0 -> x1
//                      addr = 1 -> x(-j) : -j(x+jy) = y - jx
//                  Se usan los valores EXACTOS, no los de una ROM (donde W^0
//                  quedaria como 0x7FFF por saturacion en Q1.15).
//
// Por eso el total de multiplicadores es log2(N)-2 y no log2(N).
// -----------------------------------------------------------------------------
module fft1d_r2sdf_stage #(
  parameter int    NB       = 16,
  parameter int    NBF      = 15,
  parameter int    DEPTH    = 256,      // largo de la linea de retardo
  parameter int    ADDR_W   = 8,        // $clog2(DEPTH), o 1 si DEPTH==1
  parameter bit    HAS_MULT = 1'b1,
  parameter        TW_RE_FILE = "tw_re.mem",
  parameter        TW_IM_FILE = "tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                i_clock,
  input  wire [ADDR_W-1:0]   i_addr,     // p = contador mod DEPTH
  input  wire                i_ctrl,     // bit log2(DEPTH) del contador
  input  wire [2*NB-1:0]     i_data,
  output wire [2*NB-1:0]     o_data
);

  localparam int RE_LSB = 0;
  localparam int IM_LSB = NB;

  wire [2*NB-1:0] w_d;    // sale de la linea de retardo
  wire [2*NB-1:0] w_y;    // hacia la etapa siguiente
  wire [2*NB-1:0] w_z;    // salida de la mariposa, antes del twiddle
  wire [2*NB-1:0] w_tw;   // salida del twiddle
  wire [2*NB-1:0] w_din;  // lo que realmente entra a la linea

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
        .i_2_re  ($signed(w_z[RE_LSB +: NB])),
        .i_2_im  ($signed(w_z[IM_LSB +: NB])),
        .o_re    (w_out_re),
        .o_im    (w_out_im)
      );

      assign w_tw = {w_out_im, w_out_re};
    end
    else begin : gen_trivial
      // solo 1 y -j: intercambio de componentes y cambio de signo
      wire signed [NB-1:0] z_re = $signed(w_z[RE_LSB +: NB]);
      wire signed [NB-1:0] z_im = $signed(w_z[IM_LSB +: NB]);

      // -re saturado: -(-2^(NB-1)) no entra en NB bits
      wire signed [NB-1:0] neg_re = (z_re == {1'b1, {(NB-1){1'b0}}})
                                  ? {1'b0, {(NB-1){1'b1}}}
                                  : -z_re;

      wire sel_mj = (DEPTH > 1) ? i_addr[0] : 1'b0;

      assign w_tw = sel_mj ? {neg_re, z_im} : w_z;
    end
  endgenerate

  /*
   * El twiddle SOLO actua en la fase de mariposa. Con i_ctrl=0 lo que entra a
   * la linea es la muestra cruda del stream, que todavia no fue procesada:
   * multiplicarla aca la corromperia.
   */
  assign w_din = i_ctrl ? w_tw : w_z;

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
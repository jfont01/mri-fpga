`timescale 1ns/1ps
`ifndef FFT1D_R2_CORE_V
`define FFT1D_R2_CORE_V
`include "fft1d_r2_mem.v"
`include "btfly_r2.v"
// -----------------------------------------------------------------------------
// fft1d_r2_core -- plano de DATOS de la FFT iterativa.
//
// Cablea tres piezas y no hace aritmetica propia:
//   fft1d_r2_mem : memoria de trabajo (N muestras complejas, in-place)
//   ROM twiddles : solo lectura, inicializada por $readmemh
//   btfly_r2     : producto complejo + suma/resta + escalado x1/2
//
// No tiene FSM ni contadores: recibe direcciones y enables desde fft1d_r2_fsm.
// Los complejos viajan empaquetados {imag, real} en 2*NB bits.
// -----------------------------------------------------------------------------
module fft1d_r2_core #(
  parameter int    N   = 512,
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_RE_FILE = "fft1d_r2_tw_re.mem",
  parameter        TW_IM_FILE = "fft1d_r2_tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                  i_clock,

  input  wire [2*NB-1:0]       i_cplx_sample,
  input  wire                  i_load_en,
  input  wire [$clog2(N)-1:0]  i_load_addr,

  input  wire                  i_btfly_en,
  input  wire [$clog2(N)-1:0]  i_idx_a,
  input  wire [$clog2(N)-1:0]  i_idx_b,
  input  wire [$clog2(N)-2:0]  i_idx_tw,

  input  wire [$clog2(N)-1:0]  i_out_addr,
  input  wire                  i_out_en,
  output wire  [2*NB-1:0]      o_cplx_sample
);

  localparam int NH = N / 2;

  // ------------------------------------------------------------ ROM twiddles
  // Se deja aca y no dentro de fft1d_r2_mem porque es de naturaleza distinta:
  // solo lectura, un puerto, e inicializada desde archivo.
  reg signed [NB-1:0] tw_re [0:NH-1];
  reg signed [NB-1:0] tw_im [0:NH-1];

  generate
    if (TW_FROM_FILE) begin : gen_tw_init
      initial begin
        $readmemh(TW_RE_FILE, tw_re);
        $readmemh(TW_IM_FILE, tw_im);
      end
    end
  endgenerate

  // --------------------------------------------------------------- memoria
  wire [2*NB-1:0] w_cplx_a;
  wire [2*NB-1:0] w_cplx_b;
  wire [2*NB-1:0] w_cplx_c;
  wire [2*NB-1:0] w_cplx_d;
  wire [2*NB-1:0] w_cplx_out;

  fft1d_r2_mem #(
    .N            (N),
    .NB           (NB)
  ) u_mem (
    .i_clock      (i_clock),
    .i_load_en    (i_load_en),
    .i_load_addr  (i_load_addr),
    .i_load_data  (i_cplx_sample),
    .i_btfly_en   (i_btfly_en),
    .i_addr_a     (i_idx_a),
    .i_addr_b     (i_idx_b),
    .i_cplx_a     (w_cplx_c),
    .i_cplx_b     (w_cplx_d),
    .o_cplx_a     (w_cplx_a),
    .o_cplx_b     (w_cplx_b),
    .i_out_addr   (i_out_addr),
    .o_cplx_out   (w_cplx_out)
  );

  // --------------------------------------------------------------- mariposa
  btfly_r2 #(
    .NB   (NB),
    .NBF  (NBF)
  ) u_btfly (
    .i_cplx_a  (w_cplx_a),
    .i_cplx_b  (w_cplx_b),
    .i_cplx_w  ({tw_im[i_idx_tw], tw_re[i_idx_tw]}),
    .o_cplx_c  (w_cplx_c),
    .o_cplx_d  (w_cplx_d)
  );

  // ---------------------------------------------------------------- salidas
  // El complejo sale empaquetado tal cual viene de la memoria; no hace falta
  // desempaquetar y volver a empaquetar.
  assign o_cplx_sample = i_out_en ? w_cplx_out : {(2*NB){1'b0}};

endmodule
`endif
`timescale 1ns/1ps
`ifndef FFT1D_R2_CORE_V
`define FFT1D_R2_CORE_V

`include "btfly_r2.v"

// -----------------------------------------------------------------------------
// fft1d_r2_dp -- plano de DATOS de la FFT iterativa.
//
// Memorias, ROM de twiddles y la mariposa completa. No tiene FSM ni contadores:
// recibe direcciones y enables desde fft1d_r2_ctrl. Eso deja el reporte de
// recursos del calculo (DSP, BRAM) separado del control.
//
// Punto fijo Q(NBI.NBF). Toda la cuantizacion se hace instanciando cast:
//   - dentro de cmul, al requantizar el producto complejo
//   - en las cuatro ramas de la mariposa, al aplicar el *1/2
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

  // muestra de entrada y su escritura
  input  wire signed [NB-1:0]  i_re,
  input  wire signed [NB-1:0]  i_im,
  input  wire                  i_load_en,
  input  wire [$clog2(N)-1:0]  i_load_addr,

  // mariposa
  input  wire                  i_btfly_en,
  input  wire [$clog2(N)-1:0]  i_idx_a,
  input  wire [$clog2(N)-1:0]  i_idx_b,
  input  wire [$clog2(N)-2:0]  i_idx_tw,

  // descarga
  input  wire [$clog2(N)-1:0]  i_out_addr,
  input  wire                  i_out_en,
  output wire signed [NB-1:0]  o_re,
  output wire signed [NB-1:0]  o_im
);

  localparam int NH    = N / 2;

  localparam RE    = 0;
  localparam IM    = 1;

  // ------------------------------------------------------------ ROM twiddles
  reg signed [NB-1:0] tw_re [0:NH-1];
  reg signed [NB-1:0] tw_im [0:NH-1];

  // Con TW_FROM_FILE=0 el testbench carga la ROM por referencia jerarquica
  // (evita el warning de archivo inexistente en el flujo vm). Para sintesis
  // se deja en 1: Vivado usa el $readmemh para inicializar la BRAM/ROM.
  generate
    if (TW_FROM_FILE) begin : gen_tw_init
      initial begin
        $readmemh(TW_RE_FILE, tw_re);
        $readmemh(TW_IM_FILE, tw_im);
      end
    end
  endgenerate

  // --------------------------------------------------------------- memorias
  reg signed [NB-1:0] mem_re [0:N-1];
  reg signed [NB-1:0] mem_im [0:N-1];

  // ------------------------------------------------- operandos de la mariposa
  wire signed [NB - 1 : 0] c [1 : 0];
  wire signed [NB - 1 : 0] d [1 : 0];

  btfly_r2 #(
    .NB   (NB),
    .NBF  (NBF)
  ) u_btfly_r2 (
    .i_a_re   (mem_re[i_idx_a]),
    .i_a_im   (mem_im[i_idx_a]),
    .i_b_re   (mem_re[i_idx_b]),
    .i_b_im   (mem_im[i_idx_b]),
    .i_w_re   (tw_re[i_idx_tw]),
    .i_w_im   (tw_im[i_idx_tw]),
    .o_c_re   (c[RE]),
    .o_c_im   (c[IM]),
    .o_d_re   (d[RE]),
    .o_d_im   (d[IM])
  );


  // ------------------------------------------------------ escritura de memoria
  // i_load_en e i_btfly_en son mutuamente excluyentes (estados distintos).
  always @(posedge i_clock) begin
    if (i_load_en) begin
      mem_re[i_load_addr] <= i_re;
      mem_im[i_load_addr] <= i_im;
    end
    if (i_btfly_en) begin
      mem_re[i_idx_a] <= c[RE];
      mem_im[i_idx_a] <= c[IM];
      mem_re[i_idx_b] <= d[RE];
      mem_im[i_idx_b] <= d[IM];
    end
  end

  // ---------------------------------------------------------------- salidas
  assign o_re = i_out_en ? mem_re[i_out_addr] : {NB{1'b0}};
  assign o_im = i_out_en ? mem_im[i_out_addr] : {NB{1'b0}};

endmodule
`endif
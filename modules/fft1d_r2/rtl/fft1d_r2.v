`timescale 1ns/1ps
`ifndef FFT1D_R2_V
`define FFT1D_R2_V
`include "fft1d_r2_FSM.v"
`include "fft1d_r2_core.v"
module fft1d_r2 #(
  parameter int    N   = 512,
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_RE_FILE = "fft1d_r2_tw_re.mem",
  parameter        TW_IM_FILE = "fft1d_r2_tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                 i_clock,
  input  wire                 i_rst,
  input  wire                 i_valid,
  input  wire [2*NB-1:0]      i_cplx_sample,
  output wire                 o_valid,
  output wire                 o_last,
  output wire [2*NB-1:0]      o_cplx_sample
);
  localparam int LOG2N = $clog2(N);
  wire             w_load_en, w_btfly_en, w_valid;
  wire [LOG2N-1:0] w_load_addr, w_idx_a, w_idx_b, w_out_addr;
  wire [LOG2N-2:0] w_idx_tw;

  fft1d_r2_FSM #(
    .N(N)
  ) u_fsm (
    .i_clock(i_clock), 
    .i_rst(i_rst), 
    .i_valid(i_valid),
    .o_load_en(w_load_en), 
    .o_load_addr(w_load_addr),
    .o_btfly_en(w_btfly_en), 
    .o_idx_a(w_idx_a), 
    .o_idx_b(w_idx_b), 
    .o_idx_tw(w_idx_tw),
    .o_out_addr(w_out_addr), 
    .o_valid(w_valid), 
    .o_last(o_last)
  );

  fft1d_r2_core #(
    .N(N), 
    .NB(NB), 
    .NBF(NBF),
    .TW_RE_FILE(TW_RE_FILE), 
    .TW_IM_FILE(TW_IM_FILE), 
    .TW_FROM_FILE(TW_FROM_FILE)
  ) u_core (
    .i_clock(i_clock), 
    .i_cplx_sample(i_cplx_sample),
    .i_load_en(w_load_en), 
    .i_load_addr(w_load_addr),
    .i_btfly_en(w_btfly_en), 
    .i_idx_a(w_idx_a), 
    .i_idx_b(w_idx_b), 
    .i_idx_tw(w_idx_tw),
    .i_out_addr(w_out_addr), 
    .i_out_en(w_valid), 
    .o_cplx_sample(o_cplx_sample)
  );

  assign o_valid = w_valid;
endmodule
`endif
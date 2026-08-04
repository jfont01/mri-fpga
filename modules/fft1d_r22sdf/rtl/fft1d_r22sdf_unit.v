`timescale 1ns/1ps
`default_nettype none
`ifndef FFT1D_R22SDF_UNIT_V
`define FFT1D_R22SDF_UNIT_V
`include "btfly_bf2i.v"
`include "btfly_bf2ii.v"
`include "cmul.v"
// -----------------------------------------------------------------------------
// fft1d_r22sdf_unit -- unidad radix-2^2 SDF: procesa DOS etapas radix-2 (un
// nivel radix-4) mas el multiplicador de twiddle que cierra el par.
//
//   i_din -> BF2II (linea M/2, twiddle trivial -j) -> reg
//         -> BF2I  (linea M/4)                     -> reg
//         -> multiplicador (twiddle no trivial)     -> reg -> o_dout
//
// El parametro M es la "resolucion de twiddle" de la unidad: la primera unidad
// de la cadena tiene M = N, la siguiente M = N/4, y asi hasta M = 4. La ultima
// (M = 4, LOG_M = 2) NO lleva multiplicador, de donde salen los log4(N)-1
// multiplicadores del radix-2^2.
//
// CONTROL
// -------
// A diferencia de un contador global unico, cada etapa lleva SU PROPIO contador,
// que arranca cuando los datos efectivamente llegan a esa etapa:
//
//   di_count  : cuenta las muestras que entran a la unidad.
//   bf1_count : arranca cuando di_count llega a M/2-1 (la primera etapa ya tiene
//               media linea cargada y empieza a producir).
//   bf2_count : arranca, registrado, cuando bf1_count llega a M/4-1.
//
// Esto es indispensable en un pipeline SDF: las etapas profundas procesan datos
// que salieron de las anteriores con latencia, y un contador global les daria un
// indice que no corresponde a la muestra que tienen en la mano.
//
//   bf1_bf = di_count[LOG_M-1]              fase de mariposa de la 1ra etapa
//   bf1_mj = (bf1_count[LOG_M-1:LOG_M-2]==3) twiddle trivial -j (2 bits, como
//                                            indica el paper)
//   bf2_bf = bf1_count[LOG_M-2]  (registrado, para alinear con el pipeline)
//
// TWIDDLE
// -------
//   tw_sel  = {bf2_count[LOG_M-2], bf2_count[LOG_M-1]}   cual de las 4 ramas
//   tw_num  = bf2_count << (LOG_N - LOG_M)
//   tw_addr = tw_num * tw_sel     (mod N por truncamiento)
//
// La multiplicacion tw_num * tw_sel implementa las cuatro rotaciones del nivel
// radix-4: sel=0 -> W^0 = 1 (se puentea el multiplicador), sel=1 -> W^n,
// sel=2 -> W^2n, sel=3 -> W^3n.
// -----------------------------------------------------------------------------
module fft1d_r22sdf_unit #(
  parameter int    N   = 64,          // tamano total de la FFT
  parameter int    M   = 64,          // resolucion de twiddle de esta unidad
  parameter int    NB  = 16,
  parameter int    NBF = 15,
  parameter        TW_RE_FILE = "tw_re.mem",
  parameter        TW_IM_FILE = "tw_im.mem",
  parameter bit    TW_FROM_FILE = 1'b1
)(
  input  wire                i_clock,
  input  wire                i_rst,
  input  wire                i_valid,
  input  wire [2*NB-1:0]     i_din,
  output wire                o_valid,
  output wire [2*NB-1:0]     o_dout
);

  localparam int LOG_N   = $clog2(N);
  localparam int LOG_M   = $clog2(M);
  localparam int DEPTH1  = 1 << (LOG_M - 1);
  localparam int DEPTH2  = 1 << (LOG_M - 2);
  localparam int ADDR_W1 = (LOG_M - 1 > 0) ? (LOG_M - 1) : 1;
  localparam int ADDR_W2 = (LOG_M - 2 > 0) ? (LOG_M - 2) : 1;
  localparam bit HAS_MULT = (LOG_M != 2);

  // ======================================================= 1ra etapa: BF2II
  reg [LOG_N-1:0] r_di_count;
  always @(posedge i_clock) begin
    if (i_rst) r_di_count <= {LOG_N{1'b0}};
    else       r_di_count <= i_valid ? (r_di_count + 1'b1) : {LOG_N{1'b0}};
  end

  wire w_bf1_bf = r_di_count[LOG_M-1];

  reg             r_bf1_sp_en;
  reg [LOG_N-1:0] r_bf1_count;

  wire w_bf1_start = (r_di_count == LOG_N'(DEPTH1 - 1));
  wire w_bf1_end   = (r_bf1_count == LOG_N'((1 << LOG_N) - 1));
  wire w_bf1_mj    = (r_bf1_count[LOG_M-1 -: 2] == 2'd3);

  wire [2*NB-1:0] w_bf1_sp;

  btfly_bf2ii #(
    .NB      (NB),
    .NBF     (NBF),
    .DEPTH   (DEPTH1),
    .ADDR_W  (ADDR_W1)
  ) u_bf2ii (
    .i_clock (i_clock),
    .i_rst   (i_rst),
    .i_bf    (w_bf1_bf),
    .i_mj    (w_bf1_mj),
    .i_din   (i_din),
    .o_sp    (w_bf1_sp)
  );

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_bf1_sp_en <= 1'b0;
      r_bf1_count <= {LOG_N{1'b0}};
    end
    else begin
      r_bf1_sp_en <= w_bf1_start ? 1'b1 : (w_bf1_end ? 1'b0 : r_bf1_sp_en);
      r_bf1_count <= r_bf1_sp_en ? (r_bf1_count + 1'b1) : {LOG_N{1'b0}};
    end
  end

  // registro de pipeline entre las dos mariposas
  reg [2*NB-1:0] r_bf1_do;
  always @(posedge i_clock) begin
    if (i_rst) r_bf1_do <= {(2*NB){1'b0}};
    else       r_bf1_do <= w_bf1_sp;
  end

  // ======================================================== 2da etapa: BF2I
  reg r_bf2_bf;
  always @(posedge i_clock) begin
    if (i_rst) r_bf2_bf <= 1'b0;
    else       r_bf2_bf <= r_bf1_count[LOG_M-2];
  end

  wire [2*NB-1:0] w_bf2_sp;

  btfly_bf2i #(
    .NB      (NB),
    .NBF     (NBF),
    .DEPTH   (DEPTH2),
    .ADDR_W  (ADDR_W2)
  ) u_bf2i (
    .i_clock (i_clock),
    .i_rst   (i_rst),
    .i_bf    (r_bf2_bf),
    .i_din   (r_bf1_do),
    .o_sp    (w_bf2_sp)
  );

  reg             r_bf2_sp_en;
  reg             r_bf2_start;
  reg [LOG_N-1:0] r_bf2_count;

  wire w_bf2_end = (r_bf2_count == LOG_N'((1 << LOG_N) - 1));

  /*
   * r_bf2_start DEBE resetearse: si arranca indefinido contamina r_bf2_sp_en y
   * de ahi r_bf2_count, o_valid y toda la salida (X que se propagan).
   */
  always @(posedge i_clock) begin
    if (i_rst) r_bf2_start <= 1'b0;
    else       r_bf2_start <= (r_bf1_count == LOG_N'(DEPTH2 - 1)) & r_bf1_sp_en;
  end

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_bf2_sp_en <= 1'b0;
      r_bf2_count <= {LOG_N{1'b0}};
    end
    else begin
      r_bf2_sp_en <= r_bf2_start ? 1'b1 : (w_bf2_end ? 1'b0 : r_bf2_sp_en);
      r_bf2_count <= r_bf2_sp_en ? (r_bf2_count + 1'b1) : {LOG_N{1'b0}};
    end
  end

  reg [2*NB-1:0] r_bf2_do;
  reg            r_bf2_do_en;
  always @(posedge i_clock) begin
    if (i_rst) r_bf2_do <= {(2*NB){1'b0}};
    else       r_bf2_do <= w_bf2_sp;
  end
  always @(posedge i_clock) begin
    if (i_rst) r_bf2_do_en <= 1'b0;
    else       r_bf2_do_en <= r_bf2_sp_en;
  end

  // ========================================================= multiplicador
  generate
    if (HAS_MULT) begin : gen_mult

      // tw_num se TRUNCA a LOG_N-2 bits: los dos bits altos del contador ya se
      // consumieron en tw_sel, y dejarlos en el numero desplazaria la direccion
      // en N/2 (lo que equivale a multiplicar el twiddle por -1).
      wire [1:0]       w_tw_sel  = {r_bf2_count[LOG_M-2], r_bf2_count[LOG_M-1]};
      wire [LOG_N-3:0] w_tw_num  = r_bf2_count << (LOG_N - LOG_M);
      wire [LOG_N-1:0] w_tw_addr = w_tw_num * w_tw_sel;

      reg signed [NB-1:0] tw_re_rom [0:N-1];
      reg signed [NB-1:0] tw_im_rom [0:N-1];

      if (TW_FROM_FILE) begin : gen_tw_init
        initial begin
          $readmemh(TW_RE_FILE, tw_re_rom);
          $readmemh(TW_IM_FILE, tw_im_rom);
        end
      end

      // ROM con lectura registrada (alineada con r_bf2_do)
      reg signed [NB-1:0] r_tw_re, r_tw_im;
      always @(posedge i_clock) begin
        if (i_rst) begin
          r_tw_re <= {NB{1'b0}};
          r_tw_im <= {NB{1'b0}};
        end
        else begin
          r_tw_re <= tw_re_rom[w_tw_addr];
          r_tw_im <= tw_im_rom[w_tw_addr];
        end
      end

      // Se puentea el multiplicador cuando el twiddle es W^0 = 1.
      reg r_mu_en;
      always @(posedge i_clock) begin
        if (i_rst) r_mu_en <= 1'b0;
        else       r_mu_en <= (w_tw_addr != {LOG_N{1'b0}});
      end

      wire signed [NB-1:0] w_mu_re, w_mu_im;

      cmul #(
        .NB_IN   (NB),
        .NBF_IN  (NBF),
        .NB_OUT  (NB),
        .NBF_OUT (NBF)
      ) u_cmul (
        .i_1_re  (r_tw_re),
        .i_1_im  (r_tw_im),
        .i_2_re  ($signed(r_bf2_do[0    +: NB])),
        .i_2_im  ($signed(r_bf2_do[NB   +: NB])),
        .o_re    (w_mu_re),
        .o_im    (w_mu_im)
      );

      reg [2*NB-1:0] r_mu_do;
      reg            r_mu_do_en;
      always @(posedge i_clock) begin
        if (i_rst) r_mu_do <= {(2*NB){1'b0}};
        else       r_mu_do <= r_mu_en ? {w_mu_im, w_mu_re} : r_bf2_do;
      end
      always @(posedge i_clock) begin
        if (i_rst) r_mu_do_en <= 1'b0;
        else       r_mu_do_en <= r_bf2_do_en;
      end

      assign o_dout  = r_mu_do;
      assign o_valid = r_mu_do_en;

    end
    else begin : gen_no_mult
      // ultima unidad del pipeline: sin multiplicador
      assign o_dout  = r_bf2_do;
      assign o_valid = r_bf2_do_en;
    end
  endgenerate

endmodule
`endif
`default_nettype wire
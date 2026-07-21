`timescale 1ns/1ps
`ifndef FFT1D_R2_FSM_V
`define FFT1D_R2_FSM_V
// -----------------------------------------------------------------------------
// fft1d_r2_ctrl -- plano de CONTROL de la FFT iterativa.
//
// Contiene la FSM, los contadores y toda la generacion de direcciones. No tiene
// datapath: no conoce NB ni NBF, solo indices. Eso permite verificar y
// sintetizar el control por separado del calculo.
//
// Estados:
//   LOADING : N ciclos con i_valid=1. Emite o_load_en y la direccion
//             bit-reversed o_load_addr.
//   COMPUTE : LOG2N x N/2 ciclos. Emite o_btfly_en y el trio
//             (o_idx_a, o_idx_b, o_idx_tw).
//   OUTPUT  : N ciclos. Emite o_valid y la direccion de lectura o_out_addr.
// -----------------------------------------------------------------------------
module fft1d_r2_FSM #(
  parameter int N = 512
)(
  input  wire                  i_clock,
  input  wire                  i_rst,      
  input  wire                  i_valid,

  // carga de la muestra de entrada
  output wire                  o_load_en,
  output wire [$clog2(N)-1:0]  o_load_addr,

  // mariposa en curso
  output wire                  o_btfly_en,
  output wire [$clog2(N)-1:0]  o_idx_a,
  output wire [$clog2(N)-1:0]  o_idx_b,
  output wire [$clog2(N)-2:0]  o_idx_tw,

  // descarga del resultado
  output wire [$clog2(N)-1:0]  o_out_addr,
  output wire                  o_valid,
  output wire                  o_last
);

  localparam int LOG2N = $clog2(N);
  localparam int NH    = N / 2;

  localparam int NB_COUNT = LOG2N;
  localparam int NB_STAGE = $clog2(LOG2N);
  localparam int NB_BTFLY = LOG2N - 1;
  localparam int NB_IDX   = LOG2N;

  localparam [NB_COUNT-1:0] COUNT_MAX = NB_COUNT'(N - 1);
  localparam [NB_STAGE-1:0] STAGE_MAX = NB_STAGE'(LOG2N - 1);
  localparam [NB_BTFLY-1:0] BTFLY_MAX = NB_BTFLY'(NH - 1);

  localparam [1:0] LOADING = 2'd0;
  localparam [1:0] COMPUTE = 2'd1;
  localparam [1:0] OUTPUT  = 2'd2;

  reg [1:0]           r_state;
  reg [NB_COUNT-1:0]  r_count;   // LOADING y OUTPUT (0..N-1)
  reg [NB_STAGE-1:0]  r_stage;   // COMPUTE (0..LOG2N-1)
  reg [NB_BTFLY-1:0]  r_btfly;   // COMPUTE (0..N/2-1)

  function automatic [LOG2N-1:0] bitrev(input [LOG2N-1:0] x);
    integer b;
    begin
      bitrev = {LOG2N{1'b0}};
      for (b = 0; b < LOG2N; b = b + 1) bitrev[b] = x[LOG2N-1-b];
    end
  endfunction

  // ------------------------------------------------ generacion de direcciones
  wire [NB_IDX-1:0]   half  = NB_IDX'(1) << r_stage;                    // 1..N/2
  wire [NB_BTFLY-1:0] pos   = r_btfly & NB_BTFLY'(half - NB_IDX'(1));   // 0..half-1
  wire [NB_BTFLY-1:0] block = r_btfly >> r_stage;                       // 0..N/2-1

  assign o_load_addr = bitrev(r_count);
  assign o_idx_a      = (NB_IDX'(block) << (r_stage + 1)) + NB_IDX'(pos);
  assign o_idx_b      = o_idx_a + half;

  // idx_tw = pos * (N / 2^(stage+1)) = pos << (LOG2N-1-stage).
  // Es un desplazamiento, no una multiplicacion: N/2^(stage+1) siempre es
  // potencia de dos. Indexa una ROM de NH entradas -> NB_BTFLY bits.
  assign o_idx_tw    = NB_BTFLY'(pos << (NB_STAGE'(LOG2N - 1) - r_stage));

  assign o_out_addr  = r_count;

  assign o_load_en   = (r_state == LOADING) && i_valid;
  assign o_btfly_en  = (r_state == COMPUTE);
  assign o_valid     = (r_state == OUTPUT);
  assign o_last      = o_valid && (r_count == COUNT_MAX);

  // ----------------------------------------------------------------- FSM/seq
  always @(posedge i_clock) begin
    if (i_rst) begin
      r_state <= LOADING;
      r_count <= {NB_COUNT{1'b0}};
      r_stage <= {NB_STAGE{1'b0}};
      r_btfly <= {NB_BTFLY{1'b0}};
    end
    else begin
      case (r_state)

        LOADING: begin
          if (i_valid) begin
            if (r_count == COUNT_MAX) begin
              r_state <= COMPUTE;
              r_count <= {NB_COUNT{1'b0}};
              r_stage <= {NB_STAGE{1'b0}};
              r_btfly <= {NB_BTFLY{1'b0}};
            end
            else r_count <= r_count + 1'b1;
          end
        end

        COMPUTE: begin
          if (r_btfly == BTFLY_MAX) begin
            r_btfly <= {NB_BTFLY{1'b0}};
            if (r_stage == STAGE_MAX) begin
              r_state <= OUTPUT;
              r_count <= {NB_COUNT{1'b0}};
              r_stage <= {NB_STAGE{1'b0}};
            end
            else r_stage <= r_stage + 1'b1;
          end
          else r_btfly <= r_btfly + 1'b1;
        end

        OUTPUT: begin
          if (r_count == COUNT_MAX) begin
            r_state <= LOADING;
            r_count <= {NB_COUNT{1'b0}};
          end
          else r_count <= r_count + 1'b1;
        end

        default: begin
          r_state <= LOADING;
          r_count <= {NB_COUNT{1'b0}};
        end
      endcase
    end
  end

endmodule
`endif
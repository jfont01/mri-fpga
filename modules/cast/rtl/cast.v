`timescale 1ns/1ps
`ifndef CAST_V
`define CAST_V
module cast
#(
  parameter int NB_IN  = 8,
  parameter int NBF_IN = 5,
  parameter int NB_OUT = 6,
  parameter int NBF_OUT = 4,
  parameter bit ROUND_MODE = 1'b1,
  parameter bit ROUND_CONV = 1'b0   // 1: ties to even (convergent); needs ROUND_MODE = 1
)
(
  input  wire signed [NB_IN  - 1 : 0] i_word,
  output wire signed [NB_OUT - 1 : 0] o_word
);

  localparam int LSB_TO_DROP = NBF_IN - NBF_OUT;
  localparam int NBI_IN      = NB_IN  - NBF_IN;
  localparam int NBI_OUT     = NB_OUT - NBF_OUT;
  localparam int NBI_DIFF    = NBI_IN - NBI_OUT;

  localparam int W_BASE      = NB_IN + 1;
  localparam int W_SHIFT     = W_BASE + 1;

  localparam signed [NB_OUT-1:0] MAX_OUT = {1'b0, {(NB_OUT-1){1'b1}}};
  localparam signed [NB_OUT-1:0] MIN_OUT = {1'b1, {(NB_OUT-1){1'b0}}};

  /*
   * ROUND_CONV only changes how ties are broken, so it needs ROUND_MODE = 1.
   * Double $fatal: Vivado and XSim honor the elaboration one; iverilog ignores
   * it silently, so the initial one stops it at time 0.
   */
`ifndef __SYNTHESIS__
  generate
    if (ROUND_CONV && !ROUND_MODE) begin : gen_err_conv
      $fatal(1, "cast: ROUND_CONV = 1 needs ROUND_MODE = 1");
      initial $fatal(1, "cast: ROUND_CONV = 1 needs ROUND_MODE = 1");
    end
  endgenerate
`endif

  /*
   * Saturation is needed when:
   *
   * 1. The output has fewer integer bits than the input:
   *      NBI_IN > NBI_OUT
   *
   * 2. The output has the same number of integer bits, but rounding can
   *    create a positive carry into the sign bit:
   *      NBI_IN == NBI_OUT && ROUND_MODE && LSB_TO_DROP > 0
   *
   * Example:
   *   Q8.7 -> Q4.3, round
   *   +0.9375 rounds to +1.000, but Q4.3 max is +0.875.
   */
  localparam bit RANGE_REDUCED = (NBI_DIFF > 0);

  localparam bit ROUND_CAN_OVERFLOW = (NBI_DIFF == 0) && (LSB_TO_DROP > 0) && ROUND_MODE;

  localparam bit NEED_SAT = RANGE_REDUCED || ROUND_CAN_OVERFLOW;

  localparam int UPPER_W = (W_SHIFT > NB_OUT) ? (W_SHIFT - NB_OUT) : 0;

  wire signed [W_SHIFT-1:0] shifted_word;
  
  generate
    if (LSB_TO_DROP > 0) begin : gen_drop_lsbs

      if (ROUND_MODE) begin : gen_round
        wire signed [W_BASE-1:0] extended_word;

        localparam signed [W_BASE-1:0] ONE = {{(W_BASE-1){1'b0}}, 1'b1};

        localparam signed [W_BASE-1:0] BIAS = ONE <<< (LSB_TO_DROP - 1);

        localparam signed [W_SHIFT-1:0] BIAS_EXT = {1'b0, BIAS};

        assign extended_word = {i_word[NB_IN-1], i_word};

        if (ROUND_CONV) begin : gen_conv
          /*
           * Convergent rounding (round half to even). With m = floor(x / 2^LSB_TO_DROP)
           * and r the dropped bits:
           *
           *   m + (r > half) + (r == half) * lsb(m)  =  m + round_bit & (sticky | lsb(m))
           *
           * round_bit is the MSB of r and sticky the OR of the rest of r.
           */
          localparam [W_BASE-1:0] STICKY_MASK = BIAS - 1'b1;

          wire signed [W_SHIFT-1:0] floor_word;
          wire                      round_bit;
          wire                      sticky;
          wire                      carry;

          assign floor_word = $signed({extended_word[W_BASE-1], extended_word}) >>> LSB_TO_DROP;
          assign round_bit  = extended_word[LSB_TO_DROP-1];
          assign sticky     = |(extended_word & STICKY_MASK);
          assign carry      = round_bit & (sticky | extended_word[LSB_TO_DROP]);

          assign shifted_word = floor_word + $signed({1'b0, carry});
        end
        else begin : gen_half_up
          assign shifted_word = ($signed({extended_word[W_BASE-1], extended_word}) + BIAS_EXT) >>> LSB_TO_DROP;
        end

      end
      else begin : gen_trunc

        assign shifted_word = $signed({i_word[NB_IN-1], i_word[NB_IN-1], i_word}) >>> LSB_TO_DROP;

      end

    end
    else begin : gen_no_shift

      assign shifted_word = $signed({i_word[NB_IN-1], i_word[NB_IN-1], i_word});

    end
  endgenerate

  generate
    if (NEED_SAT && (UPPER_W > 0)) begin : gen_out_sat
      wire [UPPER_W-1:0] upper;
      wire sat_flag;

      assign upper = shifted_word[W_SHIFT-1:NB_OUT];

      assign sat_flag = |(upper ^ {UPPER_W{shifted_word[NB_OUT-1]}});

      assign o_word = sat_flag ? (shifted_word[W_SHIFT-1] ? MIN_OUT : MAX_OUT) : shifted_word[NB_OUT-1:0];
    end else begin : gen_out_no_sat
      assign o_word = shifted_word[NB_OUT-1:0];
    end

  endgenerate

endmodule
`endif
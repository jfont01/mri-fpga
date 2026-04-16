module cast
#(
  parameter NB_IN = 8,
  parameter NBF_IN = 5,
  parameter NB_OUT = 6,
  parameter NBF_OUT = 4,
  parameter bit ROUND_MODE = 1'b1
)
(
  input  logic signed [NB_IN - 1 : 0]  i_word,
  output logic signed [NB_OUT - 1 : 0] o_word
);

  localparam LSB_TO_DROP = NBF_IN - NBF_OUT;
  localparam NBI_IN      = NB_IN - NBF_IN;
  localparam NBI_OUT     = NB_OUT - NBF_OUT;
  localparam NBI_DIFF    = NBI_IN - NBI_OUT;
  localparam W_BASE      = NB_IN + 1;
  localparam W_SHIFT     = W_BASE + 1;

  localparam logic signed [NB_OUT-1:0] MAX_OUT = {1'b0, {(NB_OUT-1){1'b1}}};
  localparam logic signed [NB_OUT-1:0] MIN_OUT = {1'b1, {(NB_OUT-1){1'b0}}};

  logic signed [W_SHIFT-1:0] shifted_word;

  generate
    if (LSB_TO_DROP > 0) begin : gen_drop_lsbs
      if (ROUND_MODE) begin : gen_round
        logic signed [W_BASE-1:0] extended_word;

        localparam logic signed [W_BASE-1:0] ONE      = {{(W_BASE-1){1'b0}}, 1'b1};
        localparam logic signed [W_BASE-1:0] BIAS     = ONE <<< (LSB_TO_DROP - 1);
        localparam logic signed [W_SHIFT-1:0] BIAS_EXT = {1'b0, BIAS};

        assign extended_word = {i_word[NB_IN-1], i_word};
        assign shifted_word  = ($signed({extended_word[W_BASE-1], extended_word}) + BIAS_EXT) >>> LSB_TO_DROP;
      end
      else begin : gen_trunc
        assign shifted_word = $signed({i_word[NB_IN-1], i_word[NB_IN-1], i_word}) >>> LSB_TO_DROP;
      end
    end
    else begin : gen_no_shift
      assign shifted_word = $signed({i_word[NB_IN-1], i_word[NB_IN-1], i_word});
    end
  endgenerate

  localparam UPPER_W = (W_SHIFT > NB_OUT) ? (W_SHIFT - NB_OUT) : 0;
  logic sat_flag;

  generate
    if (UPPER_W > 0) begin : gen_sat
      logic [UPPER_W-1:0] upper;

      assign upper    = shifted_word[W_SHIFT-1 : NB_OUT];
      assign sat_flag = |(upper ^ {UPPER_W{shifted_word[NB_OUT-1]}});
    end
    else begin : gen_no_sat
      assign sat_flag = 1'b0;
    end
  endgenerate

    generate
    if (NBI_DIFF > 0) begin : gen_out_sat
        always_comb begin
        if (sat_flag) begin
            o_word = shifted_word[W_SHIFT-1] ? MIN_OUT : MAX_OUT;
        end
        else begin
            o_word = shifted_word[NB_OUT-1:0];
        end
        end
    end
    else begin : gen_out_no_sat
        always_comb begin
        o_word = shifted_word[NB_OUT-1:0];
        end
    end
    endgenerate

endmodule
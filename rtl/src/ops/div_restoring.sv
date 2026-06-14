`timescale 1ns/1ps

module div_restoring #(
    parameter integer NB_NUM        = 16,
    parameter integer NBF_NUM       = 15,

    parameter integer NB_DEN        = 16,
    parameter integer NBF_DEN       = 15,
    
    parameter integer NB_QUOTIENT   = 16,
    parameter integer NBF_QUOTIENT  = 15
)
(
    input  wire                            i_clock,
    input  wire                            i_rst,
    input  wire                            i_start,
    input  wire signed [NB_NUM-1:0]        i_num,
    input  wire signed [NB_DEN-1:0]        i_den,

    output wire signed [NB_QUOTIENT-1:0]   o_quotient,
    output wire                            o_ready,
    output wire                            o_busy
);

    localparam integer SHIFT           = NBF_QUOTIENT + NBF_DEN - NBF_NUM;
    localparam integer NB_QUOTIENT_INT = NB_NUM + SHIFT;
    localparam integer NB_REMAINDER    = NB_DEN + 1;
    localparam integer NB_COUNTER      = $clog2(NB_QUOTIENT_INT + 1);



    wire                              sign_num_w;
    wire                              sign_den_w;
    wire                              sign_q_w;
    wire        [NB_NUM-1:0]          num_abs_w;
    wire        [NB_DEN-1:0]          den_abs_w;
    wire        [NB_QUOTIENT_INT-1:0] num_scaled_w;
    wire        [NB_QUOTIENT_INT-1:0] quotient_shift_w;
    wire        [NB_REMAINDER-1:0]    partial_remainder_shift_w;
    wire signed [NB_REMAINDER:0]      trial_difference_w;

    reg         [NB_COUNTER-1:0]      counter_r;
    reg         [NB_QUOTIENT_INT-1:0] quotient_r;
    reg         [NB_REMAINDER-1:0]    partial_remainder_r;
    reg                               busy_r;
    reg                               ready_r;

    // -------- agregado mínimo para trunc negativo + saturación --------
    wire                              trunc_neg_fix_w;
    wire        [NB_QUOTIENT_INT:0]   quotient_mag_w;
    wire signed [NB_QUOTIENT_INT:0]   quotient_full_w;
    // -----------------------------------------------------------------

    assign sign_num_w = i_num[NB_NUM-1];
    assign sign_den_w = i_den[NB_DEN-1];
    assign sign_q_w   = sign_num_w ^ sign_den_w;

    assign num_abs_w = sign_num_w ? -i_num : i_num;
    assign den_abs_w = sign_den_w ? -i_den : i_den;

    assign num_scaled_w = num_abs_w << SHIFT;

    assign partial_remainder_shift_w = {partial_remainder_r[NB_REMAINDER-2:0], quotient_r[NB_QUOTIENT_INT-1]};
    assign quotient_shift_w          = {quotient_r[NB_QUOTIENT_INT-2:0], 1'b0};

    assign trial_difference_w = $signed({1'b0, partial_remainder_shift_w}) - $signed({1'b0, den_abs_w});

    always_ff @(posedge i_clock) begin
        if (i_rst) begin
            counter_r           <= '0;
            quotient_r          <= '0;
            partial_remainder_r <= '0;
            busy_r              <= 1'b0;
            ready_r             <= 1'b0;
        end else begin
            ready_r <= 1'b0;

            if (i_start && !busy_r) begin
                counter_r           <= '0;
                quotient_r          <= num_scaled_w;
                partial_remainder_r <= '0;
                busy_r              <= 1'b1;
                ready_r             <= 1'b0;

            end else if (busy_r) begin
                if (trial_difference_w >= 0) begin
                    partial_remainder_r <= trial_difference_w[NB_REMAINDER-1:0];
                    quotient_r          <= {quotient_shift_w[NB_QUOTIENT_INT-1:1], 1'b1};
                end else begin
                    partial_remainder_r <= partial_remainder_shift_w;
                    quotient_r          <= quotient_shift_w;
                end

                if (counter_r == NB_QUOTIENT_INT-1) begin
                    busy_r  <= 1'b0;
                    ready_r <= 1'b1;
                end else begin
                    counter_r <= counter_r + 1'b1;
                end
            end
        end
    end

    assign trunc_neg_fix_w = sign_q_w && (partial_remainder_r != '0);
    assign quotient_mag_w  = {1'b0, quotient_r} + trunc_neg_fix_w;

    assign quotient_full_w = sign_q_w ? -$signed(quotient_mag_w) :  $signed(quotient_mag_w);

    cast #(
        .NB_IN   (NB_QUOTIENT_INT + 1),
        .NBF_IN  (NBF_QUOTIENT),
        .NB_OUT  (NB_QUOTIENT),
        .NBF_OUT (NBF_QUOTIENT),
        .ROUND_MODE (1'b0)
    ) u_round_sat_quotient (
        .i_word (quotient_full_w),
        .o_word (o_quotient)
    );

    assign o_ready = ready_r;
    assign o_busy  = busy_r;

endmodule
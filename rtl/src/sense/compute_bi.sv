module compute_Aij #(
    parameter NB_S  = 16,
    parameter NBF_S = 15,
    parameter NB_Y  = 28,
    parameter NBF_Y = 26,
    parameter NB_B  = 28,
    parameter NBF_B = 26,
    parameter L     = 4
)(
    input  logic i_clock,
    input  logic i_rst,
    input  logic i_start,

    input  logic signed [NB_S-1:0] s0_re [L-1:0],
    input  logic signed [NB_S-1:0] s0_im [L-1:0],
    input  logic signed [NB_S-1:0] s1_re [L-1:0],
    input  logic signed [NB_S-1:0] s1_im [L-1:0],
    input  logic signed [NB_Y-1:0] y_re [L-1:0],
    input  logic signed [NB_Y-1:0] y_im [L-1:0],

    output logic signed [NB_B-1:0] b0_re,
    output logic signed [NB_B-1:0] b0_im,
    output logic signed [NB_B-1:0] b1_re,
    output logic signed [NB_B-1:0] b1_im,

    output logic o_valid
);

    localparam int NB_ACC  = NB_B + 1;
    localparam int NBF_ACC = NBF_B;

    logic [NB_L-1:0] l_ptr_r;
    logic running_r;

    logic signed [NB_B - 1 : 0  ]       p0_re;
    logic signed [NB_B - 1 : 0  ]       p0_im;
    logic signed [NB_B - 1 : 0  ]       p1_re;
    logic signed [NB_B - 1 : 0  ]       p1_im;

    logic signed [NB_ACC - 1 : 0]       acc0_full;
    logic signed [NB_B - 1 : 0  ]       acc0_next;
    logic signed [NB_ACC - 1 : 0]       acc1_full;
    logic signed [NB_B - 1 : 0  ]       acc1_next;



    //! p0
        cmul #(
            .NB0_IN     (NB_S),
            .NBF0_IN    (NBF_S),
            .NB1_IN     (NB_Y),
            .NBF1_IN    (NBF_Y),
            .NB_OUT     (NB_B),
            .NBF_OUT    (NBF_B)
        ) u_cmul_p0 (
            .i_re_0     (s0_re[l_ptr_r]),
            .i_im_0     (-s0_im[l_ptr_r]),
            .i_re_1     (y_re[l_ptr_r]),
            .i_im_1     (y_im[l_ptr_r]),
            .o_re       (p0_re),
            .o_im       (p0_im)
        );

        assign acc01_re_full = A01_re + p01_re;
        assign acc01_im_full = A01_im + p01_im;

        cast #(
            .NB_IN  (NB_ACC),
            .NBF_IN (NBF_ACC),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_acc01_re (
            .i_word(acc01_re_full),
            .o_word(acc01_re_next)
        );

        cast #(
            .NB_IN  (NB_ACC),
            .NBF_IN (NBF_ACC),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_acc01_im (
            .i_word(acc01_im_full),
            .o_word(acc01_im_next)
        );

    //! A10
        assign A10_re = A01_re;
        assign A10_im = -A01_im;


    always_ff @(posedge i_clock) begin
        if (i_rst) begin
            l_ptr_r     <= '0;
            A00_re   <= '0;
            A11_re   <= '0;
            A01_re   <= '0;
            A01_im   <= '0;
            running_r <= 1'b0;
            o_valid   <= 1'b0;
        end else begin
            o_valid <= 1'b0;

            if (i_start) begin
                l_ptr_r     <= '0;
                A00_re   <= '0;
                A11_re   <= '0;
                A01_re   <= '0;
                A01_im   <= '0;
                running_r <= 1'b1;
            end else if (running_r) begin
                A00_re <= acc00_next;
                A11_re <= acc11_next;
                A01_re <= acc01_re_next;
                A01_im <= acc01_im_next;

                if (l_ptr_r == L-1) begin
                    running_r <= 1'b0;
                    o_valid   <= 1'b1;
                end else begin
                    l_ptr_r <= l_ptr_r + 1'b1;
                end
            end
        end
    end


endmodule
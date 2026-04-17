module compute_Aij #(
    parameter NB_S  = 16,
    parameter NBF_S = 15,
    parameter NB_A  = 28,
    parameter NBF_A = 26,
    parameter L     = 4
)(
    input  logic i_clock,
    input  logic i_rst,
    input  logic i_start,

    input  logic signed [NB_S-1:0] s0_re [L-1:0],
    input  logic signed [NB_S-1:0] s0_im [L-1:0],
    input  logic signed [NB_S-1:0] s1_re [L-1:0],
    input  logic signed [NB_S-1:0] s1_im [L-1:0],

    output logic signed [NB_A-1:0] A00_re,
    output logic signed [NB_A-1:0] A11_re,
    output logic signed [NB_A-1:0] A01_re,
    output logic signed [NB_A-1:0] A01_im,
    output logic signed [NB_A-1:0] A10_re,
    output logic signed [NB_A-1:0] A10_im,

    output logic o_valid
);

    localparam int NB_P  = 2*NB_S + 1;
    localparam int NBF_P = 2*NBF_S;
    localparam int NB_ACC  = NB_A + 1;
    localparam int NBF_ACC = NBF_A;
    localparam int NB_L    = $clog2(L);

    logic [NB_L-1:0] l_ptr_r;
    logic running_r;

    logic signed [NB_P - 1 : 0  ]       p00_full;
    logic signed [NB_A - 1 : 0  ]       p00;
    logic signed [NB_ACC - 1 : 0]       acc00_full;
    logic signed [NB_A - 1 : 0  ]       acc00_next;

    //! A00
        assign p00_full = s0_re[l_ptr_r] * s0_re[l_ptr_r] + s0_im[l_ptr_r] * s0_im[l_ptr_r];

        cast #(
            .NB_IN  (NB_P),
            .NBF_IN (NBF_P),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_p00 (
            .i_word(p00_full),
            .o_word(p00)
        );

        assign acc00_full = A00_re + p00;

        cast #(
            .NB_IN  (NB_ACC),
            .NBF_IN (NBF_ACC),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_acc00 (
            .i_word(acc00_full),
            .o_word(acc00_next)
        );


    //! A11
        logic signed [NB_P - 1 : 0  ]       p11_full;
        logic signed [NB_A - 1 : 0  ]       p11;
        logic signed [NB_ACC - 1 : 0]       acc11_full;
        logic signed [NB_A - 1 : 0  ]       acc11_next;
        assign p11_full = s1_re[l_ptr_r] * s1_re[l_ptr_r] + s1_im[l_ptr_r] * s1_im[l_ptr_r];
        cast #(
            .NB_IN  (NB_P),
            .NBF_IN (NBF_P),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_p11 (
            .i_word(p11_full),
            .o_word(p11)
        );

        assign acc11_full = A11_re + p11;

        cast #(
            .NB_IN  (NB_ACC),
            .NBF_IN (NBF_ACC),
            .NB_OUT (NB_A),
            .NBF_OUT(NBF_A)
        ) u_cast_acc11 (
            .i_word(acc11_full),
            .o_word(acc11_next)
        );


    //! A01
        logic signed [NB_A - 1 : 0  ]       p01_re;
        logic signed [NB_A - 1 : 0  ]       p01_im;
        logic signed [NB_ACC - 1 : 0]       acc01_re_full;
        logic signed [NB_ACC - 1 : 0]       acc01_im_full;
        logic signed [NB_A - 1 : 0]       acc01_re_next;
        logic signed [NB_A - 1 : 0]       acc01_im_next;
        cmul #(
            .NB0_IN     (NB_S),
            .NBF0_IN    (NBF_S),
            .NB1_IN     (NB_S),
            .NBF1_IN    (NBF_S),
            .NB_OUT     (NB_A),
            .NBF_OUT    (NBF_A)
        ) u_cmul_p01 (
            .i_re_0     (s0_re[l_ptr_r]),
            .i_im_0     (-s0_im[l_ptr_r]),
            .i_re_1     (s1_re[l_ptr_r]),
            .i_im_1     (s1_im[l_ptr_r]),
            .o_re       (p01_re),
            .o_im       (p01_im)
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
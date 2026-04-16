module compute_bi #(
    parameter NB_S  = 16,
    parameter NBF_S = 15,
    parameter NB_K  = 32,
    parameter NBF_K = 29,
    parameter NB_B  = 32,
    parameter NBF_B = 30,
    parameter L     = 4
)
(
    input  logic signed [NB_S - 1 : 0] s0_re [L - 1 : 0]    ,
    input  logic signed [NB_S - 1 : 0] s0_im [L - 1 : 0]    ,

    input  logic signed [NB_S - 1 : 0] s1_re [L - 1 : 0]    ,
    input  logic signed [NB_S - 1 : 0] s1_im [L - 1 : 0]    ,

    input logic signed [NB_K - 1 : 0] y0_re [L - 1 : 0]     ,


    output logic signed [NB_B - 1 : 0] b0_re                ,
    output logic signed [NB_B - 1 : 0] b0_im                , 
    output logic signed [NB_B - 1 : 0] b1_re                ,
    output logic signed [NB_B - 1 : 0] b1_im                ,
);
    localparam NB_FULL = NB_S + NB_K;
    localparam NBF_FULL = NBF_S + NBF_K;

    localparam NB_ACC = NB_B + 1 ;
    localparam NBF_ACC = NBF_B;
    // -------------------------------------------------------------------------
    // productos p0 y p1
    // -------------------------------------------------------------------------
    logic signed [NB_B - 1 : 0]     p0_re          [L - 1 : 0];
    logic signed [NB_B - 1 : 0]     p0_im          [L - 1 : 0];
    logic signed [NB_FULL - 1 : 0]  p0_full_re     [L - 1 : 0];
    logic signed [NB_FULL - 1 : 0]  p0_full_im     [L - 1 : 0];

    
    logic signed [NB_B - 1 : 0]     p1_re          [L - 1 : 0];
    logic signed [NB_B - 1 : 0]     p1_im          [L - 1 : 0];
    logic signed [NB_FULL - 1 : 0]  p1_full_re     [L - 1 : 0];
    logic signed [NB_FULL - 1 : 0]  p1_full_im     [L - 1 : 0];



    // -------------------------------------------------------------------------
    // acumuladores por etapa
    // -------------------------------------------------------------------------
    logic signed [NB_B - 1 : 0]     acc0_re        [L : 0];
    logic signed [NB_B - 1 : 0]     acc0_im        [L : 0];
    logic signed [NB_ACC - 1 : 0]   acc0_full_re   [L : 0];
    logic signed [NB_ACC - 1 : 0]   acc0_full_im   [L : 0];
    logic signed [NB_B - 1 : 0]     acc1_re        [L : 0];
    logic signed [NB_B - 1 : 0]     acc1_im        [L : 0];
    logic signed [NB_ACC - 1 : 0]   acc1_full_re   [L : 0];
    logic signed [NB_ACC - 1 : 0]   acc1_full_im   [L : 0];
                

    // -------------------------------------------------------------------------
    // etapa 0 de acumulación
    // -------------------------------------------------------------------------                    
    assign acc0_re[0] = '0;
    assign acc0_im[0] = '0;
    assign acc1_re[0] = '0;
    assign acc1_im[0] = '0;

    genvar k;
    generate
        for (k = 0; k < L; k = k + 1) begin : gen_compute_bi
            //! Partial products
                //! p0 = (s0.conj() * y0).cast(NB_B, NBF_B)
                    cmul #(
                        .NB0_IN  (NB_S),
                        .NBF0_IN (NBF_S),
                        .NB1_IN  (NB_K),
                        .NBF1_IN (NBF_K),
                        .NB_OUT (NB_B),
                        .NBF_OUT(NBF_B)
                    ) u_cmul_p0 (
                        .i_re_0 (s0_re[k]  ),
                        .i_im_0 (-s0_re[k] ),
                        .i_re_1 (y0_re[k]  ),
                        .i_im_1 (y0_im[k]  ),
                        .o_re   (p0_re[k]  ),
                        .o_im   (p0_im[k]  )
                    )

                //! p1 = (s1.conj() * y0).cast(NB_B, NBF_B)
                    cmul #(
                        .NB0_IN  (NB_S),
                        .NBF0_IN (NBF_S),
                        .NB1_IN  (NB_K),
                        .NBF1_IN (NBF_K),
                        .NB_OUT (NB_B),
                        .NBF_OUT(NBF_B)
                    ) u_cmul_p1 (
                        .i_re_0 (s1_re[k]   ),
                        .i_im_0 (-s1_re[k]  ),
                        .i_re_1 (y0_re[k]   ),
                        .i_im_1 (y0_im[k]   ),
                        .o_re   (p1_re[k]   ),
                        .o_im   (p1_im[k]   )
                    )

            //! Accumulators
                //! b0 = (b0 + p0).cast(NB_B, NBF_B)
                    assign acc0_full_re[k+1] = p0_re[k] + acc0_re[k];
                    cast #(
                        .NB_IN   (NB_ACC),
                        .NBF_IN  (NBF_ACC),
                        .NB_OUT  (NB_B),
                        .NBF_OUT (NBF_B),
                        .ROUND_MODE(1'b1)
                    ) u_cast_acc0 (
                        .i_word (acc0_full_re[k+1]),
                        .o_word (acc0_re[k+1])
                    );

                //! b1 = (b1 + p1).cast(NB_B, NBF_B)
                    assign acc1_full_re[k+1] = p1_re[k] + acc1_re[k];
                    cast #(
                        .NB_IN   (NB_ACC),
                        .NBF_IN  (NBF_ACC),
                        .NB_OUT  (NB_B),
                        .NBF_OUT (NBF_B),
                        .ROUND_MODE(1'b1)
                    ) u_cast_acc1 (
                        .i_word (acc1_full_re[k+1]),
                        .o_word (acc1_re[k+1])
                    );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // salidas
    // -------------------------------------------------------------------------

    assign b0_re = acc0_re[L];     
    assign b0_im = acc0_im[L];    

    assign b1_re = acc1_re[L];     
    assign b1_im = acc1_im[L]; 


endmodule
`timescale 1ns/1ps
`default_nettype none

`ifndef FFT1D_R2_N
`define FFT1D_R2_N 512
`endif
`ifndef FFT1D_R2_NB
`define FFT1D_R2_NB 16
`endif
`ifndef FFT1D_R2_NBF
`define FFT1D_R2_NBF 15
`endif
`ifndef FFT1D_R2_TW_RE_FILE
`define FFT1D_R2_TW_RE_FILE "fft1d_r2_tw_re.mem"
`endif
`ifndef FFT1D_R2_TW_IM_FILE
`define FFT1D_R2_TW_IM_FILE "fft1d_r2_tw_im.mem"
`endif

module fft1d_r2_tb;

    localparam int N   = `FFT1D_R2_N;
    localparam int NB  = `FFT1D_R2_NB;
    localparam int NBF = `FFT1D_R2_NBF;

    localparam string TW_RE_FILE = `FFT1D_R2_TW_RE_FILE;
    localparam string TW_IM_FILE = `FFT1D_R2_TW_IM_FILE;

    /*
     * EMPAQUETADO EN LA FRONTERA
     * --------------------------
     * El DUT usa puertos complejos empaquetados ({imag, real} en 2*NB bits),
     * pero los vectores se siguen manejando por COMPONENTE: i_re/i_im y
     * o_re/o_im, un .dat cada uno.
     *
     * Es deliberado. El modelo C++ conserva sus puertos separados (lo que le
     * permite compilar en modo DOUBLE, donde no hay patron de bits que
     * empaquetar), y el reporte de vector_match muestra enteros con signo
     * legibles en vez de un blob de 32 bits sin signo. El empaquetado queda
     * confinado a estas cuatro lineas.
     */
    logic                 i_clock;
    logic                 i_rst;
    logic                 i_valid;
    logic signed [NB-1:0] i_re;
    logic signed [NB-1:0] i_im;

    wire  [2*NB-1:0]      i_cplx_sample;
    wire  [2*NB-1:0]      o_cplx_sample;

    logic                 o_valid;
    logic                 o_last;
    wire  signed [NB-1:0] o_re;
    wire  signed [NB-1:0] o_im;

    assign i_cplx_sample = {i_im, i_re};
    assign o_re          = $signed(o_cplx_sample[0  +: NB]);
    assign o_im          = $signed(o_cplx_sample[NB +: NB]);

    string case_dir;
    int    n_cycles;

    string in_dir, act_dir, act_out_dir, act_csv_path;
    string p__i_valid, p__i_re, p__i_im, p__o_valid, p__o_last, p__o_re, p__o_im;
    string p__r_state, p__r_count, p__r_stage, p__r_btfly;
    string waves_path;

    int fd__i_valid, fd__i_re, fd__i_im, fd__o_valid, fd__o_last, fd__o_re, fd__o_im, fd__csv;
    int fd__r_state, fd__r_count, fd__r_stage, fd__r_btfly;
    int s__i_valid, s__i_re, s__i_im;

    // ------------------------------------------------------------------- DUT
    fft1d_r2 #(
        .N             (N),
        .NB            (NB),
        .NBF           (NBF),
        .TW_RE_FILE    (TW_RE_FILE),
        .TW_IM_FILE    (TW_IM_FILE)
    ) u_fft (
        .i_clock       (i_clock),
        .i_rst         (i_rst),
        .i_valid       (i_valid),
        .i_cplx_sample (i_cplx_sample),
        .o_valid       (o_valid),
        .o_last        (o_last),
        .o_cplx_sample (o_cplx_sample)
    );

    // -------------------------------------------------------------- clock
    initial i_clock = 1'b0;
    always #5 i_clock = ~i_clock;

    function real fixed_to_real(input logic signed [NB-1:0] raw);
        fixed_to_real = $itor($signed(raw)) / (2.0 ** NBF);
    endfunction

    initial begin
        if (!$value$plusargs("CASE_DIR=%s", case_dir))
            $fatal(1, "[fft1d_r2_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[fft1d_r2_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[fft1d_r2_tb] N_CYCLES invalido=%0d", n_cycles);

        in_dir       = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir      = {case_dir, "/simulation/vectors/actual"};
        act_out_dir  = {act_dir,  "/out_ports"};
        act_csv_path = {act_dir,  "/out_ports.csv"};

        p__i_valid = {in_dir, "/i_valid.dat"};
        p__i_re    = {in_dir, "/i_re.dat"};
        p__i_im    = {in_dir, "/i_im.dat"};
        p__o_valid = {act_out_dir, "/o_valid.dat"};
        p__o_last  = {act_out_dir, "/o_last.dat"};
        p__o_re    = {act_out_dir, "/o_re.dat"};
        p__o_im    = {act_out_dir, "/o_im.dat"};

        /*
         * Registros internos del control, para localizar en que ciclo exacto
         * el RTL se aparta del modelo.
         *
         * OJO CON EL NOMBRE: add_out_reg_o() del lado C++ le pega el sufijo
         * ".o" al nombre, asi que el archivo esperado es "r_state.o.dat".
         * Si aca lo llamamos "r_state.dat" el vm falla por artefacto faltante,
         * aunque los valores sean correctos.
         */
        p__r_state = {act_out_dir, "/r_state.o.dat"};
        p__r_count = {act_out_dir, "/r_count.o.dat"};
        p__r_stage = {act_out_dir, "/r_stage.o.dat"};
        p__r_btfly = {act_out_dir, "/r_btfly.o.dat"};

        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, fft1d_r2_tb);
            $display("[fft1d_r2_tb] waves: %s", waves_path);
        end

        fd__i_valid = $fopen(p__i_valid, "r");
        fd__i_re    = $fopen(p__i_re,    "r");
        fd__i_im    = $fopen(p__i_im,    "r");
        if (fd__i_valid==0 || fd__i_re==0 || fd__i_im==0)
            $fatal(1, "[fft1d_r2_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd__o_valid = $fopen(p__o_valid, "w");
        fd__o_last  = $fopen(p__o_last,  "w");
        fd__o_re    = $fopen(p__o_re,    "w");
        fd__o_im    = $fopen(p__o_im,    "w");
        fd__csv     = $fopen(act_csv_path, "w");
        fd__r_state = $fopen(p__r_state, "w");
        fd__r_count = $fopen(p__r_count, "w");
        fd__r_stage = $fopen(p__r_stage, "w");
        fd__r_btfly = $fopen(p__r_btfly, "w");
        if (fd__o_valid==0 || fd__o_last==0 || fd__o_re==0 || fd__o_im==0 || fd__csv==0
            || fd__r_state==0 || fd__r_count==0 || fd__r_stage==0 || fd__r_btfly==0)
            $fatal(1, "[fft1d_r2_tb] no pude abrir salidas en %s", act_out_dir);

        $display("[fft1d_r2_tb] CASE_DIR = %s", case_dir);
        $display("[fft1d_r2_tb] N_CYCLES = %0d  N=%0d NB=%0d NBF=%0d", n_cycles, N, NB, NBF);

        $fwrite(fd__csv, "cycle,o_valid,o_last,o_re,o_im,r_state,r_count,r_stage,r_btfly\n");

        // ---------------------------------------- reset (= init del modelo)
        i_rst   = 1'b1;
        i_valid = 1'b0;
        i_re    = '0;
        i_im    = '0;
        @(negedge i_clock);
        @(posedge i_clock);   // el modelo arranca en LOADING sin consumir ciclo
        i_rst = 1'b0;

        // ------------------------------------------------- lazo por ciclo
        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            @(negedge i_clock);

            if ($fscanf(fd__i_valid, "%d", s__i_valid) != 1)
                $fatal(1, "[fft1d_r2_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_re, "%d", s__i_re) != 1)
                $fatal(1, "[fft1d_r2_tb] i_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_im, "%d", s__i_im) != 1)
                $fatal(1, "[fft1d_r2_tb] i_im: faltan muestras en cycle=%0d", cycle);

            i_valid = s__i_valid[0];
            i_re    = s__i_re[NB-1:0];
            i_im    = s__i_im[NB-1:0];

            @(posedge i_clock);
            #1;

            $fwrite(fd__r_state, "%0d\n", u_fft.u_fsm.r_state);
            $fwrite(fd__r_count, "%0d\n", u_fft.u_fsm.r_count);
            $fwrite(fd__r_stage, "%0d\n", u_fft.u_fsm.r_stage);
            $fwrite(fd__r_btfly, "%0d\n", u_fft.u_fsm.r_btfly);

            $fwrite(fd__o_valid, "%0d\n", o_valid);
            $fwrite(fd__o_last,  "%0d\n", o_last);
            $fwrite(fd__o_re,    "%0d\n", $signed(o_re));
            $fwrite(fd__o_im,    "%0d\n", $signed(o_im));
            $fwrite(fd__csv, "%0d,%0d,%0d,%.16g,%.16g,%0d,%0d,%0d,%0d\n",
                    cycle + 1, o_valid, o_last,
                    fixed_to_real(o_re), fixed_to_real(o_im),
                    u_fft.u_fsm.r_state, u_fft.u_fsm.r_count,
                    u_fft.u_fsm.r_stage, u_fft.u_fsm.r_btfly);
        end

        $fclose(fd__i_valid); $fclose(fd__i_re); $fclose(fd__i_im);
        $fclose(fd__o_valid); $fclose(fd__o_last); $fclose(fd__o_re);
        $fclose(fd__o_im); $fclose(fd__csv);
        $fclose(fd__r_state); $fclose(fd__r_count);
        $fclose(fd__r_stage); $fclose(fd__r_btfly);
        $display("[fft1d_r2_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire
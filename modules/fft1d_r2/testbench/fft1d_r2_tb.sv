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

    // Los paths del ROM llegan como defines. El VALOR del define debe incluir
    // las comillas (ver el JSON), porque la macro se expande como texto crudo:
    //   -D FFT1D_R2_TW_RE_FILE="C:/.../fft1d_r2_tw_re.mem"
    localparam string TW_RE_FILE = `FFT1D_R2_TW_RE_FILE;
    localparam string TW_IM_FILE = `FFT1D_R2_TW_IM_FILE;

    // ---------------------------------------------------------------- señales
    logic                 i_clock;
    logic                 i_rst;
    logic                 i_valid;
    logic signed [NB-1:0] i_re;
    logic signed [NB-1:0] i_im;
    logic                 o_valid;
    logic                 o_last;
    logic signed [NB-1:0] o_re;
    logic signed [NB-1:0] o_im;

    string case_dir;
    int    n_cycles;

    string in_dir, act_dir, act_out_dir, act_csv_path;
    string p__i_valid, p__i_re, p__i_im, p__o_valid, p__o_last, p__o_re, p__o_im;
    string waves_path;

    int fd__i_valid, fd__i_re, fd__i_im, fd__o_valid, fd__o_last, fd__o_re, fd__o_im, fd__csv;

    int s__i_valid, s__i_re, s__i_im;

    // ------------------------------------------------------------------- DUT
    fft1d_r2 #(
        .N          (N),
        .NB         (NB),
        .NBF        (NBF),
        .TW_RE_FILE (TW_RE_FILE),
        .TW_IM_FILE (TW_IM_FILE)
    ) u_fft (
        .i_clock (i_clock),
        .i_rst   (i_rst),
        .i_valid (i_valid),
        .i_re    (i_re),
        .i_im    (i_im),
        .o_valid (o_valid),
        .o_last  (o_last),
        .o_re    (o_re),
        .o_im    (o_im)
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

        in_dir      = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir     = {case_dir, "/simulation/vectors/actual"};
        act_out_dir = {act_dir,  "/out_ports"};
        act_csv_path= {act_dir,  "/out_ports.csv"};

        p__i_valid = {in_dir, "/i_valid.dat"};
        p__i_re    = {in_dir, "/i_re.dat"};
        p__i_im    = {in_dir, "/i_im.dat"};
        p__o_valid = {act_out_dir, "/o_valid.dat"};
        p__o_last  = {act_out_dir, "/o_last.dat"};
        p__o_re    = {act_out_dir, "/o_re.dat"};
        p__o_im    = {act_out_dir, "/o_im.dat"};

        /*
         * Waves (opcional): correr con +WAVES para generar un VCD.
         *   vvp sim.vvp +CASE_DIR=<dir> +N_CYCLES=<n> +WAVES
         * El archivo queda en <case_dir>/simulation/waves.vcd
         *
         * Nota: iverilog NO vuelca arrays (mem_re/mem_im/tw_re/tw_im) con
         * $dumpvars por defecto. Se agregan explicitamente algunas celdas
         * abajo si hace falta inspeccionarlas.
         */
        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, fft1d_r2_tb);
            $display("[fft1d_r2_tb] waves: %s", waves_path);
        end

        fd__i_valid = $fopen(p__i_valid, "r");
        fd__i_re = $fopen(p__i_re,    "r");
        fd__i_im = $fopen(p__i_im,    "r");
        if (fd__i_valid==0 || fd__i_re==0 || fd__i_im==0)
            $fatal(1, "[fft1d_r2_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd__o_valid  = $fopen(p__o_valid, "w");
        fd__o_last  = $fopen(p__o_last,  "w");
        fd__o_re = $fopen(p__o_re,    "w");
        fd__o_im = $fopen(p__o_im,    "w");
        fd__csv = $fopen(act_csv_path, "w");
        if (fd__o_valid==0 || fd__o_last==0 || fd__o_re==0 || fd__o_im==0 || fd__csv==0)
            $fatal(1, "[fft1d_r2_tb] no pude abrir salidas en %s", act_out_dir);

        $display("[fft1d_r2_tb] CASE_DIR = %s", case_dir);
        $display("[fft1d_r2_tb] N_CYCLES = %0d  N=%0d NB=%0d NBF=%0d", n_cycles, N, NB, NBF);
        $fwrite(fd__csv, "cycle,o_valid,o_last,o_re,o_im\n");

        // ------------------------------------------------- reset (= init del modelo)
        i_rst   = 1'b1;
        i_valid = 1'b0;
        i_re    = '0;
        i_im    = '0;
        @(negedge i_clock);
        @(posedge i_clock);   // el modelo arranca en LOADING sin consumir ciclo
        i_rst = 1'b0;

        // ----------------------------------------------------------- lazo por ciclo
        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            @(negedge i_clock);                       // entradas estables p/ el posedge
            if ($fscanf(fd__i_valid, "%d", s__i_valid) != 1)
                $fatal(1, "[fft1d_r2_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_re, "%d", s__i_re) != 1)
                $fatal(1, "[fft1d_r2_tb] i_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_im, "%d", s__i_im) != 1)
                $fatal(1, "[fft1d_r2_tb] i_im: faltan muestras en cycle=%0d", cycle);

            i_valid = s__i_valid[0];
            i_re    = s__i_re[NB-1:0];
            i_im    = s__i_im[NB-1:0];

            @(posedge i_clock);                       // paso del FSM para este ciclo
            #1;                                        // asienta la salida combinacional

            $fwrite(fd__o_valid,  "%0d\n", o_valid);
            $fwrite(fd__o_last,  "%0d\n", o_last);
            $fwrite(fd__o_re, "%0d\n", $signed(o_re));
            $fwrite(fd__o_im, "%0d\n", $signed(o_im));
            $fwrite(fd__csv, "%0d,%0d,%0d,%.16g,%.16g\n", cycle + 1, o_valid, o_last, fixed_to_real(o_re), fixed_to_real(o_im));
        end

        $fclose(fd__i_valid); $fclose(fd__i_re); $fclose(fd__i_im);
        $fclose(fd__o_valid); $fclose(fd__o_last); $fclose(fd__o_re); $fclose(fd__o_im); $fclose(fd__csv);
        $display("[fft1d_r2_tb] simulacion completa");
        $finish;
    end

endmodule

`default_nettype wire
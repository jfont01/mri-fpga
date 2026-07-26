`timescale 1ns/1ps
`default_nettype none


`ifndef FFT1D_R2SDF_N
`define FFT1D_R2SDF_N 512
`endif
`ifndef FFT1D_R2SDF_NB
`define FFT1D_R2SDF_NB 16
`endif
`ifndef FFT1D_R2SDF_NBF
`define FFT1D_R2SDF_NBF 15
`endif
`ifndef FFT1D_R2SDF_TW_DIR
`define FFT1D_R2SDF_TW_DIR "twiddles"
`endif

`define STRINGIFY(x) `"x`"



// -----------------------------------------------------------------------------
// fft1d_r2sdf_tb -- testbench del pipeline vm para el R2SDF.
//
// Reinyecta los estimulos que genero fft1d_r2sdf_tb.cpp (stimuli/in_ports) en
// el RTL fft1d_r2sdf y vuelca actual/out_ports/*.dat para que
// run_regression_vm los compare linea a linea contra expected/.
//
// DIFERENCIAS CON fft1d_r2_tb.sv (el iterativo):
//
//   1. TWIDDLES POR CARPETA. El R2SDF lee twiddles por etapa desde TW_DIR
//      (tw_s0_re.mem, tw_s1_re.mem, ...), no un par de archivos. Se pasa
//      TW_DIR y TW_FROM_FILE=1 al DUT.
//
//   2. o_last NO SE COMPARA. El RTL emite o_last, pero el modelo C++ no lo
//      tiene como puerto y no lo volco en expected/. Por eso o_last se escribe
//      SOLO al CSV (inspeccion), NO como o_last.dat en actual/: si se generara,
//      el vm fallaria por artefacto sin par en expected/.
//
//   3. REGISTROS TRAZADOS. El R2SDF no tiene FSM. Se trazan los registros que
//      el tb.cpp volco con add_reg_o: r_count y r_out_valid. El sufijo ".o" lo
//      agrega el lado C++, asi que los archivos son r_count.o.dat y
//      r_out_valid.o.dat (mismo cuidado que en el R2).
//
//   4. SIN r_state / r_stage / r_btfly (no hay FSM en esta arquitectura).
// -----------------------------------------------------------------------------
module fft1d_r2sdf_tb;

    localparam int N   = `FFT1D_R2SDF_N;
    localparam int NB  = `FFT1D_R2SDF_NB;
    localparam int NBF = `FFT1D_R2SDF_NBF;

    localparam string TW_DIR = `STRINGIFY(`FFT1D_R2SDF_TW_DIR);

    /*
     * EMPAQUETADO EN LA FRONTERA (identico al R2)
     * El DUT usa puertos complejos empaquetados ({imag, real}); los vectores se
     * manejan por COMPONENTE (i_re/i_im, o_re/o_im). El empaquetado queda
     * confinado a estas cuatro asignaciones.
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
    string p__i_valid, p__i_re, p__i_im;
    string p__o_valid, p__o_re, p__o_im;
    string p__r_count, p__r_out_valid;
    string waves_path;

    int fd__i_valid, fd__i_re, fd__i_im;
    int fd__o_valid, fd__o_re, fd__o_im, fd__csv;
    int fd__r_count, fd__r_out_valid;
    int s__i_valid, s__i_re, s__i_im;

    // ------------------------------------------------------------------- DUT
    // TW_FROM_FILE=1: cada etapa carga <TW_DIR>/tw_s<k>_{re,im}.mem via $readmemh.
    fft1d_r2sdf #(
        .N            (N),
        .NB           (NB),
        .NBF          (NBF),
        .TW_DIR       (TW_DIR),
        .TW_FROM_FILE (1'b1)
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
            $fatal(1, "[fft1d_r2sdf_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[fft1d_r2sdf_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[fft1d_r2sdf_tb] N_CYCLES invalido=%0d", n_cycles);

        in_dir       = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir      = {case_dir, "/simulation/vectors/actual"};
        act_out_dir  = {act_dir,  "/out_ports"};
        act_csv_path = {act_dir,  "/out_ports.csv"};

        p__i_valid = {in_dir, "/i_valid.dat"};
        p__i_re    = {in_dir, "/i_re.dat"};
        p__i_im    = {in_dir, "/i_im.dat"};

        p__o_valid = {act_out_dir, "/o_valid.dat"};
        p__o_re    = {act_out_dir, "/o_re.dat"};
        p__o_im    = {act_out_dir, "/o_im.dat"};

        /*
         * Registros del control. add_reg_o() del lado C++ agrega el sufijo ".o",
         * asi que los archivos son r_count.o.dat y r_out_valid.o.dat.
         *
         * o_last NO se genera como .dat: el modelo no lo volco en expected/, y
         * un .dat sin par haria fallar al vm. Se escribe solo al CSV.
         */
        p__r_count     = {act_out_dir, "/r_count.o.dat"};
        p__r_out_valid = {act_out_dir, "/r_out_valid.o.dat"};

        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, fft1d_r2sdf_tb);
            $display("[fft1d_r2sdf_tb] waves: %s", waves_path);
        end

        fd__i_valid = $fopen(p__i_valid, "r");
        fd__i_re    = $fopen(p__i_re,    "r");
        fd__i_im    = $fopen(p__i_im,    "r");
        if (fd__i_valid==0 || fd__i_re==0 || fd__i_im==0)
            $fatal(1, "[fft1d_r2sdf_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd__o_valid     = $fopen(p__o_valid, "w");
        fd__o_re        = $fopen(p__o_re,    "w");
        fd__o_im        = $fopen(p__o_im,    "w");
        fd__csv         = $fopen(act_csv_path, "w");
        fd__r_count     = $fopen(p__r_count,     "w");
        fd__r_out_valid = $fopen(p__r_out_valid, "w");
        if (fd__o_valid==0 || fd__o_re==0 || fd__o_im==0 || fd__csv==0
            || fd__r_count==0 || fd__r_out_valid==0)
            $fatal(1, "[fft1d_r2sdf_tb] no pude abrir salidas en %s", act_out_dir);

        $display("[fft1d_r2sdf_tb] CASE_DIR = %s", case_dir);
        $display("[fft1d_r2sdf_tb] N_CYCLES = %0d  N=%0d NB=%0d NBF=%0d", n_cycles, N, NB, NBF);
        $display("[fft1d_r2sdf_tb] TW_DIR   = %s", TW_DIR);

        // o_last en el CSV para inspeccion; NO se compara como .dat
        $fwrite(fd__csv, "cycle,o_valid,o_last,o_re,o_im,r_count,r_out_valid\n");

        // ---------------------------------------- reset (= init del modelo)
        i_rst   = 1'b1;
        i_valid = 1'b0;
        i_re    = '0;
        i_im    = '0;
        @(negedge i_clock);
        @(posedge i_clock);   // el modelo arranca sin consumir ciclo
        i_rst = 1'b0;

        // ------------------------------------------------- lazo por ciclo
        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            @(negedge i_clock);

            if ($fscanf(fd__i_valid, "%d", s__i_valid) != 1)
                $fatal(1, "[fft1d_r2sdf_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_re, "%d", s__i_re) != 1)
                $fatal(1, "[fft1d_r2sdf_tb] i_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_im, "%d", s__i_im) != 1)
                $fatal(1, "[fft1d_r2sdf_tb] i_im: faltan muestras en cycle=%0d", cycle);

            i_valid = s__i_valid[0];
            i_re    = s__i_re[NB-1:0];
            i_im    = s__i_im[NB-1:0];

            @(posedge i_clock);
            #1;

            $fwrite(fd__r_count,     "%0d\n", u_fft.r_count);
            $fwrite(fd__r_out_valid, "%0d\n", u_fft.r_out_valid);

            $fwrite(fd__o_valid, "%0d\n", o_valid);
            $fwrite(fd__o_re,    "%0d\n", $signed(o_re));
            $fwrite(fd__o_im,    "%0d\n", $signed(o_im));
            $fwrite(fd__csv, "%0d,%0d,%0d,%.16g,%.16g,%0d,%0d\n",
                    cycle + 1, o_valid, o_last,
                    fixed_to_real(o_re), fixed_to_real(o_im),
                    u_fft.r_count, u_fft.r_out_valid);
        end

        $fclose(fd__i_valid); $fclose(fd__i_re); $fclose(fd__i_im);
        $fclose(fd__o_valid); $fclose(fd__o_re); $fclose(fd__o_im);
        $fclose(fd__csv);
        $fclose(fd__r_count); $fclose(fd__r_out_valid);
        $display("[fft1d_r2sdf_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire
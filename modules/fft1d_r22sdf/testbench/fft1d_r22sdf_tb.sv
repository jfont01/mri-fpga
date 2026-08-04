`timescale 1ns/1ps
`default_nettype none

`ifndef FFT1D_R22SDF_N
`define FFT1D_R22SDF_N 64
`endif
`ifndef FFT1D_R22SDF_NB
`define FFT1D_R22SDF_NB 16
`endif
`ifndef FFT1D_R22SDF_NBF
`define FFT1D_R22SDF_NBF 15
`endif
`ifndef FFT1D_R22SDF_TW_DIR
`define FFT1D_R22SDF_TW_DIR twiddles
`endif

/*
 * STRINGIFY: convierte un token del preprocesador en un string literal.
 * Permite pasar TW_DIR por -d SIN comillas, que se rompen al viajar por el
 * shell y por el exec de Tcl (bash, MSYS y cmd las escapan distinto). Con esto
 * las comillas las agrega Verilog, no el shell.
 */
`define STRINGIFY(x) `"x`"

// -----------------------------------------------------------------------------
// fft1d_r22sdf_tb -- testbench del pipeline vm para el R2^2SDF.
//
// Reinyecta los estimulos que genero fft1d_r22sdf_tb.cpp (stimuli/in_ports) en
// el RTL fft1d_r22sdf y vuelca actual/out_ports/*.dat para que
// run_regression_vm los compare linea a linea contra expected/.
//
// DIFERENCIAS CON fft1d_r2sdf_tb.sv:
//
//   1. TWIDDLES EN UN SOLO PAR DE ARCHIVOS. El R2^2SDF usa UNA tabla de N
//      entradas (<TW_DIR>/tw_re.mem y tw_im.mem) que todas las unidades
//      comparten y direccionan distinto segun su resolucion M. El R2SDF, en
//      cambio, tiene un par de archivos POR ETAPA.
//
//   2. LATENCIA MAYOR. El pipeline esta segmentado (registros entre las dos
//      mariposas de cada unidad y a la salida del multiplicador). El primer
//      o_valid aparece en el ciclo 70 (N=64), 265 (N=256) o 1036 (N=1024).
//      N_CYCLES debe ser >= latencia + N.
//
//   3. REGISTROS TRAZADOS. El control es POR ETAPA: cada unidad lleva sus
//      propios contadores. Se trazan r_out_count del top y los tres contadores
//      de la primera unidad (jerarquia gen_unit[0].u_unit), que son los que
//      primero divergen si el control esta desalineado.
//
//   4. o_last NO SE COMPARA como .dat (el modelo no lo vuelca en expected/ y un
//      .dat sin par haria fallar al vm). Se escribe solo al CSV.
// -----------------------------------------------------------------------------
module fft1d_r22sdf_tb;

    localparam int N   = `FFT1D_R22SDF_N;
    localparam int NB  = `FFT1D_R22SDF_NB;
    localparam int NBF = `FFT1D_R22SDF_NBF;

    localparam string TW_DIR = `STRINGIFY(`FFT1D_R22SDF_TW_DIR);

    /*
     * EMPAQUETADO EN LA FRONTERA
     * El DUT usa puertos complejos empaquetados ({imag, real}); los vectores se
     * manejan por COMPONENTE (i_re/i_im, o_re/o_im). El empaquetado queda
     * confinado a estas asignaciones.
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
    string p__r_out_count, p__u0_di_count, p__u0_bf1_count, p__u0_bf2_count;
    string waves_path;

    int fd__i_valid, fd__i_re, fd__i_im;
    int fd__o_valid, fd__o_re, fd__o_im, fd__csv;
    int fd__r_out_count, fd__u0_di_count, fd__u0_bf1_count, fd__u0_bf2_count;
    int s__i_valid, s__i_re, s__i_im;

    // ------------------------------------------------------------------- DUT
    // TW_FROM_FILE=1: las unidades cargan <TW_DIR>/tw_{re,im}.mem via $readmemh.
    fft1d_r22sdf #(
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
            $fatal(1, "[fft1d_r22sdf_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[fft1d_r22sdf_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[fft1d_r22sdf_tb] N_CYCLES invalido=%0d", n_cycles);

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
         * asi que los archivos llevan .o.dat.
         */
        p__r_out_count  = {act_out_dir, "/r_out_count.o.dat"};
        p__u0_di_count  = {act_out_dir, "/u0_di_count.o.dat"};
        p__u0_bf1_count = {act_out_dir, "/u0_bf1_count.o.dat"};
        p__u0_bf2_count = {act_out_dir, "/u0_bf2_count.o.dat"};

        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, fft1d_r22sdf_tb);
            $display("[fft1d_r22sdf_tb] waves: %s", waves_path);
        end

        fd__i_valid = $fopen(p__i_valid, "r");
        fd__i_re    = $fopen(p__i_re,    "r");
        fd__i_im    = $fopen(p__i_im,    "r");
        if (fd__i_valid==0 || fd__i_re==0 || fd__i_im==0)
            $fatal(1, "[fft1d_r22sdf_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd__o_valid      = $fopen(p__o_valid, "w");
        fd__o_re         = $fopen(p__o_re,    "w");
        fd__o_im         = $fopen(p__o_im,    "w");
        fd__csv          = $fopen(act_csv_path, "w");
        fd__r_out_count  = $fopen(p__r_out_count,  "w");
        fd__u0_di_count  = $fopen(p__u0_di_count,  "w");
        fd__u0_bf1_count = $fopen(p__u0_bf1_count, "w");
        fd__u0_bf2_count = $fopen(p__u0_bf2_count, "w");
        
        if (fd__o_valid==0 || fd__o_re==0 || fd__o_im==0 || fd__csv==0
            || fd__r_out_count==0 || fd__u0_di_count==0
            || fd__u0_bf1_count==0 || fd__u0_bf2_count==0)
            $fatal(1, "[fft1d_r22sdf_tb] no pude abrir salidas en %s", act_out_dir);

        $display("[fft1d_r22sdf_tb] CASE_DIR = %s", case_dir);
        $display("[fft1d_r22sdf_tb] N_CYCLES = %0d  N=%0d NB=%0d NBF=%0d", n_cycles, N, NB, NBF);
        $display("[fft1d_r22sdf_tb] TW_DIR   = %s", TW_DIR);

        // o_last en el CSV para inspeccion; NO se compara como .dat
        $fwrite(fd__csv, "cycle,o_valid,o_last,o_re,o_im,r_out_count,u0_di_count,u0_bf1_count,u0_bf2_count\n");

        // ---------------------------------------- reset (= init del modelo)
        //
        // El modelo C++ arranca con init() y su primer sim.cycle() avanza los
        // contadores. Para alinear el RTL (reset SINCRONO, activo alto): se
        // aplica i_rst=1 y se pasa UN posedge, luego se suelta i_rst en el
        // negedge siguiente. El lazo NO debe empezar con otro negedge.
        i_rst   = 1'b1;
        i_valid = 1'b0;
        i_re    = '0;
        i_im    = '0;
        @(posedge i_clock);   // reset sincrono efectivo
        @(negedge i_clock);
        i_rst = 1'b0;         // reset liberado; el proximo posedge ya cuenta

        // ------------------------------------------------- lazo por ciclo
        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            if (cycle != 0)
                @(negedge i_clock);

            if ($fscanf(fd__i_valid, "%d", s__i_valid) != 1)
                $fatal(1, "[fft1d_r22sdf_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_re, "%d", s__i_re) != 1)
                $fatal(1, "[fft1d_r22sdf_tb] i_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_im, "%d", s__i_im) != 1)
                $fatal(1, "[fft1d_r22sdf_tb] i_im: faltan muestras en cycle=%0d", cycle);

            i_valid = s__i_valid[0];
            i_re    = s__i_re[NB-1:0];
            i_im    = s__i_im[NB-1:0];

            @(posedge i_clock);
            #1;

            $fwrite(fd__r_out_count,  "%0d\n", u_fft.r_out_count);
            $fwrite(fd__u0_di_count,  "%0d\n", u_fft.gen_unit[0].u_unit.r_di_count);
            $fwrite(fd__u0_bf1_count, "%0d\n", u_fft.gen_unit[0].u_unit.r_bf1_count);
            $fwrite(fd__u0_bf2_count, "%0d\n", u_fft.gen_unit[0].u_unit.r_bf2_count);

            $fwrite(fd__o_valid, "%0d\n", o_valid);
            $fwrite(fd__o_re,    "%0d\n", $signed(o_re));
            $fwrite(fd__o_im,    "%0d\n", $signed(o_im));
            $fwrite(fd__csv, "%0d,%0d,%0d,%.16g,%.16g,%0d,%0d,%0d,%0d\n",
                    cycle + 1, o_valid, o_last,
                    fixed_to_real(o_re), fixed_to_real(o_im),
                    u_fft.r_out_count,
                    u_fft.gen_unit[0].u_unit.r_di_count,
                    u_fft.gen_unit[0].u_unit.r_bf1_count,
                    u_fft.gen_unit[0].u_unit.r_bf2_count);
        end

        $fclose(fd__i_valid); $fclose(fd__i_re); $fclose(fd__i_im);
        $fclose(fd__o_valid); $fclose(fd__o_re); $fclose(fd__o_im);
        $fclose(fd__csv);
        $fclose(fd__r_out_count);  $fclose(fd__u0_di_count);
        $fclose(fd__u0_bf1_count); $fclose(fd__u0_bf2_count);
        $display("[fft1d_r22sdf_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire
`timescale 1ns/1ps
`default_nettype none

`ifndef CMUL_NB_IN
`define CMUL_NB_IN 16
`endif
`ifndef CMUL_NBF_IN
`define CMUL_NBF_IN 14
`endif
`ifndef CMUL_NB_OUT
`define CMUL_NB_OUT 16
`endif
`ifndef CMUL_NBF_OUT
`define CMUL_NBF_OUT 14
`endif

module cmul_tb;

    localparam int NB_IN   = `CMUL_NB_IN;
    localparam int NBF_IN  = `CMUL_NBF_IN;
    localparam int NB_OUT  = `CMUL_NB_OUT;
    localparam int NBF_OUT = `CMUL_NBF_OUT;

    logic signed [NB_IN  - 1 : 0] i_1_re, i_1_im, i_2_re, i_2_im;
    logic signed [NB_OUT - 1 : 0] o_re, o_im;

    string case_dir;
    int    n_cycles;

    string in_dir, act_dir, act_out_dir, act_csv_path;
    string p_i_1_re, p_i_1_im, p_i_2_re, p_i_2_im;
    string p_o_re, p_o_im;
    string waves_path;

    int fd_i1r, fd_i1i, fd_i2r, fd_i2i;
    int fd_ore, fd_oim, fd_csv;

    int s_i1r, s_i1i, s_i2r, s_i2i;

    cmul #(
        .NB_IN   (NB_IN)   ,
        .NBF_IN  (NBF_IN)  ,
        .NB_OUT  (NB_OUT)  ,
        .NBF_OUT (NBF_OUT)
    ) u_cmul (
        .i_1_re (i_1_re) ,
        .i_1_im (i_1_im) ,
        .i_2_re (i_2_re) ,
        .i_2_im (i_2_im) ,
        .o_re   (o_re)   ,
        .o_im   (o_im)
    );

    function real fixed_out_to_real(input logic signed [NB_OUT-1:0] raw);
        fixed_out_to_real = $itor($signed(raw)) / (2.0 ** NBF_OUT);
    endfunction

    initial begin
        if (!$value$plusargs("CASE_DIR=%s", case_dir))
            $fatal(1, "[cmul_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[cmul_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[cmul_tb] N_CYCLES invalido=%0d", n_cycles);

        in_dir       = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir      = {case_dir, "/simulation/vectors/actual"};
        act_out_dir  = {act_dir,  "/out_ports"};
        act_csv_path = {act_dir,  "/out_ports.csv"};

        p_i_1_re = {in_dir, "/i_1_re.dat"};
        p_i_1_im = {in_dir, "/i_1_im.dat"};
        p_i_2_re = {in_dir, "/i_2_re.dat"};
        p_i_2_im = {in_dir, "/i_2_im.dat"};
        p_o_re   = {act_out_dir, "/o_re.dat"};
        p_o_im   = {act_out_dir, "/o_im.dat"};

        fd_i1r = $fopen(p_i_1_re, "r");
        fd_i1i = $fopen(p_i_1_im, "r");
        fd_i2r = $fopen(p_i_2_re, "r");
        fd_i2i = $fopen(p_i_2_im, "r");
        if (fd_i1r==0 || fd_i1i==0 || fd_i2r==0 || fd_i2i==0)
            $fatal(1, "[cmul_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd_ore = $fopen(p_o_re, "w");
        fd_oim = $fopen(p_o_im, "w");
        fd_csv = $fopen(act_csv_path, "w");
        if (fd_ore==0 || fd_oim==0 || fd_csv==0)
            $fatal(1, "[cmul_tb] no pude abrir salidas en %s", act_out_dir);

        /*
         * Waves opcional: +WAVES -> <case_dir>/simulation/waves.vcd
         */
        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, cmul_tb);
            $display("[cmul_tb] waves: %s", waves_path);
        end

        $display("[cmul_tb] CASE_DIR = %s", case_dir);
        $display("[cmul_tb] N_CYCLES = %0d", n_cycles);
        $display("[cmul_tb] NB_IN=%0d NBF_IN=%0d NB_OUT=%0d NBF_OUT=%0d",
                 NB_IN, NBF_IN, NB_OUT, NBF_OUT);

        $fwrite(fd_csv, "cycle,o_re,o_im\n");

        i_1_re = '0; i_1_im = '0; i_2_re = '0; i_2_im = '0;

        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            if ($fscanf(fd_i1r, "%d", s_i1r) != 1)
                $fatal(1, "[cmul_tb] i_1_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd_i1i, "%d", s_i1i) != 1)
                $fatal(1, "[cmul_tb] i_1_im: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd_i2r, "%d", s_i2r) != 1)
                $fatal(1, "[cmul_tb] i_2_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd_i2i, "%d", s_i2i) != 1)
                $fatal(1, "[cmul_tb] i_2_im: faltan muestras en cycle=%0d", cycle);

            i_1_re = s_i1r[NB_IN-1:0];
            i_1_im = s_i1i[NB_IN-1:0];
            i_2_re = s_i2r[NB_IN-1:0];
            i_2_im = s_i2i[NB_IN-1:0];

            /*
             * DUT combinacional: esperar un paso de tiempo antes de muestrear.
             * (misma politica que cast_tb.sv)
             */
            #1;

            $fwrite(fd_ore, "%0d\n", $signed(o_re));
            $fwrite(fd_oim, "%0d\n", $signed(o_im));
            $fwrite(fd_csv, "%0d,%.16g,%.16g\n",
                    cycle + 1, fixed_out_to_real(o_re), fixed_out_to_real(o_im));
        end

        $fclose(fd_i1r); $fclose(fd_i1i); $fclose(fd_i2r); $fclose(fd_i2i);
        $fclose(fd_ore); $fclose(fd_oim); $fclose(fd_csv);
        $display("[cmul_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire
`timescale 1ns/1ps
`default_nettype none

`ifndef CAST_NB_IN
`define CAST_NB_IN 8
`endif

`ifndef CAST_NBF_IN
`define CAST_NBF_IN 5
`endif

`ifndef CAST_NB_OUT
`define CAST_NB_OUT 6
`endif

`ifndef CAST_NBF_OUT
`define CAST_NBF_OUT 4
`endif

`ifndef CAST_ROUND_MODE
`define CAST_ROUND_MODE 1
`endif

module cast_tb;

    localparam int NB_IN      = `CAST_NB_IN;
    localparam int NBF_IN     = `CAST_NBF_IN;
    localparam int NB_OUT     = `CAST_NB_OUT;
    localparam int NBF_OUT    = `CAST_NBF_OUT;
    localparam bit ROUND_MODE = `CAST_ROUND_MODE;

    logic signed [NB_IN-1:0]  i_word;
    logic signed [NB_OUT-1:0] o_word;

    string case_dir;
    int    n_cycles;

    string stimuli_dat_path;
    string actual_dir;
    string actual_out_ports_dir;
    string actual_csv_path;
    string actual_dat_path;

    int fd_i_word_dat;
    int fd_out_csv;
    int fd_o_word_dat;

    string line;
    int scan_status;

    logic signed [NB_IN-1:0] raw_i_word;

    cast #(
        .NB_IN      (NB_IN),
        .NBF_IN     (NBF_IN),
        .NB_OUT     (NB_OUT),
        .NBF_OUT    (NBF_OUT),
        .ROUND_MODE (ROUND_MODE)
    ) u_cast (
        .i_word (i_word),
        .o_word (o_word)
    );

    function real fixed_out_to_real(input logic signed [NB_OUT-1:0] raw);
        fixed_out_to_real = $itor($signed(raw)) / (2.0 ** NBF_OUT);
    endfunction

    initial begin
        if (!$value$plusargs("CASE_DIR=%s", case_dir)) begin
            $fatal(1, "[cast_tb] missing plusarg: +CASE_DIR=<path>");
        end

        if (!$value$plusargs("N_CYCLES=%d", n_cycles)) begin
            $fatal(1, "[cast_tb] missing plusarg: +N_CYCLES=<value>");
        end

        if (n_cycles < 0) begin
            $fatal(1, "[cast_tb] invalid N_CYCLES=%0d", n_cycles);
        end

        stimuli_dat_path     = {case_dir, "/simulation/vectors/stimuli/in_ports/i_word.dat"};
        actual_dir           = {case_dir, "/simulation/vectors/actual"};
        actual_out_ports_dir = {actual_dir, "/out_ports"};
        actual_csv_path      = {actual_dir, "/out_ports.csv"};
        actual_dat_path      = {actual_out_ports_dir, "/o_word.dat"};

        fd_i_word_dat = $fopen(stimuli_dat_path, "r");
        if (fd_i_word_dat == 0) begin
            $fatal(1, "[cast_tb] could not open input DAT: %s", stimuli_dat_path);
        end

        fd_out_csv = $fopen(actual_csv_path, "w");
        if (fd_out_csv == 0) begin
            $fatal(1, "[cast_tb] could not open output CSV: %s", actual_csv_path);
        end

        fd_o_word_dat = $fopen(actual_dat_path, "w");
        if (fd_o_word_dat == 0) begin
            $fatal(1, "[cast_tb] could not open output DAT: %s", actual_dat_path);
        end

        $display("[cast_tb] CASE_DIR   = %s", case_dir);
        $display("[cast_tb] N_CYCLES   = %0d", n_cycles);
        $display("[cast_tb] NB_IN      = %0d", NB_IN);
        $display("[cast_tb] NBF_IN     = %0d", NBF_IN);
        $display("[cast_tb] NB_OUT     = %0d", NB_OUT);
        $display("[cast_tb] NBF_OUT    = %0d", NBF_OUT);
        $display("[cast_tb] ROUND_MODE = %0d", ROUND_MODE);

        $fwrite(fd_out_csv, "cycle,o_word\n");

        i_word = '0;

        for (int cycle = 0; cycle < n_cycles; cycle++) begin
            if ($fgets(line, fd_i_word_dat) == 0) begin
                $fatal(
                    1,
                    "[cast_tb] not enough samples in %s. Requested N_CYCLES=%0d, failed at cycle=%0d",
                    stimuli_dat_path,
                    n_cycles,
                    cycle
                );
            end

            scan_status = $sscanf(line, "%d", raw_i_word);

            if (scan_status != 1) begin
                $fatal(
                    1,
                    "[cast_tb] could not parse signed decimal input at cycle=%0d. line='%s'",
                    cycle,
                    line
                );
            end

            i_word = raw_i_word;

            /*
             * Combinational DUT.
             * Wait one delta/time step before sampling.
             */
            #1;

            $fwrite(fd_out_csv, "%0d,%.16g\n", cycle + 1, fixed_out_to_real(o_word));

            /*
             * DAT policy:
             *   signed raw decimal integer
             */
            $fwrite(fd_o_word_dat, "%0d\n", $signed(o_word));
        end

        $fclose(fd_i_word_dat);
        $fclose(fd_out_csv);
        $fclose(fd_o_word_dat);

        $display("[cast_tb] simulation completed");
        $display("[cast_tb] actual CSV : %s", actual_csv_path);
        $display("[cast_tb] actual DAT : %s", actual_dat_path);

        $finish;
    end

endmodule

`default_nettype wire
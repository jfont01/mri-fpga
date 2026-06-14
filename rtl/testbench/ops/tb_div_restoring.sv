`timescale 1ns/1ps

import track_params_pkg::*;

module tb_div_restoring;

    localparam int NB_NUM        = NB_DIV_NUM;
    localparam int NBF_NUM       = NBF_DIV_NUM;

    localparam int NB_DEN        = NB_DIV_DEN;
    localparam int NBF_DEN       = NBF_DIV_DEN;

    localparam int NB_QUOTIENT   = NB_DIV_QUOTIENT;
    localparam int NBF_QUOTIENT  = NBF_DIV_QUOTIENT;

    localparam int CLK_PERIOD_NS = 10;

    localparam int OUT_NHEX      = (NB_QUOTIENT + 3) / 4;
    localparam int OUT_PAD_BITS  = OUT_NHEX * 4;

    localparam int TIMEOUT_CYCLES = 4096;

    logic                           i_clock;
    logic                           i_rst;
    logic                           i_start;

    logic signed [NB_NUM-1:0]       i_num;
    logic signed [NB_DEN-1:0]       i_den;

    logic signed [NB_QUOTIENT-1:0]  o_quotient;
    logic                           o_ready;
    logic                           o_busy;

    integer fd_in;
    integer fd_out;
    integer status;
    integer case_idx;

    logic [NB_NUM-1:0]              num_hex_r;
    logic [NB_DEN-1:0]              den_hex_r;

    logic [NB_QUOTIENT-1:0]         quotient_bits;
    logic [OUT_PAD_BITS-1:0]        quotient_padded;

    string in_file;
    string out_file;

    div_restoring #(
        .NB_NUM       (NB_NUM),
        .NBF_NUM      (NBF_NUM),
        .NB_DEN       (NB_DEN),
        .NBF_DEN      (NBF_DEN),
        .NB_QUOTIENT  (NB_QUOTIENT),
        .NBF_QUOTIENT (NBF_QUOTIENT)
    ) dut (
        .i_clock    (i_clock),
        .i_rst      (i_rst),
        .i_start    (i_start),
        .i_num      (i_num),
        .i_den      (i_den),
        .o_quotient (o_quotient),
        .o_ready    (o_ready),
        .o_busy     (o_busy)
    );

    always #(CLK_PERIOD_NS/2) i_clock = ~i_clock;

    task automatic pulse_start;
    begin
        @(posedge i_clock);
        i_start <= 1'b1;

        @(posedge i_clock);
        i_start <= 1'b0;
    end
    endtask

    task automatic reset_dut;
    begin
        i_rst   <= 1'b1;
        i_start <= 1'b0;
        i_num   <= '0;
        i_den   <= '0;

        repeat (4) @(posedge i_clock);

        i_rst <= 1'b0;

        repeat (2) @(posedge i_clock);
    end
    endtask

    task automatic wait_ready_or_timeout;
        integer timeout_count;
    begin
        timeout_count = 0;

        while (o_ready !== 1'b1) begin
            @(posedge i_clock);
            timeout_count = timeout_count + 1;

            if (timeout_count >= TIMEOUT_CYCLES) begin
                $display("[TB][ERROR] Timeout waiting for o_ready. case_idx=%0d", case_idx);
                $display("[TB][ERROR] i_num=0x%0h i_den=0x%0h o_busy=%0b o_ready=%0b",
                         i_num, i_den, o_busy, o_ready);
                $fclose(fd_in);
                $fclose(fd_out);
                $finish;
            end
        end
    end
    endtask

    task automatic write_hex_fixed_width(
        input logic [NB_QUOTIENT-1:0] value
    );
        integer n;
    begin
        quotient_padded = '0;
        quotient_padded[NB_QUOTIENT-1:0] = value;

        for (n = OUT_NHEX - 1; n >= 0; n = n - 1) begin
            $fwrite(fd_out, "%1h", quotient_padded[n*4 +: 4]);
        end

        $fwrite(fd_out, "\n");
    end
    endtask

    task automatic run_one_case(
        input logic [NB_NUM-1:0] num_bits,
        input logic [NB_DEN-1:0] den_bits
    );
    begin
        i_num <= $signed(num_bits);
        i_den <= $signed(den_bits);

        pulse_start();

        wait_ready_or_timeout();

        quotient_bits = o_quotient;
        write_hex_fixed_width(quotient_bits);

        @(posedge i_clock);
    end
    endtask

    initial begin
        i_clock = 1'b0;
        i_rst   = 1'b0;
        i_start = 1'b0;
        i_num   = '0;
        i_den   = '0;

        if (!$value$plusargs("IN_FILE=%s", in_file)) begin
            in_file = "stimuli/div_restoring_trunc_in.dat";
        end

        if (!$value$plusargs("OUT_FILE=%s", out_file)) begin
            out_file = "vectors/rtl/rtl_div_restoring_trunc.dat";
        end

        $display("[TB][INFO] IN_FILE  = %s", in_file);
        $display("[TB][INFO] OUT_FILE = %s", out_file);

        $display("[TB][INFO] NB_NUM        = %0d", NB_NUM);
        $display("[TB][INFO] NBF_NUM       = %0d", NBF_NUM);
        $display("[TB][INFO] NB_DEN        = %0d", NB_DEN);
        $display("[TB][INFO] NBF_DEN       = %0d", NBF_DEN);
        $display("[TB][INFO] NB_QUOTIENT   = %0d", NB_QUOTIENT);
        $display("[TB][INFO] NBF_QUOTIENT  = %0d", NBF_QUOTIENT);

        fd_in = $fopen(in_file, "r");
        if (fd_in == 0) begin
            $display("[TB][ERROR] Could not open input file: %s", in_file);
            $finish;
        end

        fd_out = $fopen(out_file, "w");
        if (fd_out == 0) begin
            $display("[TB][ERROR] Could not open output file: %s", out_file);
            $fclose(fd_in);
            $finish;
        end

        case_idx = 0;

        reset_dut();

        while (!$feof(fd_in)) begin
            status = $fscanf(fd_in, "%h %h\n", num_hex_r, den_hex_r);

            if (status == 2) begin
                run_one_case(num_hex_r, den_hex_r);
                case_idx = case_idx + 1;
            end
            else if (status != -1) begin
                $display("[TB][WARNING] Ignoring malformed line. status=%0d", status);
            end
        end

        $display("[TB][OK] Processed cases: %0d", case_idx);

        $fclose(fd_in);
        $fclose(fd_out);

        repeat (5) @(posedge i_clock);

        $finish;
    end

endmodule
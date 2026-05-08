`timescale 1ns/1ps

module tb_div_restoring;

    parameter integer NB_NUM        = 16;
    parameter integer NBF_NUM       = 15;

    parameter integer NB_DEN        = 16;
    parameter integer NBF_DEN       = 15;

    parameter integer NB_QUOTIENT   = 16;
    parameter integer NBF_QUOTIENT  = 15;

    localparam integer CLK_PERIOD_NS = 10;

    reg                             i_clock;
    reg                             i_rst;
    reg                             i_start;
    reg signed [NB_NUM-1:0]         i_num;
    reg signed [NB_DEN-1:0]         i_den;

    wire signed [NB_QUOTIENT-1:0]   o_quotient;
    wire                            o_ready;
    wire                            o_busy;

    integer fd_in;
    integer fd_out;
    integer status;
    integer case_idx;

    reg [NB_NUM-1:0]                num_hex_r;
    reg [NB_DEN-1:0]                den_hex_r;

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

    task automatic run_one_case(
        input [NB_NUM-1:0] num_bits,
        input [NB_DEN-1:0] den_bits
    );
    begin
        i_num <= $signed(num_bits);
        i_den <= $signed(den_bits);

        pulse_start();

        wait (o_ready == 1'b1);
        @(posedge i_clock);

        $fwrite(fd_out, "%0h\n", o_quotient);
    end
    endtask

    initial begin
        i_clock = 1'b0;

        if (!$value$plusargs("IN_FILE=%s", in_file)) begin
            in_file = "/home/jfont/Desktop/mri-fpga/rtl/testbench/ops/dat_div_restoring_16b/div_restoring_trunc_in.dat";
        end

        if (!$value$plusargs("OUT_FILE=%s", out_file)) begin
            out_file = "/home/jfont/Desktop/mri-fpga/rtl/testbench/ops/rtl_div_restoring_trunc_out.dat";
        end

        fd_in = $fopen(in_file, "r");
        if (fd_in == 0) begin
            $display("[TB][ERROR] No pude abrir archivo de entrada: %s", in_file);
            $finish;
        end

        fd_out = $fopen(out_file, "w");
        if (fd_out == 0) begin
            $display("[TB][ERROR] No pude abrir archivo de salida: %s", out_file);
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
        end

        $display("[TB][OK] Casos procesados: %0d", case_idx);

        $fclose(fd_in);
        $fclose(fd_out);

        repeat (5) @(posedge i_clock);
        $finish;
    end

endmodule
`timescale 1ns/1ps

module tb_compute_Aij;

    localparam int NB_S  = 16;
    localparam int NBF_S = 15;
    localparam int NB_A  = 28;
    localparam int NBF_A = 26;

    localparam int L      = 4;
    localparam int Nx     = 32;
    localparam int Ny     = 32;
    localparam int Af     = 2;
    localparam int offset = Ny / Af;

    logic i_clock;
    logic i_rst;
    logic i_start;
    logic o_valid;

    logic signed [NB_S-1:0] S_re [0:L-1][0:Nx-1][0:Ny-1];
    logic signed [NB_S-1:0] S_im [0:L-1][0:Nx-1][0:Ny-1];

    logic signed [NB_S-1:0] s0_re [L-1:0];
    logic signed [NB_S-1:0] s0_im [L-1:0];
    logic signed [NB_S-1:0] s1_re [L-1:0];
    logic signed [NB_S-1:0] s1_im [L-1:0];

    logic signed [NB_A-1:0] A00_re;
    logic signed [NB_A-1:0] A01_re;
    logic signed [NB_A-1:0] A01_im;
    logic signed [NB_A-1:0] A10_re;
    logic signed [NB_A-1:0] A10_im;
    logic signed [NB_A-1:0] A11_re;

    integer fin;
    integer fout;
    integer rc;

    integer l;
    integer nx;
    integer ny;
    reg [NB_S-1:0] re_tmp;
    reg [NB_S-1:0] im_tmp;

    integer k;
    integer ny_alias;

    compute_Aij #(
        .NB_S  (NB_S),
        .NBF_S (NBF_S),
        .NB_A  (NB_A),
        .NBF_A (NBF_A),
        .L     (L)
    ) dut (
        .i_clock(i_clock),
        .i_rst  (i_rst),
        .i_start(i_start),

        .s0_re  (s0_re),
        .s0_im  (s0_im),
        .s1_re  (s1_re),
        .s1_im  (s1_im),

        .A00_re (A00_re),
        .A11_re (A11_re),
        .A01_re (A01_re),
        .A01_im (A01_im),
        .A10_re (A10_re),
        .A10_im (A10_im),

        .o_valid(o_valid)
    );

    // Clock: 100 MHz
    initial begin
        i_clock = 1'b0;
        forever #5 i_clock = ~i_clock;
    end

    task automatic init_inputs;
        integer kk;
        begin
            i_rst   = 1'b0;
            i_start = 1'b0;
            for (kk = 0; kk < L; kk = kk + 1) begin
                s0_re[kk] = '0;
                s0_im[kk] = '0;
                s1_re[kk] = '0;
                s1_im[kk] = '0;
            end
        end
    endtask

    task automatic apply_reset;
        begin
            i_rst   = 1'b1;
            i_start = 1'b0;
            repeat (3) @(posedge i_clock);
            i_rst   = 1'b0;
            repeat (2) @(posedge i_clock);
        end
    endtask

    task automatic init_zeros;
        integer ll, nxx, nyy;
        begin
            for (ll = 0; ll < L; ll = ll + 1) begin
                for (nxx = 0; nxx < Nx; nxx = nxx + 1) begin
                    for (nyy = 0; nyy < Ny; nyy = nyy + 1) begin
                        S_re[ll][nxx][nyy] = '0;
                        S_im[ll][nxx][nyy] = '0;
                    end
                end
            end
        end
    endtask

    task automatic load_S_dat(input string path);
        begin
            fin = $fopen(path, "r");
            if (fin == 0) begin
                $fatal(1, "No se pudo abrir %s", path);
            end

            while (!$feof(fin)) begin
                rc = $fscanf(fin, "%d %d %d %h %h\n", l, nx, ny, re_tmp, im_tmp);
                if (rc == 5) begin
                    if ((l >= 0) && (l < L) &&
                        (nx >= 0) && (nx < Nx) &&
                        (ny >= 0) && (ny < Ny)) begin
                        S_re[l][nx][ny] = re_tmp;
                        S_im[l][nx][ny] = im_tmp;
                    end
                end
            end
            $fclose(fin);
        end
    endtask

    task automatic drive_case(input integer nx_i, input integer ny_alias_i);
        begin
            for (k = 0; k < L; k = k + 1) begin
                s0_re[k] = S_re[k][nx_i][ny_alias_i];
                s0_im[k] = S_im[k][nx_i][ny_alias_i];
                s1_re[k] = S_re[k][nx_i][ny_alias_i + offset];
                s1_im[k] = S_im[k][nx_i][ny_alias_i + offset];
            end
        end
    endtask

    task automatic start_case;
        begin
            @(posedge i_clock);
            i_start <= 1'b1;
            @(posedge i_clock);
            i_start <= 1'b0;
        end
    endtask

    task automatic wait_valid;
        begin
            @(posedge o_valid);
            #1;
        end
    endtask

    task automatic write_case(input integer nx_i, input integer ny_alias_i);
        begin
            // shape = (2,2,Nx,offset)
            $fdisplay(fout, "0 0 %0d %0d %0h %0h", nx_i, ny_alias_i, A00_re, {NB_A{1'b0}});
            $fdisplay(fout, "0 1 %0d %0d %0h %0h", nx_i, ny_alias_i, A01_re, A01_im);
            $fdisplay(fout, "1 0 %0d %0d %0h %0h", nx_i, ny_alias_i, A10_re, A10_im);
            $fdisplay(fout, "1 1 %0d %0d %0h %0h", nx_i, ny_alias_i, A11_re, {NB_A{1'b0}});
        end
    endtask

    initial begin
        init_inputs();
        init_zeros();

        load_S_dat("py_S.dat");

        fout = $fopen("rtl_A.dat", "w");
        if (fout == 0) begin
            $fatal(1, "No se pudo abrir rtl_A.dat para escritura");
        end


        apply_reset();

        for (nx = 0; nx < Nx; nx = nx + 1) begin
            for (ny_alias = 0; ny_alias < offset; ny_alias = ny_alias + 1) begin
                drive_case(nx, ny_alias);
                start_case();
                wait_valid();
                write_case(nx, ny_alias);
            end
        end

        $fclose(fout);

        $display("Listo. Archivo generado: rtl_A.dat");
        $finish;
    end

endmodule
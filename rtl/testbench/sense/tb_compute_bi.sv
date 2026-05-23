`timescale 1ns/1ps

module tb_compute_bi;

    localparam int NB_S  = track_params_pkg::NB_S;
    localparam int NBF_S = track_params_pkg::NBF_S;

    localparam int NB_Y  = track_params_pkg::NB_Y;
    localparam int NBF_Y = track_params_pkg::NBF_Y;

    localparam int NB_B  = track_params_pkg::NB_B;
    localparam int NBF_B = track_params_pkg::NBF_B;

    localparam int L     = track_params_pkg::L;
    localparam int AF    = track_params_pkg::AF;
    localparam int N     = track_params_pkg::N;

    localparam int Ny = N;
    localparam int Nx = N;

    localparam int offset = Ny / AF;

    logic i_clock;
    logic i_rst;
    logic i_start;
    logic o_valid;

    logic signed [NB_S-1:0] S_re [0:L-1][0:Nx-1][0:Ny-1];
    logic signed [NB_S-1:0] S_im [0:L-1][0:Nx-1][0:Ny-1];

    logic signed [NB_Y-1:0] Y_re [0:L-1][0:Nx-1][0:Ny-1];
    logic signed [NB_Y-1:0] Y_im [0:L-1][0:Nx-1][0:Ny-1];

    logic signed [NB_S-1:0] s0_re [L-1:0];
    logic signed [NB_S-1:0] s0_im [L-1:0];

    logic signed [NB_S-1:0] s1_re [L-1:0];
    logic signed [NB_S-1:0] s1_im [L-1:0];

    logic signed [NB_Y-1:0] y_re [L-1:0];
    logic signed [NB_Y-1:0] y_im [L-1:0];

    logic signed [NB_B-1:0] b0_re;
    logic signed [NB_B-1:0] b0_im;
    logic signed [NB_B-1:0] b1_re;
    logic signed [NB_B-1:0] b1_im;

    integer fin;
    integer fout;
    integer rc;

    integer l;
    integer nx;
    integer ny;
    reg [NB_S-1:0] S_re_tmp;
    reg [NB_S-1:0] S_im_tmp;
    reg [NB_Y-1:0] Y_re_tmp;
    reg [NB_Y-1:0] Y_im_tmp;

    integer k;
    integer ny_alias;

    compute_bi #(
        .NB_S  (NB_S),
        .NBF_S (NBF_S),
        .NB_B  (NB_B),
        .NBF_B (NBF_B),
        .NB_Y  (NB_Y),
        .NBF_Y (NBF_Y),
        .L     (L)
    ) dut (
        .i_clock(i_clock),
        .i_rst  (i_rst),
        .i_start(i_start),

        .s0_re  (s0_re),
        .s0_im  (s0_im),
        .s1_re  (s1_re),
        .s1_im  (s1_im),
        .y_re   (y_re),
        .y_im   (y_im),

        .b0_re  (b0_re),
        .b0_im  (b0_im),
        .b1_re  (b1_re),
        .b1_im  (b1_im),

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
                y_re[kk] = '0;
                y_im[kk] = '0;
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
                        Y_re[ll][nxx][nyy] = '0;
                        Y_im[ll][nxx][nyy] = '0;
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
                rc = $fscanf(fin, "%d %d %d %h %h\n", l, nx, ny, S_re_tmp, S_im_tmp);
                if (rc == 5) begin
                    if ((l >= 0) && (l < L) && (nx >= 0) && (nx < Nx) && (ny >= 0) && (ny < Ny)) begin
                        S_re[l][nx][ny] = S_re_tmp;
                        S_im[l][nx][ny] = S_im_tmp;
                    end
                end
            end
            $fclose(fin);
        end
    endtask

    task automatic load_y_dat(input string path);
        begin
            fin = $fopen(path, "r");
            if (fin == 0) begin
                $fatal(1, "No se pudo abrir %s", path);
            end

            while (!$feof(fin)) begin
                rc = $fscanf(fin, "%d %d %d %h %h\n", l, nx, ny, Y_re_tmp, Y_im_tmp);
                if (rc == 5) begin
                    if ((l >= 0) && (l < L) && (nx >= 0) && (nx < Nx) && (ny >= 0) && (ny < Ny)) begin
                        Y_re[l][nx][ny] = Y_re_tmp;
                        Y_im[l][nx][ny] = Y_im_tmp;
                    end
                end
            end
            $fclose(fin);
        end
    endtask


    task automatic drive_case(input integer nx_i, input integer ny_alias_i);
        begin
            for (k = 0; k < L; k = k + 1) begin
                s0_re[k]    = S_re[k][nx_i][ny_alias_i];
                s0_im[k]    = S_im[k][nx_i][ny_alias_i];
                s1_re[k]    = S_re[k][nx_i][ny_alias_i + offset];
                s1_im[k]    = S_im[k][nx_i][ny_alias_i + offset];
                y_re[k]     = Y_re[k][nx_i][ny_alias_i];
                y_im[k]     = Y_im[k][nx_i][ny_alias_i];
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
            // shape = (2, Nx, Ny/2)
            $fdisplay(fout, "0 %0d %0d %0h %0h", nx_i, ny_alias_i, b0_re, b0_im);
            $fdisplay(fout, "1 %0d %0d %0h %0h", nx_i, ny_alias_i, b1_re, b1_im);
        end
    endtask

    initial begin
        init_inputs();
        init_zeros();

        load_S_dat("py_S.dat");
        load_y_dat("py_y.dat");

        fout = $fopen("rtl_b.dat", "w");
        if (fout == 0) begin
            $fatal(1, "No se pudo abrir rtl_b.dat para escritura");
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

        $display("Listo. Archivo generado: rtl_b.dat");
        $finish;
    end

endmodule
`timescale 1ns/1ps
`default_nettype none

/*
 * Parametros: los mismos defines y defaults que el modelo (cpp/unfold.hpp) y el
 * RTL. Las claves en MAYUSCULAS del JSON llegan como -d UNFOLD_<KEY>; las que el
 * banco no usa (VEC_DIR, GAP_PCT, GAP_SEED, E2E: son del estimulo del modelo, que
 * ya esta en los .dat) se ignoran.
 */
`ifndef UNFOLD_L
`define UNFOLD_L 8
`endif
`ifndef UNFOLD_AF
`define UNFOLD_AF 2
`endif
`ifndef UNFOLD_NB_A
`define UNFOLD_NB_A 16
`endif
`ifndef UNFOLD_NB_U
`define UNFOLD_NB_U 16
`endif
`ifndef UNFOLD_G
`define UNFOLD_G 5
`endif
// Misma regla que el modelo y el RTL: NB_M = 17 + G.
`ifndef UNFOLD_NB_M
`define UNFOLD_NB_M (17 + `UNFOLD_G)
`endif
// 0 = half-up (default de D2), 1 = empates al par.
`ifndef UNFOLD_ROUND_CONV
`define UNFOLD_ROUND_CONV 0
`endif
`ifndef UNFOLD_NB_D
`define UNFOLD_NB_D 6
`endif

// -----------------------------------------------------------------------------
// unfold_tb -- banco del vm para unfold.
//
// Reinyecta los estimulos que escribio unfold_tb.cpp (run_regression_sim) en el
// RTL y vuelca actual/out_ports/*.dat para que run_regression_vm los compare
// linea por linea contra expected/ (el modelo). Se compara TODO ciclo, con o sin
// o_valid: o_m y o_err tienen que quedar quietos entre salidas, como el modelo.
//
//   entrada  (stimuli/in_ports):  i_valid, i_last, i_a_re, i_a_im,
//                                 i_u_re_<a>, i_u_im_<a> (a = 0 ... AF-1), i_d
//   salida   (actual/out_ports):  o_valid, o_err, o_m_re_<a>, o_m_im_<a>
//
// Un archivo por campo, en decimal (con signo: A, U, M; sin signo: i_d). El banco
// arma los buses empaquetados del RTL ({im, re}; el elemento a de i_u y o_m en
// [a*2*NB +: 2*NB]) y separa o_m en campos: el empaquetado queda confinado aca.
//
// N_CYCLES (plusarg, del JSON):
//   0  automatico (el default del JSON, como en run_regression_sim): corre hasta
//      el fin de i_valid.dat, o sea los mismos ciclos que vuelca el modelo, y
//      chequea que los otros .dat de entrada terminen en el mismo ciclo.
//   >0 corre exactamente N_CYCLES ciclos (faltan muestras: $fatal).
//
// Semantica de muestreo (igual que el modelo y que fft1d_r22sdf_tb.sv): por
// cada ciclo se fija la entrada, pasa un posedge y se muestrea la salida. La
// linea c tiene la entrada aplicada en el ciclo c y la salida DESPUES del flanco
// c. Sin reset: el RTL arranca con todos los registros en 0 (valor inicial),
// igual que init() del modelo, y el primer posedge ya es el ciclo 0.
//
// Con +WAVES vuelca <case_dir>/simulation/waves.vcd.
// -----------------------------------------------------------------------------
module unfold_tb;

    localparam int L          = `UNFOLD_L;
    localparam int AF         = `UNFOLD_AF;
    localparam int NB_A       = `UNFOLD_NB_A;
    localparam int NB_U       = `UNFOLD_NB_U;
    localparam int G          = `UNFOLD_G;
    localparam int NB_M       = `UNFOLD_NB_M;
    localparam int ROUND_CONV = `UNFOLD_ROUND_CONV;
    localparam int NB_D       = `UNFOLD_NB_D;

    // ------------------------------------------------- puertos por campo
    logic                     i_clock;
    logic                     i_valid;
    logic                     i_last;
    logic signed [NB_A-1:0]   i_a_re;
    logic signed [NB_A-1:0]   i_a_im;
    logic signed [NB_U-1:0]   i_u_re [AF];
    logic signed [NB_U-1:0]   i_u_im [AF];
    logic        [NB_D-1:0]   i_d;

    wire                      o_valid;
    wire                      o_err;
    wire  signed [NB_M-1:0]   o_m_re [AF];
    wire  signed [NB_M-1:0]   o_m_im [AF];

    // ------------------------------------------ empaquetado en la frontera
    wire [2*NB_A-1:0]         w_i_a = {i_a_im, i_a_re};
    wire [AF*2*NB_U-1:0]      w_i_u;
    wire [AF*2*NB_M-1:0]      w_o_m;

    genvar ga;
    generate
        for (ga = 0; ga < AF; ga = ga + 1) begin : g_pack
            assign w_i_u[ga*2*NB_U +: 2*NB_U] = {i_u_im[ga], i_u_re[ga]};
            assign o_m_re[ga] = $signed(w_o_m[ga*2*NB_M        +: NB_M]);
            assign o_m_im[ga] = $signed(w_o_m[ga*2*NB_M + NB_M +: NB_M]);
        end
    endgenerate

    // ------------------------------------------------------------------- DUT
    unfold #(
        .L          (L),
        .AF         (AF),
        .NB_A       (NB_A),
        .NB_U       (NB_U),
        .G          (G),
        .NB_M       (NB_M),
        .ROUND_CONV (ROUND_CONV),
        .NB_D       (NB_D)
    ) u_dut (
        .i_clock    (i_clock),
        .i_valid    (i_valid),
        .i_last     (i_last),
        .i_a        (w_i_a),
        .i_u        (w_i_u),
        .i_d        (i_d),
        .o_valid    (o_valid),
        .o_err      (o_err),
        .o_m        (w_o_m)
    );

    // -------------------------------------------------------------- clock
    initial i_clock = 1'b0;
    always #5 i_clock = ~i_clock;

    // -------------------------------------------------------------- archivos
    string case_dir, in_dir, act_dir, act_out_dir, waves_path;
    int    n_cycles;
    bit    auto_cycles;

    int fd_i_valid, fd_i_last, fd_i_a_re, fd_i_a_im, fd_i_d;
    int fd_i_u_re [AF];
    int fd_i_u_im [AF];
    int fd_o_valid, fd_o_err, fd_csv;
    int fd_o_m_re [AF];
    int fd_o_m_im [AF];

    // Abre un .dat (lectura o escritura); si no puede, corta.
    function automatic int open_dat(input string path, input string mode);
        int fd;
        fd = $fopen(path, mode);
        if (fd == 0)
            $fatal(1, "[unfold_tb] no pude abrir %s", path);
        return fd;
    endfunction

    // Lee la muestra del ciclo de un .dat de entrada; si falta, corta.
    task automatic read_dat(input int fd, input string name, input int cycle, output int v);
        if ($fscanf(fd, "%d", v) != 1)
            $fatal(1, "[unfold_tb] %s: faltan muestras en cycle=%0d", name, cycle);
    endtask

    // Con N_CYCLES automatico, un .dat de entrada tiene que terminar junto con i_valid.dat.
    task automatic check_end(input int fd, input string name, input int cycles);
        int v;
        if ($fscanf(fd, "%d", v) == 1)
            $fatal(1, "[unfold_tb] %s: tiene mas muestras que i_valid.dat (%0d ciclos)", name, cycles);
    endtask

    int cycle, n_run, n_valid, n_err;
    int s_valid, s_last, s_a_re, s_a_im, s_d, s_u;

    initial begin
        if (!$value$plusargs("CASE_DIR=%s", case_dir))
            $fatal(1, "[unfold_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[unfold_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[unfold_tb] N_CYCLES invalido=%0d", n_cycles);
        auto_cycles = (n_cycles == 0);

        in_dir      = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir     = {case_dir, "/simulation/vectors/actual"};
        act_out_dir = {act_dir,  "/out_ports"};

        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, unfold_tb);
            $display("[unfold_tb] waves: %s", waves_path);
        end

        fd_i_valid = open_dat({in_dir, "/i_valid.dat"}, "r");
        fd_i_last  = open_dat({in_dir, "/i_last.dat"},  "r");
        fd_i_a_re  = open_dat({in_dir, "/i_a_re.dat"},  "r");
        fd_i_a_im  = open_dat({in_dir, "/i_a_im.dat"},  "r");
        fd_i_d     = open_dat({in_dir, "/i_d.dat"},     "r");
        for (int a = 0; a < AF; a++) begin
            fd_i_u_re[a] = open_dat($sformatf("%s/i_u_re_%0d.dat", in_dir, a), "r");
            fd_i_u_im[a] = open_dat($sformatf("%s/i_u_im_%0d.dat", in_dir, a), "r");
        end

        fd_o_valid = open_dat({act_out_dir, "/o_valid.dat"}, "w");
        fd_o_err   = open_dat({act_out_dir, "/o_err.dat"},   "w");
        for (int a = 0; a < AF; a++) begin
            fd_o_m_re[a] = open_dat($sformatf("%s/o_m_re_%0d.dat", act_out_dir, a), "w");
            fd_o_m_im[a] = open_dat($sformatf("%s/o_m_im_%0d.dat", act_out_dir, a), "w");
        end
        fd_csv = open_dat({act_dir, "/out_ports.csv"}, "w");

        $display("[unfold_tb] CASE_DIR = %s", case_dir);
        $display("[unfold_tb] N_CYCLES = %0d%s", n_cycles, auto_cycles ? " (automatico: hasta el fin de i_valid.dat)" : "");
        $display("[unfold_tb] L=%0d AF=%0d NB_A=%0d NB_U=%0d G=%0d NB_M=%0d ROUND_CONV=%0d NB_D=%0d",
                 L, AF, NB_A, NB_U, G, NB_M, ROUND_CONV, NB_D);

        // Mismo encabezado que el CSV del modelo (no se compara: solo los .dat).
        $fwrite(fd_csv, "cycle,o_valid,o_err");
        for (int a = 0; a < AF; a++)
            $fwrite(fd_csv, ",o_m_re_%0d,o_m_im_%0d", a, a);
        $fwrite(fd_csv, "\n");

        // Entradas en reposo hasta el primer ciclo (el RTL no tiene reset).
        i_valid = 1'b0;
        i_last  = 1'b0;
        i_a_re  = '0;
        i_a_im  = '0;
        i_d     = '0;
        for (int a = 0; a < AF; a++) begin
            i_u_re[a] = '0;
            i_u_im[a] = '0;
        end

        n_run   = 0;
        n_valid = 0;
        n_err   = 0;

        // ------------------------------------------------- lazo por ciclo
        for (cycle = 0; auto_cycles || cycle < n_cycles; cycle++) begin
            if (cycle != 0)
                @(negedge i_clock);

            if ($fscanf(fd_i_valid, "%d", s_valid) != 1) begin
                if (auto_cycles)
                    break;
                $fatal(1, "[unfold_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            end
            read_dat(fd_i_last, "i_last", cycle, s_last);
            read_dat(fd_i_a_re, "i_a_re", cycle, s_a_re);
            read_dat(fd_i_a_im, "i_a_im", cycle, s_a_im);
            read_dat(fd_i_d,    "i_d",    cycle, s_d);

            i_valid = s_valid[0];
            i_last  = s_last[0];
            i_a_re  = s_a_re[NB_A-1:0];
            i_a_im  = s_a_im[NB_A-1:0];
            i_d     = s_d[NB_D-1:0];
            for (int a = 0; a < AF; a++) begin
                read_dat(fd_i_u_re[a], $sformatf("i_u_re_%0d", a), cycle, s_u);
                i_u_re[a] = s_u[NB_U-1:0];
                read_dat(fd_i_u_im[a], $sformatf("i_u_im_%0d", a), cycle, s_u);
                i_u_im[a] = s_u[NB_U-1:0];
            end

            @(posedge i_clock);
            #1;

            n_run++;
            n_valid += o_valid;
            n_err   += (o_valid && o_err);

            $fwrite(fd_o_valid, "%0d\n", o_valid);
            $fwrite(fd_o_err,   "%0d\n", o_err);
            $fwrite(fd_csv, "%0d,%0d,%0d", cycle + 1, o_valid, o_err);
            for (int a = 0; a < AF; a++) begin
                $fwrite(fd_o_m_re[a], "%0d\n", o_m_re[a]);
                $fwrite(fd_o_m_im[a], "%0d\n", o_m_im[a]);
                $fwrite(fd_csv, ",%0d,%0d", o_m_re[a], o_m_im[a]);
            end
            $fwrite(fd_csv, "\n");
        end

        if (n_run == 0)
            $fatal(1, "[unfold_tb] no corrio ningun ciclo (i_valid.dat vacio?)");

        if (auto_cycles) begin
            check_end(fd_i_last, "i_last", n_run);
            check_end(fd_i_a_re, "i_a_re", n_run);
            check_end(fd_i_a_im, "i_a_im", n_run);
            check_end(fd_i_d,    "i_d",    n_run);
            for (int a = 0; a < AF; a++) begin
                check_end(fd_i_u_re[a], $sformatf("i_u_re_%0d", a), n_run);
                check_end(fd_i_u_im[a], $sformatf("i_u_im_%0d", a), n_run);
            end
        end

        $fclose(fd_i_valid); $fclose(fd_i_last); $fclose(fd_i_a_re); $fclose(fd_i_a_im); $fclose(fd_i_d);
        $fclose(fd_o_valid); $fclose(fd_o_err);  $fclose(fd_csv);
        for (int a = 0; a < AF; a++) begin
            $fclose(fd_i_u_re[a]); $fclose(fd_i_u_im[a]);
            $fclose(fd_o_m_re[a]); $fclose(fd_o_m_im[a]);
        end

        $display("[unfold_tb] ciclos = %0d, o_valid = %0d (con o_err: %0d)", n_run, n_valid, n_err);
        $display("[unfold_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire

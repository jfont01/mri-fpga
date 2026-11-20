`timescale 1ns/1ps
`default_nettype none

/*
 * Parametros del caso, por define desde el JSON (CORDIC_MAG_<CLAVE>). Mismos
 * defaults que el RTL y el modelo (cordic_mag.hpp): un caso que no los menciona
 * se comporta exactamente igual en los tres.
 */
`ifndef CORDIC_MAG_NB_OUT
`define CORDIC_MAG_NB_OUT 16
`endif
`ifndef CORDIC_MAG_G
`define CORDIC_MAG_G 5
`endif
`ifndef CORDIC_MAG_N_IT
`define CORDIC_MAG_N_IT 12
`endif
`ifndef CORDIC_MAG_ROUND_CONV
`define CORDIC_MAG_ROUND_CONV 0
`endif

/*
 * NB_M: si el caso no lo fija, el DUT se instancia SIN NB_M, para ejercitar el
 * default del RTL (NB_OUT + 1 + G). El tb usa la misma regla para el ancho de
 * sus senales y en tiempo 0 chequea que coincida con la del DUT. Si el caso lo
 * fija (nbm24), se pasa explicito.
 */

// -----------------------------------------------------------------------------
// cordic_mag_tb -- testbench del vm para cordic_mag.
//
// Reinyecta en el RTL los estimulos que escribio cordic_mag_tb.cpp (etapa sim)
// y vuelca actual/out_ports/{o_valid,o_mag}.dat, que run_regression_vm compara
// linea a linea contra expected/ (todas las lineas: los ciclos sin o_valid
// tambien, asi que el clock enable de o_mag queda verificado):
//
//   <CASE_DIR>/simulation/vectors/stimuli/in_ports/{i_valid,i_m_re,i_m_im}.dat
//   <CASE_DIR>/simulation/vectors/actual/out_ports/{o_valid,o_mag}.dat
//   <CASE_DIR>/simulation/vectors/actual/out_ports.csv   (inspeccion)
//
// N_CYCLES (plusarg; run_regression_vm le pasa el del JSON):
//
//   0   automatico (default del JSON de cordic_mag): corre hasta que se acaba
//       i_valid.dat. Es la cantidad de ciclos que eligio el tb de C++ (entradas
//       + huecos + LATENCY + cola, ver cordic_mag_stimulus.hpp), asi que el RTL
//       produce exactamente tantas lineas como expected/. Al final chequea que
//       i_m_re.dat e i_m_im.dat tambien se hayan terminado.
//   n   corre n ciclos; si algun .dat tiene menos lineas, $fatal. (Con n > 0
//       el tb de C++ escribe exactamente n lineas.)
//
// Semantica de muestreo (la de los otros tb del vm y la de VectorTraceFiles):
// en el ciclo c la entrada se pone antes del flanco (en el negedge) y la salida
// se lee 1 ns despues del flanco: la linea c tiene la entrada del ciclo c y la
// salida despues del flanco c. Sin reset: los registros del DUT arrancan en 0,
// igual que init() del modelo, y el primer flanco ya es el ciclo 0.
//
// Las entradas se leen como longint (los .dat son enteros con signo en
// decimal) y se chequea que entren en NB_M bits con signo: si el JSON y los
// defaults del tb no coincidieran con los del modelo, lo dice aca en vez de
// truncar en silencio.
// -----------------------------------------------------------------------------
module cordic_mag_tb;

    localparam int NB_OUT     = `CORDIC_MAG_NB_OUT;
    localparam int G          = `CORDIC_MAG_G;
    localparam int N_IT       = `CORDIC_MAG_N_IT;
    localparam int ROUND_CONV = `CORDIC_MAG_ROUND_CONV;
`ifdef CORDIC_MAG_NB_M
    localparam int NB_M       = `CORDIC_MAG_NB_M;
`else
    localparam int NB_M       = NB_OUT + 1 + G;
`endif
    localparam int LATENCY    = N_IT + 1;

    localparam longint M_MIN  = -(64'sd1 <<< (NB_M - 1));
    localparam longint M_MAX  =  (64'sd1 <<< (NB_M - 1)) - 1;

    logic                     i_clock = 1'b0;
    logic                     i_valid = 1'b0;
    logic signed [NB_M-1:0]   i_m_re  = '0;
    logic signed [NB_M-1:0]   i_m_im  = '0;
    wire                      o_valid;
    wire         [NB_OUT-1:0] o_mag;

    // ------------------------------------------------------------------- DUT
`ifdef CORDIC_MAG_NB_M
    cordic_mag #(
        .NB_OUT     (NB_OUT),
        .G          (G),
        .N_IT       (N_IT),
        .NB_M       (NB_M),
        .ROUND_CONV (ROUND_CONV)
    ) u_dut (
`else
    cordic_mag #(
        .NB_OUT     (NB_OUT),
        .G          (G),
        .N_IT       (N_IT),
        .ROUND_CONV (ROUND_CONV)
    ) u_dut (
`endif
        .i_clock (i_clock),
        .i_valid (i_valid),
        .i_m_re  (i_m_re),
        .i_m_im  (i_m_im),
        .o_valid (o_valid),
        .o_mag   (o_mag)
    );

    // -------------------------------------------------------------- clock
    always #5 i_clock = ~i_clock;

    string  case_dir;
    int     n_cycles;
    string  in_dir, act_dir, act_out_dir, act_csv_path, waves_path;

    int     fd__i_valid, fd__i_m_re, fd__i_m_im;
    int     fd__o_valid, fd__o_mag, fd__csv;
    longint s__i_valid, s__i_m_re, s__i_m_im;
    int     rc, cycle;
    int     n_in_valid, n_out_valid;

    initial begin
        if (!$value$plusargs("CASE_DIR=%s", case_dir))
            $fatal(1, "[cordic_mag_tb] falta plusarg: +CASE_DIR=<path>");
        if (!$value$plusargs("N_CYCLES=%d", n_cycles))
            $fatal(1, "[cordic_mag_tb] falta plusarg: +N_CYCLES=<value>");
        if (n_cycles < 0)
            $fatal(1, "[cordic_mag_tb] N_CYCLES invalido=%0d", n_cycles);

        // Mismo default de NB_M en el tb y en el RTL (ver arriba).
        if (u_dut.NB_M != NB_M)
            $fatal(1, "[cordic_mag_tb] NB_M del DUT = %0d, del tb = %0d", u_dut.NB_M, NB_M);

        in_dir       = {case_dir, "/simulation/vectors/stimuli/in_ports"};
        act_dir      = {case_dir, "/simulation/vectors/actual"};
        act_out_dir  = {act_dir,  "/out_ports"};
        act_csv_path = {act_dir,  "/out_ports.csv"};

        if ($test$plusargs("WAVES")) begin
            waves_path = {case_dir, "/simulation/waves.vcd"};
            $dumpfile(waves_path);
            $dumpvars(0, cordic_mag_tb);
            $display("[cordic_mag_tb] waves: %s", waves_path);
        end

        fd__i_valid = $fopen({in_dir, "/i_valid.dat"}, "r");
        fd__i_m_re  = $fopen({in_dir, "/i_m_re.dat"},  "r");
        fd__i_m_im  = $fopen({in_dir, "/i_m_im.dat"},  "r");
        if (fd__i_valid == 0 || fd__i_m_re == 0 || fd__i_m_im == 0)
            $fatal(1, "[cordic_mag_tb] no pude abrir algun .dat de estimulo en %s", in_dir);

        fd__o_valid = $fopen({act_out_dir, "/o_valid.dat"}, "w");
        fd__o_mag   = $fopen({act_out_dir, "/o_mag.dat"},   "w");
        fd__csv     = $fopen(act_csv_path, "w");
        if (fd__o_valid == 0 || fd__o_mag == 0 || fd__csv == 0)
            $fatal(1, "[cordic_mag_tb] no pude abrir salidas en %s", act_out_dir);

        $display("[cordic_mag_tb] CASE_DIR = %s", case_dir);
        $display("[cordic_mag_tb] N_CYCLES = %0d%s", n_cycles, (n_cycles == 0) ? " (automatico: hasta el fin de i_valid.dat)" : "");
        $display("[cordic_mag_tb] NB_OUT = %0d  G = %0d  N_IT = %0d  NB_M = %0d  ROUND_CONV = %0d  LATENCY = %0d",
                 NB_OUT, G, N_IT, NB_M, ROUND_CONV, LATENCY);

        $fwrite(fd__csv, "cycle,o_valid,o_mag\n");

        // --------------------------------------------------- lazo por ciclo
        cycle       = 0;
        n_in_valid  = 0;
        n_out_valid = 0;
        while (n_cycles == 0 || cycle < n_cycles) begin
            rc = $fscanf(fd__i_valid, "%d", s__i_valid);
            if (rc != 1) begin
                if (n_cycles == 0)
                    break;          // fin de los estimulos: modo automatico
                $fatal(1, "[cordic_mag_tb] i_valid: faltan muestras en cycle=%0d", cycle);
            end
            if ($fscanf(fd__i_m_re, "%d", s__i_m_re) != 1)
                $fatal(1, "[cordic_mag_tb] i_m_re: faltan muestras en cycle=%0d", cycle);
            if ($fscanf(fd__i_m_im, "%d", s__i_m_im) != 1)
                $fatal(1, "[cordic_mag_tb] i_m_im: faltan muestras en cycle=%0d", cycle);

            if (s__i_valid != 0 && s__i_valid != 1)
                $fatal(1, "[cordic_mag_tb] i_valid = %0d en cycle=%0d", s__i_valid, cycle);
            if (s__i_m_re < M_MIN || s__i_m_re > M_MAX || s__i_m_im < M_MIN || s__i_m_im > M_MAX)
                $fatal(1, "[cordic_mag_tb] i_m = (%0d, %0d) no entra en NB_M = %0d bits en cycle=%0d",
                       s__i_m_re, s__i_m_im, NB_M, cycle);

            if (cycle != 0)
                @(negedge i_clock);

            i_valid = s__i_valid[0];
            i_m_re  = s__i_m_re[NB_M-1:0];
            i_m_im  = s__i_m_im[NB_M-1:0];

            @(posedge i_clock);
            #1;

            n_in_valid  += int'(i_valid);
            n_out_valid += int'(o_valid);

            $fwrite(fd__o_valid, "%0d\n", o_valid);
            $fwrite(fd__o_mag,   "%0d\n", o_mag);
            $fwrite(fd__csv, "%0d,%0d,%0d\n", cycle + 1, o_valid, o_mag);

            cycle++;
        end

        // En automatico, los tres .dat de entrada tienen que terminar juntos.
        if (n_cycles == 0) begin
            if ($fscanf(fd__i_m_re, "%d", s__i_m_re) == 1 || $fscanf(fd__i_m_im, "%d", s__i_m_im) == 1)
                $fatal(1, "[cordic_mag_tb] i_m_re.dat o i_m_im.dat tienen mas lineas que i_valid.dat (%0d)", cycle);
            if (cycle == 0)
                $fatal(1, "[cordic_mag_tb] i_valid.dat vacio en %s", in_dir);
        end

        $fclose(fd__i_valid); $fclose(fd__i_m_re); $fclose(fd__i_m_im);
        $fclose(fd__o_valid); $fclose(fd__o_mag);  $fclose(fd__csv);

        // Informativo: la comparacion es la del vm, linea por linea.
        $display("[cordic_mag_tb] ciclos = %0d, entradas validas = %0d, salidas validas = %0d",
                 cycle, n_in_valid, n_out_valid);
        $display("[cordic_mag_tb] simulacion completa");
        $finish;
    end

endmodule
`default_nettype wire

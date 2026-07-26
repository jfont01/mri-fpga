// fft1d_r2sdf_tb.cpp
//
// Generador de vectores para el pipeline vm (NO es el gtest).
//
// Corre el golden C++ (fft1d_r2sdf_model) e inyecta un frame en STREAMING
// (una muestra por ciclo, sin huecos) seguido de idle para drenar la latencia.
// Vuelca con VectorTraceFiles:
//
//   simulation/vectors/stimuli/in_ports/{i_valid,i_re,i_im}.dat
//   simulation/vectors/expected/out_ports/{o_valid,o_re,o_im}.dat
//
// (o_last lo emite el RTL pero NO el modelo C++, asi que no se vuelca desde
//  aca; ver la nota en run_tb sobre como maneja esto el vm.)
//
// El testbench SystemVerilog (fft1d_r2sdf_tb.sv) reinyecta esos estimulos en el
// RTL y produce .../actual/out_ports/*.dat, que run_regression_vm compara linea
// a linea contra expected.
//
// DIFERENCIAS CON fft1d_r2_tb.cpp (el iterativo):
//   - No hay r_state/r_stage/r_btfly que trazar: el R2SDF no tiene FSM. Se
//     trazan los registros que SI existen (r_count, r_out_valid) para localizar
//     divergencias RTL vs modelo.
//   - El estimulo es streaming continuo: i_valid=1 durante N ciclos seguidos.
//     La latencia es N (cadena N-1 + registro de salida), asi que N_CYCLES debe
//     ser >= 2N para capturar el frame completo de salida.
//
// Semantica de muestreo (identica a los otros tb): por cada ciclo se fija la
// entrada, se avanza sim.cycle(), y se muestrea. La linea c contiene la entrada
// aplicada en el ciclo c y la salida DESPUES del flanco c.

#include "fft1d_r2sdf_tb.hpp"

#include "rtlsim.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace fft1d_r2sdf_tb {

using fft1d_r2sdf::N;
using fft1d_r2sdf::in_t;
using fft1d_r2sdf::fft1d_r2sdf_model;

namespace {

constexpr double PI = 3.14159265358979323846;

/*
 * Estimulo: un frame de senoidal compleja (k0, amp) durante los primeros N
 * ciclos, seguido de idle (i_valid=0) para drenar la latencia del pipeline.
 *
 * A diferencia del iterativo, aca i_valid=1 se mantiene N ciclos SEGUIDOS: el
 * R2SDF exige una muestra por ciclo sin huecos dentro del frame.
 */
constexpr int    K0  = 3;
constexpr double AMP = 0.5;

void frame_stimulus(int cycle, bool& valid, in_t& re, in_t& im)
{
    if (cycle < N) {
        const double angle = 2.0 * PI * static_cast<double>(K0)
                             * static_cast<double>(cycle) / static_cast<double>(N);
        valid = true;
        re = in_t(AMP * std::cos(angle));
        im = in_t(AMP * std::sin(angle));
    }
    else {
        valid = false;
        re = in_t(0);
        im = in_t(0);
    }
}

} // namespace

tb_args parse_args(int argc, char** argv)
{
    tb_args args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (key == "--case-dir") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for --case-dir");
            }
            args.case_dir = argv[i];
        }
        else if (key == "--n-cycles" || key == "--N_CYCLES") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            args.n_cycles = std::stoi(argv[i]);
        }
        else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.n_cycles < 0) {
        throw std::runtime_error("n_cycles must be non-negative");
    }

    return args;
}

void run_tb(const tb_args& args)
{
    const fs::path reports_dir = args.case_dir / "reports";
    fs::create_directories(reports_dir);

    std::ofstream report_file(reports_dir / "vm_summary.rpt");
    if (!report_file) {
        throw std::runtime_error("could not open report file");
    }

    rtl::ClockDomain  clk;
    fft1d_r2sdf_model dut;
    rtl::Simulator    sim(clk);

    sim.add(dut);
    sim.init();

    /*
     * Vector trace manager.
     *
     *   stimuli/in_ports/{i_valid,i_re,i_im}.dat
     *   expected/out_ports/{o_valid,o_re,o_im}.dat
     *
     * o_last no existe como puerto del modelo C++ (solo lo emite el RTL); no se
     * traza aca. Si se quisiera comparar o_last, habria que agregarlo al modelo.
     */
    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_valid", dut.i_valid);
    vectors.add_in_port("i_re",    dut.i_re);
    vectors.add_in_port("i_im",    dut.i_im);

    vectors.add_out_port("o_valid", dut.o_valid);
    vectors.add_out_port("o_re",    dut.o_re);
    vectors.add_out_port("o_im",    dut.o_im);

    // Registros escalares que existen en el R2SDF (no hay FSM).
    vectors.add_reg_o("r_count",     dut.r_count);
    vectors.add_reg_o("r_out_valid", dut.r_out_valid);

    for (int cycle = 0; cycle < args.n_cycles; ++cycle) {
        bool  v = false;
        in_t  re(0);
        in_t  im(0);
        frame_stimulus(cycle, v, re, im);

        dut.i_valid = rtl::bit_t(v);
        dut.i_re    = re;
        dut.i_im    = im;

        sim.cycle();

        vectors.sample(sim);
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : fft1d_r2sdf\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "n_cycles    : " << args.n_cycles << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "N           : " << fft1d_r2sdf::N       << "\n";
    report_file << "NB          : " << fft1d_r2sdf::NB      << "\n";
    report_file << "NBF         : " << fft1d_r2sdf::NBF     << "\n";
    report_file << "LOG2N       : " << fft1d_r2sdf::LOG2N   << "\n";
    report_file << "LATENCY     : " << fft1d_r2sdf::LATENCY << " (+1 por reg de salida -> N)\n";
    report_file << "\n";
    report_file << "stimulus    : complex sine k0=" << K0 << " amp=" << AMP << "\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv()  << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir()    << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir()   << "\n";
}

} // namespace fft1d_r2sdf_tb

int main(int argc, char** argv)
{
    try {
        const auto args = fft1d_r2sdf_tb::parse_args(argc, argv);
        fft1d_r2sdf_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[fft1d_r2sdf_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}
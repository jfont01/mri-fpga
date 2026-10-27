// fft2d_tb.cpp
//
// Generador de vectores de la fft2d (etapa sim; NO es el gtest).
//
// Corre el modelo ciclo a ciclo (fft2d_model) con el k-space del caso
// (fft2d_stimulus.hpp: ruido, impulso, tono o archivo), todas las bobinas en
// streaming, una muestra por ciclo y sin huecos, y despues idle para drenar.
// Vuelca con VectorTraceFiles:
//
//   simulation/vectors/stimuli/in_ports/{i_valid,i_re,i_im}.dat
//   simulation/vectors/expected/out_ports/{o_valid,o_last,o_re,o_im}.dat
//   simulation/vectors/expected/out_ports/{wr_count,rd_count,out_count}.o.dat
//
// Los tres contadores son la plomeria (posicion de escritura de la pasada 1,
// de lectura de la pasada 2 y de salida): si algun dia hay RTL de la fft2d,
// son los que primero divergen si el corner-turn esta mal.
//
// N_CYCLES = 0 (default del JSON): automatico, latencia + N_COILS * T + 8, que
// alcanza justo para la ultima muestra de la ultima bobina.
//
// Semantica de muestreo (igual que los otros tb): por cada ciclo se fija la
// entrada, se avanza sim.cycle() y se muestrea. La linea c tiene la entrada
// aplicada en el ciclo c y la salida DESPUES del flanco c.

#include "fft2d_tb.hpp"
#include "fft2d_stimulus.hpp"

#include "rtlsim.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace fft2d_tb {

using fft2d::in_t;

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

    const fft2d::Params p;
    const int  n_coils = FFT2D_N_COILS;
    const long t       = p.frame();
    const long n_in    = t * n_coils;
    const long n_cycles = args.n_cycles > 0 ? args.n_cycles : p.latency() + n_in + 8;

    const fft2d_stimulus::KSpace ks = fft2d_stimulus::generate(p, n_coils);

    rtl::ClockDomain   clk;
    fft2d::fft2d_model dut(p);
    rtl::Simulator     sim(clk);

    sim.add(dut);
    sim.init();

    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_valid", dut.i_valid);
    vectors.add_in_port("i_re",    dut.i_re);
    vectors.add_in_port("i_im",    dut.i_im);

    vectors.add_out_port("o_valid", dut.o_valid);
    vectors.add_out_port("o_last",  dut.o_last);
    vectors.add_out_port("o_re",    dut.o_re);
    vectors.add_out_port("o_im",    dut.o_im);

    vectors.add_reg_o("wr_count",  dut.wr_count);
    vectors.add_reg_o("rd_count",  dut.rd_count);
    vectors.add_reg_o("out_count", dut.out_count);

    std::cout << "[fft2d_tb] stimulus: " << fft2d_stimulus::type_name()
              << "  N1=" << p.n1() << " N2=" << p.n2() << " coils=" << n_coils
              << " n_cycles=" << n_cycles << "\n";

    long first_valid = -1, last_valid = -1, n_valid = 0, n_last = 0;

    for (long c = 0; c < n_cycles; ++c) {
        if (c < n_in) {
            const long l = c / t;
            const long i = c % t;
            dut.i_valid = fft2d::bit_t(true);
            dut.i_re    = ks[l][2 * i];
            dut.i_im    = ks[l][2 * i + 1];
        }
        else {
            dut.i_valid = fft2d::bit_t(false);
            dut.i_re    = in_t(0);
            dut.i_im    = in_t(0);
        }

        sim.cycle();
        vectors.sample(sim);

        if (dut.o_valid.value().to_bool()) {
            if (first_valid < 0) first_valid = c;
            last_valid = c;
            ++n_valid;
        }
        if (dut.o_last.value().to_bool()) ++n_last;
    }

    /*
     * Con N_CYCLES automatico, el tb se chequea solo: tienen que salir todas
     * las muestras, la primera en la latencia de la formula y un o_last por
     * bobina. Con N_CYCLES fijo puede ser a proposito mas corto: solo avisa.
     */
    const bool timing_ok = (n_valid == n_in) && (n_last == n_coils) && (first_valid == p.latency());
    if (!timing_ok) {
        const std::string msg = "salida incompleta o fuera de tiempo: " + std::to_string(n_valid) + " de " +
                                std::to_string(n_in) + " muestras, " + std::to_string(n_last) +
                                " o_last, primer o_valid en " + std::to_string(first_valid) +
                                " (formula " + std::to_string(p.latency()) + ")";
        if (args.n_cycles == 0) {
            throw std::runtime_error(msg);
        }
        std::cerr << "[fft2d_tb] AVISO: " << msg << "\n";
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : fft2d\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "n_cycles    : " << n_cycles << (args.n_cycles > 0 ? "" : " (automatico)") << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "N1 (fila)   : " << p.n1() << "  mask " << p.pass1.scale_mask
                << "  S1 " << p.pass1.scale_shift() << "  latency " << p.pass1.latency()
                << "  tw " << p.pass1.tw_dir << "\n";
    report_file << "N2 (filas)  : " << p.n2() << "  mask " << p.pass2.scale_mask
                << "  S2 " << p.pass2.scale_shift() << "  latency " << p.pass2.latency()
                << "  tw " << p.pass2.tw_dir << "\n";
    report_file << "INVERSE     : " << p.pass1.inverse << "\n";
    report_file << "CMUL_PIPE   : " << p.pass1.cmul_pipe << "\n";
    report_file << "ROUND_CONV  : " << p.pass1.round_conv << "\n";
    report_file << "NB/NBF      : " << fft1d_r22sdf::NB << " / " << fft1d_r22sdf::NBF << "\n";
    report_file << "NB_TW/NBF_TW: " << fft1d_r22sdf::NB_TW << " / " << fft1d_r22sdf::NBF_TW << "\n";
    report_file << "NB_IN/NBF_IN: " << fft1d_r22sdf::NB_IN << " / " << fft1d_r22sdf::NBF_IN << "\n";
    report_file << "NB_OUT/NBF_OUT: " << fft1d_r22sdf::NB_OUT << " / " << fft1d_r22sdf::NBF_OUT << "\n";
    report_file << "N_COILS     : " << n_coils << "\n";
    report_file << "T = N1*N2   : " << t << "\n";
    report_file << "LATENCY     : " << p.latency() << " (L1 + T + 2 + L2)\n";
    report_file << "\n";
    report_file << "stimulus    : " << fft2d_stimulus::type_name();
    if (FFT2D_STIM_TYPE == 3) report_file << " (" << fft2d_stimulus::file_path() << ")";
    report_file << "\n\n";

    report_file << "results\n";
    report_file << "-------\n";
    report_file << "first_valid : " << first_valid << "\n";
    report_file << "last_valid  : " << last_valid << "\n";
    report_file << "n_valid     : " << n_valid << " (esperadas " << n_in << ")\n";
    report_file << "o_last      : " << n_last << " (esperados " << n_coils << ")\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv()  << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir()    << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir()   << "\n";
}

} // namespace fft2d_tb

int main(int argc, char** argv)
{
    try {
        const auto args = fft2d_tb::parse_args(argc, argv);
        fft2d_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[fft2d_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}

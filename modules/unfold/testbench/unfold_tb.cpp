// unfold_tb.cpp
//
// Generador de vectores de unfold (etapa sim; NO es el gtest).
//
// Corre el modelo ciclo a ciclo (unfold_model) con el estimulo del caso
// (unfold_stimulus.hpp: el stream de VEC_DIR, con huecos y basura segun
// GAP_PCT / GAP_SEED) y vuelca con VectorTraceFiles, un archivo por puerto y
// los arreglos un archivo por elemento (a = 0 ... AF - 1):
//
//   simulation/vectors/stimuli/in_ports/{i_valid,i_last,i_a_re,i_a_im,
//       i_u_re_<a>,i_u_im_<a>,i_d}.dat
//   simulation/vectors/expected/out_ports/{o_valid,o_err,o_m_re_<a>,o_m_im_<a>}.dat
//
// N_CYCLES = 0 (default del JSON): automatico, el estimulo entero: muestras +
// huecos + LATENCY + una cola de N_STAGES ciclos. Con N_CYCLES mayor, la cola
// sigue con basura; con N_CYCLES menor, se corta (solo avisa).
//
// Semantica de muestreo (igual que los otros tb): por cada ciclo se fija la
// entrada, se avanza sim.cycle() y se muestrea. La linea c tiene la entrada
// aplicada en el ciclo c y la salida DESPUES del flanco c.
//
// Con N_CYCLES automatico el tb se chequea solo (si no, sim falla): una salida
// por posicion cerrada, cada una LATENCY ciclos despues de la muestra que la
// cerro, y o_m y o_err iguales al Python (m_ref.bin, close_ref.bin). La
// verificacion completa es el gtest.

#include "unfold_tb.hpp"
#include "unfold_stimulus.hpp"

#include "rtlsim.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace unfold_tb {

using unfold::AF;
using unfold::word_t;
namespace stim = unfold_stimulus;

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

    const stim::Case cs = stim::load(stim::vec_dir());
    const std::string mismatch = stim::param_mismatch(cs);
    if (!mismatch.empty()) {
        throw std::runtime_error(cs.dir + ": " + mismatch + ": " + stim::regen_hint());
    }

    const stim::Schedule sch = stim::build(cs, static_cast<std::size_t>(args.n_cycles));
    const std::size_t n_cycles = args.n_cycles > 0 ? static_cast<std::size_t>(args.n_cycles) : sch.cycles.size();

    rtl::ClockDomain     clk;
    unfold::unfold_model dut;
    rtl::Simulator       sim(clk);

    sim.add(dut);
    sim.init();

    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_valid", dut.i_valid);
    vectors.add_in_port("i_last",  dut.i_last);
    vectors.add_in_port("i_a_re",  dut.i_a_re);
    vectors.add_in_port("i_a_im",  dut.i_a_im);
    for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
        vectors.add_in_port("i_u_re_" + std::to_string(a), dut.i_u_re[a]);
        vectors.add_in_port("i_u_im_" + std::to_string(a), dut.i_u_im[a]);
    }
    vectors.add_in_port("i_d", dut.i_d);

    vectors.add_out_port("o_valid", dut.o_valid);
    vectors.add_out_port("o_err",   dut.o_err);
    for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
        vectors.add_out_port("o_m_re_" + std::to_string(a), dut.o_m_re[a]);
        vectors.add_out_port("o_m_im_" + std::to_string(a), dut.o_m_im[a]);
    }

    std::cout << "[unfold_tb] " << cs.dir << ": muestras = " << cs.n << ", posiciones = " << cs.n_out
              << ", GAP_PCT = " << stim::GAP_PCT << ", n_cycles = " << n_cycles
              << (args.n_cycles > 0 ? "" : " (automatico)") << "\n";

    std::size_t n_valid = 0, n_late = 0, n_bad = 0, n_err = 0;
    long first_valid = -1, last_valid = -1;
    std::string first_err;

    for (std::size_t c = 0; c < n_cycles; ++c) {
        stim::drive(dut, sch.cycles[c]);

        sim.cycle();
        vectors.sample(sim);

        if (dut.o_valid.value().to_bool()) {
            const std::size_t k = n_valid++;
            if (first_valid < 0) first_valid = static_cast<long>(c);
            last_valid = static_cast<long>(c);
            n_err += dut.o_err.value().to_bool() ? 1 : 0;
            if (k < cs.n_out) {
                const long in = sch.in_cycle[static_cast<std::size_t>(cs.close[k])];
                if (static_cast<long>(c) != in + unfold::LATENCY && n_late++ == 0 && first_err.empty()) {
                    first_err = "salida " + std::to_string(k) + " en el ciclo " + std::to_string(c) +
                                ", la muestra que cierra entro en el " + std::to_string(in);
                }
                bool same = dut.o_err.value().to_bool() == (cs.err[k] != 0);
                for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
                    same = same && dut.o_m_re[a].value() == cs.m_re[k][a] && dut.o_m_im[a].value() == cs.m_im[k][a];
                }
                if (!same && n_bad++ == 0 && first_err.empty()) {
                    first_err = "salida " + std::to_string(k) + ": o_m u o_err distintos del Python";
                }
            }
        }
    }

    const bool ok = n_valid == cs.n_out && n_late == 0 && n_bad == 0;
    if (!ok) {
        const std::string msg = std::to_string(n_valid) + " de " + std::to_string(cs.n_out) + " salidas, " +
                                std::to_string(n_late) + " fuera de tiempo, " + std::to_string(n_bad) +
                                " distintas del Python" + (first_err.empty() ? "" : " (primera: " + first_err + ")");
        if (args.n_cycles == 0) {
            throw std::runtime_error(msg);
        }
        std::cerr << "[unfold_tb] AVISO: " << msg << "\n";
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : unfold\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "vec_dir     : " << cs.dir << "\n";
    report_file << "n_cycles    : " << n_cycles << (args.n_cycles > 0 ? "" : " (automatico)") << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "L           : " << unfold::L << "\n";
    report_file << "AF          : " << unfold::AF << "\n";
    report_file << "NB_A        : " << unfold::NB_A << "\n";
    report_file << "NB_U        : " << unfold::NB_U << "\n";
    report_file << "G           : " << unfold::G << "\n";
    report_file << "NB_M        : " << unfold::NB_M << "\n";
    report_file << "ROUND_CONV  : " << unfold::ROUND_CONV << "\n";
    report_file << "NB_D        : " << unfold::NB_D << "\n";
    report_file << "NB_P        : " << unfold::NB_P << "\n";
    report_file << "N_STAGES    : " << unfold::N_STAGES << "\n";
    report_file << "LATENCY     : " << unfold::LATENCY << " (N_STAGES - 1)\n";
    report_file << "GAP_PCT     : " << stim::GAP_PCT << "\n";
    report_file << "GAP_SEED    : " << stim::GAP_SEED << "\n";
    report_file << "\n";

    report_file << "case.txt\n";
    report_file << "--------\n";
    for (const auto& kv : cs.info) {
        report_file << kv.first << " " << kv.second << "\n";
    }
    report_file << "\n";

    report_file << "results\n";
    report_file << "-------\n";
    report_file << "inputs      : " << cs.n << " muestras (huecos antes de la ultima: " << sch.idle_before_last << ")\n";
    report_file << "first_valid : " << first_valid << "\n";
    report_file << "last_valid  : " << last_valid << "\n";
    report_file << "n_valid     : " << n_valid << " (esperadas " << cs.n_out << ")\n";
    report_file << "n_err       : " << n_err << "\n";
    report_file << "late        : " << n_late << "\n";
    report_file << "o_m != py   : " << n_bad << "\n";
    report_file << "status      : " << (ok ? "OK" : "INCOMPLETO") << "\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv()  << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir()    << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir()   << "\n";
}

} // namespace unfold_tb

int main(int argc, char** argv)
{
    try {
        const auto args = unfold_tb::parse_args(argc, argv);
        unfold_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[unfold_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}

// cordic_mag_tb.cpp
//
// Generador de vectores de cordic_mag (etapa sim; NO es el gtest).
//
// Corre el modelo ciclo a ciclo (cordic_mag_model) con el estimulo del caso
// (cordic_mag_stimulus.hpp: los M de VEC_DIR, con huecos y basura segun
// GAP_PCT / GAP_SEED) y vuelca con VectorTraceFiles:
//
//   simulation/vectors/stimuli/in_ports/{i_valid,i_m_re,i_m_im}.dat
//   simulation/vectors/expected/out_ports/{o_valid,o_mag}.dat
//
// N_CYCLES = 0 (default del JSON): automatico, el estimulo entero: entradas +
// huecos + LATENCY + una cola de N_STAGES ciclos. Con N_CYCLES mayor, la cola
// sigue con basura; con N_CYCLES menor, se corta (solo avisa).
//
// Semantica de muestreo (igual que los otros tb): por cada ciclo se fija la
// entrada, se avanza sim.cycle() y se muestrea. La linea c tiene la entrada
// aplicada en el ciclo c y la salida DESPUES del flanco c.
//
// Con N_CYCLES automatico el tb se chequea solo (si no, sim falla): tantas
// salidas como entradas, cada una LATENCY ciclos despues de su entrada y o_mag
// igual al Python (out_ref.bin). La verificacion completa es el gtest.

#include "cordic_mag_tb.hpp"
#include "cordic_mag_stimulus.hpp"

#include "rtlsim.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace cordic_mag_tb {

using cordic_mag::word_t;
namespace stim = cordic_mag_stimulus;

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

    rtl::ClockDomain              clk;
    cordic_mag::cordic_mag_model  dut;
    rtl::Simulator                sim(clk);

    sim.add(dut);
    sim.init();

    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_valid", dut.i_valid);
    vectors.add_in_port("i_m_re",  dut.i_m_re);
    vectors.add_in_port("i_m_im",  dut.i_m_im);

    vectors.add_out_port("o_valid", dut.o_valid);
    vectors.add_out_port("o_mag",   dut.o_mag);

    std::cout << "[cordic_mag_tb] " << cs.dir << ": n = " << cs.n << ", GAP_PCT = " << stim::GAP_PCT
              << ", n_cycles = " << n_cycles << (args.n_cycles > 0 ? "" : " (automatico)") << "\n";

    std::size_t n_valid = 0, n_late = 0, n_bad = 0;
    long first_valid = -1, last_valid = -1;
    std::string first_err;

    for (std::size_t c = 0; c < n_cycles; ++c) {
        const stim::Cycle& in = sch.cycles[c];
        dut.i_valid = cordic_mag::bit_t(in.valid);
        dut.i_m_re  = in.re;
        dut.i_m_im  = in.im;

        sim.cycle();
        vectors.sample(sim);

        if (dut.o_valid.value().to_bool()) {
            const std::size_t k = n_valid++;
            if (first_valid < 0) first_valid = static_cast<long>(c);
            last_valid = static_cast<long>(c);
            if (k < cs.n) {
                if (static_cast<long>(c) != sch.in_cycle[k] + cordic_mag::LATENCY && n_late++ == 0 && first_err.empty()) {
                    first_err = "salida " + std::to_string(k) + " en el ciclo " + std::to_string(c) + ", entro en el " +
                                std::to_string(sch.in_cycle[k]);
                }
                if (dut.o_mag.value() != cs.out_ref[k] && n_bad++ == 0 && first_err.empty()) {
                    first_err = "salida " + std::to_string(k) + ": o_mag = " + std::to_string(dut.o_mag.value()) +
                                ", Python " + std::to_string(cs.out_ref[k]);
                }
            }
        }
    }

    const bool ok = n_valid == cs.n && n_late == 0 && n_bad == 0;
    if (!ok) {
        const std::string msg = std::to_string(n_valid) + " de " + std::to_string(cs.n) + " salidas, " +
                                std::to_string(n_late) + " fuera de tiempo, " + std::to_string(n_bad) +
                                " distintas del Python" + (first_err.empty() ? "" : " (primera: " + first_err + ")");
        if (args.n_cycles == 0) {
            throw std::runtime_error(msg);
        }
        std::cerr << "[cordic_mag_tb] AVISO: " << msg << "\n";
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : cordic_mag\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "vec_dir     : " << cs.dir << "\n";
    report_file << "n_cycles    : " << n_cycles << (args.n_cycles > 0 ? "" : " (automatico)") << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "NB_OUT      : " << cordic_mag::NB_OUT << "\n";
    report_file << "G           : " << cordic_mag::G << "\n";
    report_file << "N_IT        : " << cordic_mag::N_IT << "\n";
    report_file << "NB_M        : " << cordic_mag::NB_M << "\n";
    report_file << "ROUND_CONV  : " << cordic_mag::ROUND_CONV << "\n";
    report_file << "NB_W        : " << cordic_mag::NB_W << "\n";
    report_file << "N_STAGES    : " << cordic_mag::N_STAGES << "\n";
    report_file << "LATENCY     : " << cordic_mag::LATENCY << " (N_IT + 1)\n";
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
    report_file << "inputs      : " << cs.n << " (huecos antes de la ultima: " << sch.idle_before_last << ")\n";
    report_file << "first_valid : " << first_valid << "\n";
    report_file << "last_valid  : " << last_valid << "\n";
    report_file << "n_valid     : " << n_valid << " (esperadas " << cs.n << ")\n";
    report_file << "late        : " << n_late << "\n";
    report_file << "o_mag != py : " << n_bad << "\n";
    report_file << "status      : " << (ok ? "OK" : "INCOMPLETO") << "\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv()  << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir()    << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir()   << "\n";
}

} // namespace cordic_mag_tb

int main(int argc, char** argv)
{
    try {
        const auto args = cordic_mag_tb::parse_args(argc, argv);
        cordic_mag_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[cordic_mag_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}

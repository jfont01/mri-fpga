#include "cast_tb.hpp"

#include "rtlsim.hpp"

#include <ap_fixed.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace cast_tb {

using raw_in_t = ap_int<cast::NB_IN>;

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

static raw_in_t min_raw()
{
    raw_in_t value = 0;
    value[cast::NB_IN - 1] = 1;
    return value;
}

static raw_in_t max_raw()
{
    raw_in_t value = 0;

    for (int i = 0; i < cast::NB_IN - 1; ++i) {
        value[i] = 1;
    }

    value[cast::NB_IN - 1] = 0;

    return value;
}

static cast::in_t raw_to_input(const raw_in_t& raw)
{
    cast::in_t value;
    value.range(cast::NB_IN - 1, 0) = raw.range(cast::NB_IN - 1, 0);
    return value;
}

static raw_in_t rounding_half_lsb()
{
    raw_in_t value = 0;

    if constexpr (cast::LSB_TO_DROP > 0) {
        value[cast::LSB_TO_DROP - 1] = 1;
    }

    return value;
}

static raw_in_t gen_raw_input(int cycle)
{
    const raw_in_t half = rounding_half_lsb();

    /*
     * Deterministic stimulus set.
     *
     * Includes:
     * - zero
     * - small positive and negative values
     * - input min/max
     * - rounding boundary values when LSB_TO_DROP > 0
     * - pseudo-random signed sweep by natural ap_int wrapping
     */
    switch (cycle % 16) {
    case 0:
        return raw_in_t(0);

    case 1:
        return raw_in_t(1);

    case 2:
        return raw_in_t(-1);

    case 3:
        return max_raw();

    case 4:
        return min_raw();

    case 5:
        return half;

    case 6:
        return raw_in_t(half - 1);

    case 7:
        return raw_in_t(half + 1);

    case 8:
        return raw_in_t(-half);

    case 9:
        return raw_in_t(-half - 1);

    case 10:
        return raw_in_t(-half + 1);

    default:
        return raw_in_t((cycle * 37) - 211);
    }
}

static cast::in_t gen_word(int cycle)
{
    return raw_to_input(gen_raw_input(cycle));
}

void run_tb(const tb_args& args)
{
    const fs::path reports_dir = args.case_dir / "reports";

    fs::create_directories(reports_dir);

    std::ofstream report_file(reports_dir / "vm_summary.rpt");

    if (!report_file) {
        throw std::runtime_error("could not open report file");
    }

    rtl::ClockDomain clk;
    cast::cast_model dut;
    rtl::Simulator sim(clk);

    sim.add(dut);
    sim.init();

    /*
     * Vector trace manager.
     *
     * Generated files:
     *
     *   simulation/vectors/stimuli/in_ports.csv
     *   simulation/vectors/stimuli/in_ports/i_word.dat
     *
     *   simulation/vectors/expected/out_ports.csv
     *   simulation/vectors/expected/out_ports/o_word.dat
     *
     * Policy:
     *   CSV -> human-readable values
     *   DAT -> raw integer values for bit-accurate comparison
     *
     * For ap_fixed:
     *   DAT uses signed raw decimal integer.
     */
    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_word", dut.i_word);
    vectors.add_out_port("o_word", dut.o_word);

    for (int cycle = 0; cycle < args.n_cycles; ++cycle) {
        dut.i_word = gen_word(cycle);

        sim.cycle();

        vectors.sample(sim);
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : cast\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "n_cycles    : " << args.n_cycles << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "NB_IN       : " << cast::NB_IN << "\n";
    report_file << "NBF_IN      : " << cast::NBF_IN << "\n";
    report_file << "NB_OUT      : " << cast::NB_OUT << "\n";
    report_file << "NBF_OUT     : " << cast::NBF_OUT << "\n";
    report_file << "ROUND_MODE  : " << cast::ROUND_MODE << "\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv() << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir() << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir() << "\n";
}

} // namespace cast_tb

int main(int argc, char** argv)
{
    try {
        const auto args = cast_tb::parse_args(argc, argv);
        cast_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[cast_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}
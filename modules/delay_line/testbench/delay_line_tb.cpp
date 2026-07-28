#include "delay_line_tb.hpp"

#include "rtlsim.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace delay_line_tb {

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

    return args;
}

void run_tb(const tb_args& args)
{
    /*
     * Expected case layout:
     *
     *   build/<CASE>/
     *     binary/
     *     simulation/
     *       vectors/
     *         stimuli/
     *           in_ports.csv
     *           in_ports/
     *             <input_signal>.dat
     *         expected/
     *           out_ports.csv
     *           out_ports/
     *             <output_signal>.dat
     *     reports/
     *       vm_summary.rpt
     */

    const fs::path sim_dir      = args.case_dir / "simulation";
    const fs::path vectors_dir  = sim_dir / "vectors";
    const fs::path stimuli_dir  = vectors_dir / "stimuli";
    const fs::path expected_dir = vectors_dir / "expected";
    const fs::path reports_dir  = args.case_dir / "reports";

    const fs::path in_dat_dir  = stimuli_dir / "in_ports";
    const fs::path out_dat_dir = expected_dir / "out_ports";

    fs::create_directories(stimuli_dir);
    fs::create_directories(expected_dir);
    fs::create_directories(in_dat_dir);
    fs::create_directories(out_dat_dir);
    fs::create_directories(reports_dir);

    std::ofstream report_file(reports_dir / "vm_summary.rpt");

    if (!report_file) {
        throw std::runtime_error("could not open report file");
    }

    rtl::ClockDomain clk;
    delay_line::delay_line_model dut;
    rtl::Simulator sim(clk);

    sim.add(dut);
    sim.init();

    /*
     * CSV + DAT traces.
     *
     * These files are used by run_regression_sim.py:
     *
     *   stimuli/in_ports.csv
     *   stimuli/in_ports/<signal>.dat
     *
     *   expected/out_ports.csv
     *   expected/out_ports/<signal>.dat
     *
     * The generic template only writes the cycle column in the CSV files.
     * Add module-specific ports below when the model defines them.
     */

    rtl::CsvDatTrace in_ports_trace(
        (stimuli_dir / "in_ports.csv").string(),
        in_dat_dir.string()
    );

    rtl::CsvDatTrace out_ports_trace(
        (expected_dir / "out_ports.csv").string(),
        out_dat_dir.string()
    );

    /*
     * Input port trace examples:
     *
     * in_ports_trace.add_value("i_en", [&dut]() {
     *     return rtl::value_to_string(dut.i_en.value());
     * });
     *
     * in_ports_trace.add_value("i_word", [&dut]() {
     *     return rtl::value_to_string(dut.i_word.value());
     * });
     */

    /*
     * Output port trace examples:
     *
     * out_ports_trace.add_value("o_en", [&dut]() {
     *     return rtl::value_to_string(dut.o_en.value());
     * });
     *
     * out_ports_trace.add_value("o_word", [&dut]() {
     *     return rtl::value_to_string(dut.o_word.value());
     * });
     */

    for (int cycle = 0; cycle < args.n_cycles; ++cycle) {
        /*
         * Drive module-specific inputs here.
         *
         * Example:
         *
         * dut.i_en   = true;
         * dut.i_word = input_value;
         */

        sim.cycle();

        in_ports_trace.sample(sim.cycle_count());
        out_ports_trace.sample(sim.cycle_count());
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : delay_line\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "n_cycles    : " << args.n_cycles << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";
    report_file << "input_csv   : " << (stimuli_dir / "in_ports.csv") << "\n";
    report_file << "output_csv  : " << (expected_dir / "out_ports.csv") << "\n";
    report_file << "input_dat   : " << in_dat_dir << "\n";
    report_file << "output_dat  : " << out_dat_dir << "\n";
}

} // namespace delay_line_tb

int main(int argc, char** argv)
{
    try {
        const auto args = delay_line_tb::parse_args(argc, argv);
        delay_line_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[delay_line_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}

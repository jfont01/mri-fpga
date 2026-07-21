// cmul_karatsuba_tb.cpp
//
// Generador de vectores para el pipeline vm.
//
// Corre el golden C++ (cmul_karatsuba_model) y vuelca con VectorTraceFiles:
//
//   simulation/vectors/stimuli/in_ports/{i_1_re,i_1_im,i_2_re,i_2_im}.dat
//   simulation/vectors/expected/out_ports/{o_re,o_im}.dat
//
// Politica DAT (igual que cast_tb.cpp):
//   ap_fixed -> entero crudo decimal con signo
//
// El testbench SystemVerilog (cmul_karatsuba_tb.sv) reinyecta esos estimulos en el RTL
// y produce .../actual/out_ports/*.dat, que run_regression_vm compara linea
// a linea contra expected.

#include "cmul_karatsuba_tb.hpp"

#include "rtlsim.hpp"

#include <ap_fixed.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace cmul_karatsuba_tb {

using raw_in_t = ap_int<cmul_karatsuba::NB_IN>;

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

namespace {

raw_in_t min_raw()
{
    raw_in_t value = 0;
    value[cmul_karatsuba::NB_IN - 1] = 1;
    return value;
}

raw_in_t max_raw()
{
    raw_in_t value = 0;

    for (int i = 0; i < cmul_karatsuba::NB_IN - 1; ++i) {
        value[i] = 1;
    }

    value[cmul_karatsuba::NB_IN - 1] = 0;

    return value;
}

cmul_karatsuba::in_t raw_to_input(const raw_in_t& raw)
{
    cmul_karatsuba::in_t value;
    value.range(cmul_karatsuba::NB_IN - 1, 0) = raw.range(cmul_karatsuba::NB_IN - 1, 0);
    return value;
}

/*
 * Valor crudo que representa 1.0 en el formato de entrada, si es
 * representable (NBI_IN > 1). Si no lo es, devuelve el maximo.
 */
raw_in_t one_raw()
{
    if constexpr (cmul_karatsuba::NBI_IN > 1) {
        raw_in_t value = 0;
        value[cmul_karatsuba::NBF_IN] = 1;
        return value;
    }
    else {
        return max_raw();
    }
}

/*
 * LCG determinista (no depende de la implementacion de <random>, asi el
 * generador da lo mismo en cualquier toolchain).
 */
std::uint32_t lcg(std::uint32_t x)
{
    return x * 1664525u + 1013904223u;
}

raw_in_t pseudo_raw(int cycle, int lane)
{
    std::uint32_t s = static_cast<std::uint32_t>(cycle) * 4u + static_cast<std::uint32_t>(lane) + 1u;
    s = lcg(lcg(s));
    return raw_in_t(static_cast<int>(s));   // ap_int recorta y toma el signo natural
}

struct sample {
    raw_in_t a;
    raw_in_t b;
    raw_in_t c;
    raw_in_t d;
};

/*
 * Set de estimulos determinista.
 *
 * Cubre:
 * - cero
 * - +-1 LSB
 * - min/max de entrada (fuerzan saturacion en la salida)
 * - productos que saturan por ambas componentes
 * - 1.0 * 1.0 (si 1.0 es representable en el formato de entrada)
 * - solo-real x solo-imaginario (chequea el cruce de terminos)
 * - barrido pseudoaleatorio con signo
 */
sample gen_sample(int cycle)
{
    const raw_in_t zero = raw_in_t(0);
    const raw_in_t one  = one_raw();

    switch (cycle % 16) {
    case 0:  return { zero, zero, zero, zero };
    case 1:  return { raw_in_t(1), zero, raw_in_t(1), zero };
    case 2:  return { raw_in_t(-1), zero, raw_in_t(1), zero };
    case 3:  return { max_raw(), zero, max_raw(), zero };
    case 4:  return { min_raw(), zero, min_raw(), zero };
    case 5:  return { max_raw(), max_raw(), max_raw(), max_raw() };
    case 6:  return { min_raw(), min_raw(), max_raw(), max_raw() };
    case 7:  return { one, zero, one, zero };
    case 8:  return { zero, one, one, zero };            // j * 1  ->  j
    case 9:  return { zero, one, zero, one };            // j * j  -> -1
    case 10: return { one, one, one, raw_in_t(-1) };     // (1+j)(1-j) = 2
    case 11: return { max_raw(), zero, raw_in_t(1), zero };
    case 12: return { min_raw(), zero, raw_in_t(-1), zero };
    default:
        return {
            pseudo_raw(cycle, 0),
            pseudo_raw(cycle, 1),
            pseudo_raw(cycle, 2),
            pseudo_raw(cycle, 3),
        };
    }
}

} // namespace

void run_tb(const tb_args& args)
{
    const fs::path reports_dir = args.case_dir / "reports";
    fs::create_directories(reports_dir);

    std::ofstream report_file(reports_dir / "vm_summary.rpt");
    if (!report_file) {
        throw std::runtime_error("could not open report file");
    }

    rtl::ClockDomain clk;
    cmul_karatsuba::cmul_karatsuba_model dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    rtl::VectorTraceFiles vectors(args.case_dir);

    vectors.add_in_port("i_1_re", dut.i_1_re);
    vectors.add_in_port("i_1_im", dut.i_1_im);
    vectors.add_in_port("i_2_re", dut.i_2_re);
    vectors.add_in_port("i_2_im", dut.i_2_im);

    vectors.add_out_port("o_re", dut.o_re);
    vectors.add_out_port("o_im", dut.o_im);

    for (int cycle = 0; cycle < args.n_cycles; ++cycle) {
        const sample s = gen_sample(cycle);

        dut.i_1_re = raw_to_input(s.a);
        dut.i_1_im = raw_to_input(s.b);
        dut.i_2_re = raw_to_input(s.c);
        dut.i_2_im = raw_to_input(s.d);

        sim.cycle();

        vectors.sample(sim);
    }

    report_file << "VM SUMMARY\n";
    report_file << "==========\n\n";
    report_file << "module      : cmul_karatsuba\n";
    report_file << "case_dir    : " << args.case_dir << "\n";
    report_file << "n_cycles    : " << args.n_cycles << "\n";
    report_file << "cycles      : " << sim.cycle_count() << "\n";
    report_file << "\n";

    report_file << "parameters\n";
    report_file << "----------\n";
    report_file << "NB_IN       : " << cmul_karatsuba::NB_IN   << "\n";
    report_file << "NBF_IN      : " << cmul_karatsuba::NBF_IN  << "\n";
    report_file << "NB_OUT      : " << cmul_karatsuba::NB_OUT  << "\n";
    report_file << "NBF_OUT     : " << cmul_karatsuba::NBF_OUT << "\n";
    report_file << "NB_FULL     : " << cmul_karatsuba::NB_FULL << "\n";
    report_file << "\n";

    report_file << "input_csv   : " << vectors.in_ports_csv()  << "\n";
    report_file << "output_csv  : " << vectors.out_ports_csv() << "\n";
    report_file << "input_dat   : " << vectors.in_dat_dir()    << "\n";
    report_file << "output_dat  : " << vectors.out_dat_dir()   << "\n";
}

} // namespace cmul_karatsuba_tb

int main(int argc, char** argv)
{
    try {
        const auto args = cmul_karatsuba_tb::parse_args(argc, argv);
        cmul_karatsuba_tb::run_tb(args);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[cmul_karatsuba_tb] ERROR: " << e.what() << "\n";
        return 1;
    }
}
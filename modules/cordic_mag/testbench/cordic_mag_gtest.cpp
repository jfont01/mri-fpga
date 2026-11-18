// cordic_mag_gtest.cpp -- cordic_mag contra model/mac_cordic_fxp.py (m_to_out),
// bit a bit y ciclo a ciclo.
//
// Los vectores los escribe model/gen_cordic_mag_vectors.py; el estimulo ciclo a
// ciclo (huecos y basura) lo arma cordic_mag_stimulus.hpp, el mismo que usa el
// tb de sim. La carpeta del caso es CORDIC_MAG_VEC_DIR (clave VEC_DIR del JSON).
//
// Que se chequea por caso (diseno D2, C6):
//   - case.txt coincide con los parametros compilados (si no: regenerar);
//   - ningun chequeo de rango del modelo dispara (std::range_error);
//   - tantas salidas como entradas;
//   - cada salida exactamente LATENCY ciclos despues de su entrada, con la
//     convencion de LatencyMatchesFormula de la FFT: la entrada se pone antes
//     del sim.cycle() numero c y la salida se ve despues del numero c + LATENCY;
//   - o_mag y x_n (dbg_xn) iguales al Python, bit a bit;
//   - o_mag quieto entre salidas, 0 antes de la primera y en NB_OUT bits.
//
// La simulacion corre una vez, adentro del primer test que la pide (no en
// SetUpTestSuite: una excepcion ahi deja los tests en skipped, ver
// doc/08_trampas.md), y guarda las salidas de cada ciclo.
#include "cordic_mag.hpp"
#include "cordic_mag_stimulus.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace cordic_mag;
namespace stim = cordic_mag_stimulus;

struct SimRun {
    std::string err_load;       // no se pudo leer el caso
    std::string mismatch;       // parametros de case.txt != compilados
    std::string err_sim;        // excepcion del modelo (chequeo de rango)
    stim::Case cs;
    stim::Schedule sch;
    // salidas, una por ciclo (despues del sim.cycle() numero c)
    std::vector<bool>   o_valid;
    std::vector<word_t> o_mag;
    std::vector<word_t> xn;
    std::vector<long>   out_cycle;   // ciclos con o_valid = 1, en orden
};

SimRun simulate()
{
    SimRun r;
    try {
        r.cs = stim::load(stim::vec_dir());
        r.mismatch = stim::param_mismatch(r.cs);
        r.sch = stim::build(r.cs);
    }
    catch (const std::exception& e) {
        r.err_load = e.what();
        return r;
    }

    rtl::ClockDomain clk;
    cordic_mag_model dut;
    rtl::Simulator sim(clk);
    sim.add(dut);
    sim.init();

    const std::size_t nc = r.sch.cycles.size();
    r.o_valid.reserve(nc);
    r.o_mag.reserve(nc);
    r.xn.reserve(nc);
    try {
        for (std::size_t c = 0; c < nc; ++c) {
            const stim::Cycle& in = r.sch.cycles[c];
            dut.i_valid = bit_t(in.valid);
            dut.i_m_re  = in.re;
            dut.i_m_im  = in.im;

            sim.cycle();

            const bool v = dut.o_valid.value().to_bool();
            r.o_valid.push_back(v);
            r.o_mag.push_back(dut.o_mag.value());
            r.xn.push_back(dut.dbg_xn());
            if (v) r.out_cycle.push_back(static_cast<long>(c));
        }
    }
    catch (const std::exception& e) {
        r.err_sim = e.what();
    }
    return r;
}

const SimRun& sim_run()
{
    static const SimRun r = simulate();
    return r;
}

// Lo que todos los tests necesitan: el caso leido, con los parametros
// compilados, y la simulacion completa.
#define REQUIRE_RUN()                                                                \
    const SimRun& R = sim_run();                                                     \
    ASSERT_TRUE(R.err_load.empty()) << R.err_load;                                   \
    ASSERT_TRUE(R.mismatch.empty()) << R.mismatch << ": " << stim::regen_hint();     \
    ASSERT_TRUE(R.err_sim.empty()) << "la simulacion se corto: " << R.err_sim

std::string m_str(const stim::Case& cs, std::size_t k)
{
    return "M[" + std::to_string(k) + "] = (" + std::to_string(cs.m_re[k]) + ", " + std::to_string(cs.m_im[k]) + ")";
}

// ------------------------------------------------------------------ tests

TEST(CordicMag, CaseMatchesCompiledParams)
{
    const SimRun& R = sim_run();
    ASSERT_TRUE(R.err_load.empty()) << R.err_load;
    EXPECT_TRUE(R.mismatch.empty()) << R.mismatch << ": " << stim::regen_hint();
}

TEST(CordicMag, NoRangeCheckFires)
{
    const SimRun& R = sim_run();
    ASSERT_TRUE(R.err_load.empty()) << R.err_load;
    EXPECT_TRUE(R.err_sim.empty()) << R.err_sim;
    std::cout << "[cordic_mag] " << R.cs.dir << ": n = " << R.cs.n << ", ciclos = " << R.sch.cycles.size()
              << ", huecos antes de la ultima muestra = " << R.sch.idle_before_last
              << ", GAP_PCT = " << stim::GAP_PCT << ", LATENCY = " << LATENCY << "\n";
}

TEST(CordicMag, OneOutputPerInput)
{
    REQUIRE_RUN();
    EXPECT_EQ(R.out_cycle.size(), R.cs.n);
}

TEST(CordicMag, LatencyMatchesFormula)
{
    REQUIRE_RUN();
    EXPECT_EQ(static_cast<int>(LATENCY), N_IT + 1);
    EXPECT_EQ(static_cast<int>(N_STAGES), N_IT + 2);
    ASSERT_EQ(R.out_cycle.size(), R.cs.n);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t k = 0; k < R.cs.n; ++k) {
        const long want = R.sch.in_cycle[k] + LATENCY;
        if (R.out_cycle[k] != want && bad++ == 0) {
            first = "salida " + std::to_string(k) + ": entro en el ciclo " + std::to_string(R.sch.in_cycle[k]) +
                    ", salio en el " + std::to_string(R.out_cycle[k]) + " (esperado " + std::to_string(want) + ")";
        }
    }
    EXPECT_EQ(bad, 0u) << first;
}

// La salida k-esima (en orden) contra el Python. No depende de la latencia:
// si la latencia esta mal, falla LatencyMatchesFormula y esta no.
TEST(CordicMag, MagMatchesPythonBitwise)
{
    REQUIRE_RUN();
    const std::size_t n = std::min(R.out_cycle.size(), R.cs.n);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t k = 0; k < n; ++k) {
        const word_t got = R.o_mag[static_cast<std::size_t>(R.out_cycle[k])];
        if (got != R.cs.out_ref[k] && bad++ == 0) {
            first = m_str(R.cs, k) + ": o_mag = " + std::to_string(got) + ", Python " + std::to_string(R.cs.out_ref[k]) +
                    " (x_n Python " + std::to_string(R.cs.xn_ref[k]) + ")";
        }
    }
    EXPECT_EQ(bad, 0u) << first;
    EXPECT_EQ(n, R.cs.n);
}

TEST(CordicMag, XnMatchesPythonBitwise)
{
    REQUIRE_RUN();
    const std::size_t n = std::min(R.out_cycle.size(), R.cs.n);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t k = 0; k < n; ++k) {
        const word_t got = R.xn[static_cast<std::size_t>(R.out_cycle[k])];
        if (got != R.cs.xn_ref[k] && bad++ == 0) {
            first = m_str(R.cs, k) + ": x_n = " + std::to_string(got) + ", Python " + std::to_string(R.cs.xn_ref[k]);
        }
    }
    EXPECT_EQ(bad, 0u) << first;
    EXPECT_EQ(n, R.cs.n);
}

// El vm compara todos los ciclos: entre salidas, o_mag tiene que quedarse con el
// ultimo valor (clock enable), y antes de la primera, en 0 (sin reset, valor
// inicial 0). Con huecos, en esos ciclos i_m lleva basura.
TEST(CordicMag, OutputHoldsBetweenOutputs)
{
    REQUIRE_RUN();
    word_t prev = 0;
    std::size_t moved = 0, out_of_range = 0;
    std::string first;
    for (std::size_t c = 0; c < R.o_mag.size(); ++c) {
        const word_t m = R.o_mag[c];
        if (m < 0 || m > umax(NB_OUT)) {
            ++out_of_range;
        }
        if (!R.o_valid[c] && m != prev && moved++ == 0) {
            first = "ciclo " + std::to_string(c) + " sin o_valid: o_mag paso de " + std::to_string(prev) + " a " +
                    std::to_string(m);
        }
        prev = m;
    }
    EXPECT_EQ(moved, 0u) << first;
    EXPECT_EQ(out_of_range, 0u) << "o_mag fuera de NB_OUT = " << NB_OUT << " bits sin signo";
}

// La regla de redondeo, a mano (no depende del caso): x / 2^g con empates.
TEST(CordicMag, RoundShiftTies)
{
    // half-up: los empates suben
    EXPECT_EQ(round_shift(3, 1, false), 2);      // 1,5
    EXPECT_EQ(round_shift(5, 1, false), 3);      // 2,5
    EXPECT_EQ(round_shift(10, 2, false), 3);     // 2,5
    EXPECT_EQ(round_shift(9, 2, false), 2);      // 2,25
    EXPECT_EQ(round_shift(11, 2, false), 3);     // 2,75
    // convergente: los empates van al par
    EXPECT_EQ(round_shift(3, 1, true), 2);       // 1,5 -> 2
    EXPECT_EQ(round_shift(5, 1, true), 2);       // 2,5 -> 2
    EXPECT_EQ(round_shift(7, 1, true), 4);       // 3,5 -> 4
    EXPECT_EQ(round_shift(10, 2, true), 2);      // 2,5 -> 2
    EXPECT_EQ(round_shift(9, 2, true), 2);       // 2,25
    EXPECT_EQ(round_shift(11, 2, true), 3);      // 2,75
    // g = 0: sin redondeo
    EXPECT_EQ(round_shift(12345, 0, false), 12345);
    EXPECT_EQ(round_shift(12345, 0, true), 12345);
}

}  // namespace

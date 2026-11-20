// unfold_gtest.cpp -- unfold contra model/mac_cordic_fxp.py (mac y p_to_m) y la
// regla de cierre de la decision 13, bit a bit y ciclo a ciclo.
//
// Los vectores los escribe model/gen_unfold_vectors.py; el estimulo ciclo a
// ciclo (huecos y basura) lo arma unfold_stimulus.hpp, el mismo que usa el tb
// de sim. La carpeta del caso es UNFOLD_VEC_DIR (clave VEC_DIR del JSON).
//
// Que se chequea por caso (diseno D2, U7):
//   - case.txt coincide con los parametros compilados (si no: regenerar);
//   - ningun chequeo de rango del modelo dispara (std::range_error);
//   - una salida por posicion cerrada (segun la regla de cierre del generador);
//   - cada salida exactamente LATENCY ciclos despues de la muestra que cerro la
//     posicion, con la convencion de LatencyMatchesFormula de la FFT: la muestra
//     se pone antes del sim.cycle() numero c y la salida se ve despues del
//     numero c + LATENCY;
//   - M (o_m), P (dbg_p) y o_err iguales al Python, bit a bit;
//   - o_m, o_err y P quietos entre salidas, en 0 antes de la primera y M en
//     NB_M bits;
//   - y, sin depender del caso: que los chequeos de rango disparen con una
//     muestra valida fuera de su ancho (y no con basura), la regla de redondeo y
//     saturacion a mano y NB_P con una cuenta independiente.
// Con UNFOLD_E2E = 1 (caso punta a punta, clave E2E del JSON), ademas: unfold ->
// serializador del banco -> cordic_mag (con sus defaults) contra
// mac_cordic_fxp.output_stage, con los datos del fantoma (U8).
//
// La simulacion corre una vez, adentro del primer test que la pide (no en
// SetUpTestSuite: una excepcion ahi deja los tests en skipped, ver
// doc/08_trampas.md), y guarda las salidas de cada ciclo.
#include "unfold.hpp"
#include "unfold_stimulus.hpp"

#if UNFOLD_E2E
#include "../../cordic_mag/cpp/cordic_mag.hpp"
#include <deque>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace unfold;
namespace stim = unfold_stimulus;
using stim::arr_t;

struct SimRun {
    std::string err_load;       // no se pudo leer el caso
    std::string mismatch;       // parametros de case.txt != compilados
    std::string err_sim;        // excepcion del modelo (chequeo de rango)
    stim::Case cs;
    stim::Schedule sch;
    // salidas, una por ciclo (despues del sim.cycle() numero c)
    std::vector<bool>  o_valid, o_err;
    std::vector<arr_t> m_re, m_im, p_re, p_im;
    std::vector<long>  out_cycle;   // ciclos con o_valid = 1, en orden
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
    unfold_model dut;
    rtl::Simulator sim(clk);
    sim.add(dut);
    sim.init();

    const std::size_t nc = r.sch.cycles.size();
    for (auto* v : {&r.m_re, &r.m_im, &r.p_re, &r.p_im}) v->reserve(nc);
    try {
        for (std::size_t c = 0; c < nc; ++c) {
            stim::drive(dut, r.sch.cycles[c]);

            sim.cycle();

            const bool v = dut.o_valid.value().to_bool();
            r.o_valid.push_back(v);
            r.o_err.push_back(dut.o_err.value().to_bool());
            arr_t mr {}, mi {}, pr {}, pi {};
            for (int a = 0; a < AF; ++a) {
                const std::size_t k = static_cast<std::size_t>(a);
                mr[k] = dut.o_m_re[k].value();
                mi[k] = dut.o_m_im[k].value();
                pr[k] = dut.dbg_p_re(a);
                pi[k] = dut.dbg_p_im(a);
            }
            r.m_re.push_back(mr);
            r.m_im.push_back(mi);
            r.p_re.push_back(pr);
            r.p_im.push_back(pi);
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

std::string arr_str(const arr_t& v)
{
    std::string s = "(";
    for (std::size_t a = 0; a < v.size(); ++a) {
        s += (a ? ", " : "") + std::to_string(v[a]);
    }
    return s + ")";
}

// Descripcion de la salida k para los mensajes: P y D de la posicion.
std::string out_str(const stim::Case& cs, std::size_t k)
{
    const stim::Sample& last = cs.stream[static_cast<std::size_t>(cs.close[k])];
    return "salida " + std::to_string(k) + " (cierra la muestra " + std::to_string(cs.close[k]) + ", D = " +
           std::to_string(last.d) + ", P_re = " + arr_str(cs.p_re[k]) + ", P_im = " + arr_str(cs.p_im[k]) + ")";
}

// Compara un arreglo por salida (re e im) del modelo contra lo esperado.
void expect_bitwise(const SimRun& R, const std::vector<arr_t>& got_re, const std::vector<arr_t>& got_im,
                    const std::vector<arr_t>& want_re, const std::vector<arr_t>& want_im, const char* what)
{
    const std::size_t n = std::min(R.out_cycle.size(), R.cs.n_out);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t c = static_cast<std::size_t>(R.out_cycle[k]);
        if ((got_re[c] != want_re[k] || got_im[c] != want_im[k]) && bad++ == 0) {
            first = out_str(R.cs, k) + ": " + what + " = " + arr_str(got_re[c]) + " + j " + arr_str(got_im[c]) +
                    ", Python " + arr_str(want_re[k]) + " + j " + arr_str(want_im[k]);
        }
    }
    EXPECT_EQ(bad, 0u) << first;
    EXPECT_EQ(n, R.cs.n_out);
}

// ------------------------------------------------------------------ tests

TEST(Unfold, CaseMatchesCompiledParams)
{
    const SimRun& R = sim_run();
    ASSERT_TRUE(R.err_load.empty()) << R.err_load;
    EXPECT_TRUE(R.mismatch.empty()) << R.mismatch << ": " << stim::regen_hint();
}

TEST(Unfold, NoRangeCheckFires)
{
    const SimRun& R = sim_run();
    ASSERT_TRUE(R.err_load.empty()) << R.err_load;
    EXPECT_TRUE(R.err_sim.empty()) << R.err_sim;
    std::cout << "[unfold] " << R.cs.dir << ": muestras = " << R.cs.n << ", posiciones cerradas = " << R.cs.n_out
              << " (con err: " << std::count(R.cs.err.begin(), R.cs.err.end(), 1) << "), ciclos = "
              << R.sch.cycles.size() << ", huecos antes de la ultima muestra = " << R.sch.idle_before_last
              << ", GAP_PCT = " << stim::GAP_PCT << ", LATENCY = " << LATENCY << ", NB_P = " << NB_P << "\n";
}

TEST(Unfold, OneOutputPerClosedPosition)
{
    REQUIRE_RUN();
    EXPECT_EQ(R.out_cycle.size(), R.cs.n_out);
}

TEST(Unfold, LatencyMatchesFormula)
{
    REQUIRE_RUN();
    EXPECT_EQ(static_cast<int>(N_STAGES), 4);
    EXPECT_EQ(static_cast<int>(LATENCY), N_STAGES - 1);
    ASSERT_EQ(R.out_cycle.size(), R.cs.n_out);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t k = 0; k < R.cs.n_out; ++k) {
        const long in = R.sch.in_cycle[static_cast<std::size_t>(R.cs.close[k])];
        const long want = in + LATENCY;
        if (R.out_cycle[k] != want && bad++ == 0) {
            first = out_str(R.cs, k) + ": la muestra que cierra entro en el ciclo " + std::to_string(in) +
                    ", salio en el " + std::to_string(R.out_cycle[k]) + " (esperado " + std::to_string(want) + ")";
        }
    }
    EXPECT_EQ(bad, 0u) << first;
}

// NB_P con una cuenta independiente de clog2: el menor b con 2^b >= L.
TEST(Unfold, NbPMatchesFormula)
{
    int b = 0;
    while ((1 << b) < L) ++b;
    EXPECT_EQ(static_cast<int>(NB_P), NB_A + NB_U + 1 + b);
    EXPECT_EQ(static_cast<int>(S_MIN), -G);
    EXPECT_EQ(static_cast<int>(S_MAX), (1 << NB_D) - 1 - G);
}

// Las salidas k-esimas (en orden) contra el Python. No dependen de la latencia:
// si la latencia esta mal, falla LatencyMatchesFormula y estas no.
TEST(Unfold, MMatchesPythonBitwise)
{
    REQUIRE_RUN();
    expect_bitwise(R, R.m_re, R.m_im, R.cs.m_re, R.cs.m_im, "M");
}

TEST(Unfold, PMatchesPythonBitwise)
{
    REQUIRE_RUN();
    expect_bitwise(R, R.p_re, R.p_im, R.cs.p_re, R.cs.p_im, "P");
}

TEST(Unfold, ErrMatchesExpected)
{
    REQUIRE_RUN();
    const std::size_t n = std::min(R.out_cycle.size(), R.cs.n_out);
    std::size_t bad = 0, n_err = 0;
    std::string first;
    for (std::size_t k = 0; k < n; ++k) {
        const bool got = R.o_err[static_cast<std::size_t>(R.out_cycle[k])];
        n_err += got ? 1 : 0;
        if (got != (R.cs.err[k] != 0) && bad++ == 0) {
            first = out_str(R.cs, k) + ": o_err = " + std::to_string(got) + ", esperado " + std::to_string(R.cs.err[k]);
        }
    }
    EXPECT_EQ(bad, 0u) << first;
    EXPECT_EQ(n, R.cs.n_out);
    EXPECT_EQ(static_cast<long>(n_err), R.cs.get("n_err"));
}

// El vm compara todos los ciclos: entre salidas, o_m y o_err tienen que quedarse
// con el ultimo valor (clock enable), y antes de la primera, en 0 (sin reset,
// valor inicial 0). Lo mismo el P de depuracion. Con huecos, en esos ciclos las
// entradas llevan basura.
TEST(Unfold, OutputHoldsBetweenOutputs)
{
    REQUIRE_RUN();
    arr_t z {};
    arr_t pmr = z, pmi = z, ppr = z, ppi = z;
    bool perr = false;
    std::size_t moved = 0, out_of_range = 0;
    std::string first;
    for (std::size_t c = 0; c < R.o_valid.size(); ++c) {
        for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
            if (R.m_re[c][a] < smin(NB_M) || R.m_re[c][a] > smax(NB_M) || R.m_im[c][a] < smin(NB_M) ||
                R.m_im[c][a] > smax(NB_M)) {
                ++out_of_range;
            }
        }
        const bool same = R.m_re[c] == pmr && R.m_im[c] == pmi && R.p_re[c] == ppr && R.p_im[c] == ppi &&
                          R.o_err[c] == perr;
        if (!R.o_valid[c] && !same && moved++ == 0) {
            first = "ciclo " + std::to_string(c) + " sin o_valid: o_m_re paso de " + arr_str(pmr) + " a " +
                    arr_str(R.m_re[c]) + ", o_err de " + std::to_string(perr) + " a " + std::to_string(R.o_err[c]) +
                    " (o P, o o_m_im)";
        }
        pmr = R.m_re[c];
        pmi = R.m_im[c];
        ppr = R.p_re[c];
        ppi = R.p_im[c];
        perr = R.o_err[c];
    }
    EXPECT_EQ(moved, 0u) << first;
    EXPECT_EQ(out_of_range, 0u) << "o_m fuera de NB_M = " << NB_M << " bits con signo";
}

// Los chequeos de rango (P5) tienen que disparar con una muestra valida fuera de
// su ancho, y no con los extremos del rango ni con basura fuera de rango cuando
// i_valid = 0. Sin esto, un chequeo mas laxo que el ancho del RTL pasaria igual:
// los vectores siempre estan en rango. Un modelo nuevo por prueba.
TEST(Unfold, RangeChecksFire)
{
    const auto fires = [](bool valid, const std::function<void(unfold_model&)>& set) {
        rtl::ClockDomain clk;
        unfold_model dut;
        rtl::Simulator sim(clk);
        sim.add(dut);
        sim.init();
        dut.i_valid = bit_t(valid);
        dut.i_last = bit_t(L == 1);
        set(dut);
        try {
            sim.cycle();
        }
        catch (const std::range_error&) {
            return true;
        }
        return false;
    };
    const std::size_t a0 = 0, a1 = static_cast<std::size_t>(AF - 1);
    const std::vector<std::pair<const char*, std::function<void(unfold_model&)>>> bad = {
        {"i_a_re = max + 1", [](unfold_model& d) { d.i_a_re = smax(NB_A) + 1; }},
        {"i_a_re = min - 1", [](unfold_model& d) { d.i_a_re = smin(NB_A) - 1; }},
        {"i_a_im = max + 1", [](unfold_model& d) { d.i_a_im = smax(NB_A) + 1; }},
        {"i_u_re[AF-1] = max + 1", [a1](unfold_model& d) { d.i_u_re[a1] = smax(NB_U) + 1; }},
        {"i_u_im[0] = min - 1", [a0](unfold_model& d) { d.i_u_im[a0] = smin(NB_U) - 1; }},
        {"i_d = 2^NB_D", [](unfold_model& d) { d.i_d = umax(NB_D) + 1; }},
        {"i_d = -1", [](unfold_model& d) { d.i_d = word_t(-1); }},
    };
    for (const auto& b : bad) {
        EXPECT_TRUE(fires(true, b.second)) << b.first << " con i_valid = 1 no disparo el chequeo de rango";
        EXPECT_FALSE(fires(false, b.second)) << b.first << " con i_valid = 0 disparo el chequeo de rango";
    }
    EXPECT_FALSE(fires(true, [a0, a1](unfold_model& d) {
        d.i_a_re = smin(NB_A);
        d.i_a_im = smax(NB_A);
        d.i_u_re[a0] = smin(NB_U);
        d.i_u_im[a1] = smax(NB_U);
        d.i_d = umax(NB_D);
    })) << "los extremos del rango dispararon el chequeo";
}

// La regla de redondeo y saturacion, a mano (no depende del caso).
TEST(Unfold, RoundShiftAndSat)
{
    // half-up: los empates suben, tambien los negativos
    EXPECT_EQ(round_shift(3, 1, false), 2);       // 1,5
    EXPECT_EQ(round_shift(5, 1, false), 3);       // 2,5
    EXPECT_EQ(round_shift(-3, 1, false), -1);     // -1,5
    EXPECT_EQ(round_shift(-5, 1, false), -2);     // -2,5
    EXPECT_EQ(round_shift(-9, 2, false), -2);     // -2,25
    EXPECT_EQ(round_shift(-11, 2, false), -3);    // -2,75
    // convergente: los empates van al par
    EXPECT_EQ(round_shift(3, 1, true), 2);        // 1,5 -> 2
    EXPECT_EQ(round_shift(5, 1, true), 2);        // 2,5 -> 2
    EXPECT_EQ(round_shift(-3, 1, true), -2);      // -1,5 -> -2
    EXPECT_EQ(round_shift(-5, 1, true), -2);      // -2,5 -> -2
    EXPECT_EQ(round_shift(-7, 1, true), -4);      // -3,5 -> -4
    EXPECT_EQ(round_shift(-9, 2, true), -2);      // -2,25
    EXPECT_EQ(round_shift(-11, 2, true), -3);     // -2,75
    // s grande: la esquina de H31 y su opuesto
    EXPECT_EQ(round_shift(word_t(1) << 34, 35, false), 1);    // 0,5 -> 1
    EXPECT_EQ(round_shift(word_t(1) << 34, 35, true), 0);     // 0,5 -> 0
    EXPECT_EQ(round_shift(-(word_t(1) << 34), 35, false), 0); // -0,5 -> 0
    EXPECT_EQ(round_shift(-(word_t(1) << 34), 35, true), 0);
    EXPECT_EQ(round_shift(-1, 62, false), 0);
    EXPECT_EQ(round_shift(-1, 62, true), 0);
    // s <= 0: corrimiento a la izquierda exacto
    EXPECT_EQ(round_shift(-7, 0, false), -7);
    EXPECT_EQ(round_shift(-7, -5, true), -224);
    EXPECT_EQ(round_shift(word_t(1) << 34, -5, false), word_t(1) << 39);
    // saturacion
    EXPECT_EQ(sat(word_t(1) << 21, 22), (word_t(1) << 21) - 1);
    EXPECT_EQ(sat(-(word_t(1) << 21), 22), -(word_t(1) << 21));
    EXPECT_EQ(sat(-(word_t(1) << 21) - 1, 22), -(word_t(1) << 21));
}

#if UNFOLD_E2E
// ------------------------------------------------------------ punta a punta
//
// unfold -> serializador -> cordic_mag. El serializador es del banco (no es un
// modulo): cuando unfold saca una posicion, encola sus AF valores de M (a = 0,
// 1, ...); en cada ciclo, si la cola no esta vacia, le pasa el primero al CORDIC
// con i_valid = 1. La salida del CORDIC, en orden, tiene que ser
// mac_cordic_fxp.output_stage(P) de cada posicion y cada a.

static_assert(static_cast<int>(cordic_mag::NB_M) == static_cast<int>(NB_M), "E2E: NB_M de unfold y de cordic_mag tienen que coincidir");
static_assert(static_cast<int>(cordic_mag::G) == static_cast<int>(G), "E2E: G de unfold y de cordic_mag tienen que coincidir");

struct E2eRun {
    std::string err_load, mismatch, err_sim;
    stim::Case cs;
    std::vector<word_t> mag, xn;   // salidas del CORDIC, en orden
    std::size_t fifo_max = 0;
    std::size_t cycles = 0;
};

E2eRun simulate_e2e()
{
    E2eRun r;
    stim::Schedule sch;
    try {
        r.cs = stim::load(stim::vec_dir());
        r.mismatch = stim::param_mismatch(r.cs);
        const auto chk = [&](const char* key, long compiled) {
            if (r.cs.get(key) != compiled) {
                r.mismatch += std::string(r.mismatch.empty() ? "" : ", ") + key + " = " +
                              std::to_string(r.cs.get(key)) + " en case.txt y " + std::to_string(compiled) +
                              " en cordic_mag";
            }
        };
        chk("n_it", cordic_mag::N_IT);
        chk("nb_out", cordic_mag::NB_OUT);
        chk("round_conv_out", cordic_mag::ROUND_CONV);
        sch = stim::build(r.cs);
    }
    catch (const std::exception& e) {
        r.err_load = e.what();
        return r;
    }

    rtl::ClockDomain clk;
    unfold_model u;
    cordic_mag::cordic_mag_model cm;
    rtl::Simulator sim(clk);
    sim.add(u);
    sim.add(cm);
    sim.init();

    std::deque<std::array<word_t, 2>> fifo;
    const std::size_t want = r.cs.n_out * static_cast<std::size_t>(AF);
    const std::size_t cap = sch.cycles.size() + want + cordic_mag::N_STAGES + 16;
    const stim::Cycle idle {false, stim::Sample {}, -1};
    try {
        for (std::size_t c = 0; c < cap && r.mag.size() < want; ++c) {
            stim::drive(u, c < sch.cycles.size() ? sch.cycles[c] : idle);
            if (!fifo.empty()) {
                cm.i_valid = cordic_mag::bit_t(true);
                cm.i_m_re = fifo.front()[0];
                cm.i_m_im = fifo.front()[1];
                fifo.pop_front();
            }
            else {
                cm.i_valid = cordic_mag::bit_t(false);
                cm.i_m_re = word_t(0);
                cm.i_m_im = word_t(0);
            }

            sim.cycle();
            ++r.cycles;

            if (u.o_valid.value().to_bool()) {
                for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
                    fifo.push_back({u.o_m_re[a].value(), u.o_m_im[a].value()});
                }
                r.fifo_max = std::max(r.fifo_max, fifo.size());
            }
            if (cm.o_valid.value().to_bool()) {
                r.mag.push_back(cm.o_mag.value());
                r.xn.push_back(cm.dbg_xn());
            }
        }
    }
    catch (const std::exception& e) {
        r.err_sim = e.what();
    }
    return r;
}

const E2eRun& e2e_run()
{
    static const E2eRun r = simulate_e2e();
    return r;
}

#define REQUIRE_E2E()                                                                \
    const E2eRun& E = e2e_run();                                                     \
    ASSERT_TRUE(E.err_load.empty()) << E.err_load;                                   \
    ASSERT_TRUE(E.mismatch.empty()) << E.mismatch << ": " << stim::regen_hint();     \
    ASSERT_TRUE(E.err_sim.empty()) << "la simulacion se corto: " << E.err_sim

TEST(UnfoldE2e, CaseMatchesCordicParams)
{
    REQUIRE_E2E();
    EXPECT_EQ(E.cs.get("e2e"), 1);
    std::cout << "[unfold e2e] " << E.cs.dir << ": posiciones = " << E.cs.n_out << ", magnitudes = " << E.mag.size()
              << ", ciclos = " << E.cycles << ", cola maxima del serializador = " << E.fifo_max
              << ", D = " << E.cs.get("d_phantom") << ", maximo = " << E.cs.get("mag_max") << "\n";
}

void expect_e2e(const E2eRun& E, const std::vector<word_t>& got, const std::vector<arr_t>& want, const char* what)
{
    const std::size_t na = static_cast<std::size_t>(AF);
    ASSERT_EQ(got.size(), E.cs.n_out * na);
    std::size_t bad = 0;
    std::string first;
    for (std::size_t j = 0; j < got.size(); ++j) {
        const std::size_t k = j / na, a = j % na;
        if (got[j] != want[k][a] && bad++ == 0) {
            first = out_str(E.cs, k) + ", a = " + std::to_string(a) + ": " + what + " = " + std::to_string(got[j]) +
                    ", Python " + std::to_string(want[k][a]) + " (M = " + std::to_string(E.cs.m_re[k][a]) + " + j " +
                    std::to_string(E.cs.m_im[k][a]) + ")";
        }
    }
    EXPECT_EQ(bad, 0u) << first;
}

TEST(UnfoldE2e, MagMatchesOutputStage)
{
    REQUIRE_E2E();
    expect_e2e(E, E.mag, E.cs.mag, "o_mag");
}

TEST(UnfoldE2e, XnMatchesOutputStage)
{
    REQUIRE_E2E();
    expect_e2e(E, E.xn, E.cs.xn, "x_n");
}
#endif

}  // namespace

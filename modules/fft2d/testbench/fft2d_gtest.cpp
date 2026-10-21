// -----------------------------------------------------------------------------
// fft2d_gtest.cpp -- tests del MODELO C++ de la fft2d (fft2d_model), a tamano
// chico. La configuracion sale de los macros FFT2D_* (el JSON los pasa con
// auto_defines); fft2d::Params{} los toma.
//
// Tres niveles:
//   1. Plomeria: la salida es, bit a bit, la composicion de dos modelos 1D
//      sueltos (filas por el nucleo 1, columnas por el nucleo 2). Separa los
//      errores de orden, buffers y tiempos de los aritmeticos.
//   2. Matematica: contra la IDFT 2D en double, en el orden del hardware
//      (o = t1 * N2 + t2 -> x = br(t1), p = br(t2)) y por 2^-(S1+S2). En modo
//      double el error tiene que ser de redondeo; en fixed, SQNR minima.
//   3. Tiempos: primer o_valid en latency(), salida continua, un o_last por
//      bobina, y lo mismo con huecos entre filas a la entrada.
//
// El estimulo es el mismo que usa el testbench de sim para el caso
// (fft2d_stimulus.hpp: ruido, impulso, tono o archivo). Parametros propios:
//   FFT2D_N_COILS       bobinas por corrida                     (default 3)
//   FFT2D_MIN_SQNR_DB   SQNR minima en fixed contra la IDFT 2D   (default 40)
// -----------------------------------------------------------------------------

#include "fft2d.hpp"
#include "fft2d_stimulus.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>

#ifndef FFT2D_MIN_SQNR_DB
#define FFT2D_MIN_SQNR_DB 40
#endif

namespace {

using cd = std::complex<double>;
using fft2d::in_t;
using fft2d::out_t;

constexpr int    N_COILS = FFT2D_N_COILS;
constexpr double PI      = 3.14159265358979323846;

int bitrev(int v, int bits)
{
    int r = 0;
    for (int b = 0; b < bits; ++b) {
        r = (r << 1) | ((v >> b) & 1);
    }
    return r;
}

int ilog2(int n)
{
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

// k-space de todas las bobinas, ya cuantizado al formato de entrada:
// kspace[l][2 * (k' * N1 + r) + {0, 1}]
std::vector<std::vector<in_t>> make_kspace(const fft2d::Params& p)
{
    return fft2d_stimulus::generate(p, N_COILS);
}

struct SimRun {
    std::vector<cd>   out;          // muestras validas, en orden
    std::vector<long> valid_cycle;  // ciclo de cada muestra valida
    std::vector<long> last_cycle;   // ciclos con o_last
};

/*
 * Corre el modelo 2D con todas las bobinas. gap_seed = 0: entrada continua;
 * si no, huecos de 0..3 ciclos entre filas (y entre bobinas).
 */
SimRun run_2d(const fft2d::Params& p, const std::vector<std::vector<in_t>>& ks, unsigned gap_seed)
{
    rtl::ClockDomain   clk;
    fft2d::fft2d_model dut(p);
    rtl::Simulator     sim(clk);
    sim.add(dut);
    sim.init();

    // secuencia de entrada: (valid, re, im) por ciclo
    std::mt19937 rng(gap_seed);
    std::uniform_int_distribution<int> gap(0, 3);
    struct In { bool v; in_t re, im; };
    std::vector<In> seq;
    for (const auto& coil : ks) {
        for (int k = 0; k < p.n2(); ++k) {
            for (int r = 0; r < p.n1(); ++r) {
                const long i = static_cast<long>(k) * p.n1() + r;
                seq.push_back({true, coil[2 * i], coil[2 * i + 1]});
            }
            if (gap_seed != 0) {
                for (int g = gap(rng); g > 0; --g) seq.push_back({false, in_t(0), in_t(0)});
            }
        }
    }

    SimRun run;
    const long total  = static_cast<long>(N_COILS) * p.frame();
    const long budget = static_cast<long>(seq.size()) + p.latency() + 2 * p.frame() + 16;

    for (long c = 0; c < budget && static_cast<long>(run.out.size()) < total; ++c) {
        if (c < static_cast<long>(seq.size())) {
            dut.i_valid = fft2d::bit_t(seq[c].v);
            dut.i_re    = seq[c].re;
            dut.i_im    = seq[c].im;
        }
        else {
            dut.i_valid = fft2d::bit_t(false);
            dut.i_re    = in_t(0);
            dut.i_im    = in_t(0);
        }

        sim.cycle();

        if (dut.o_valid.value().to_bool()) {
            run.out.emplace_back(static_cast<double>(dut.o_re.value()),
                                 static_cast<double>(dut.o_im.value()));
            run.valid_cycle.push_back(c);
        }
        if (dut.o_last.value().to_bool()) {
            run.last_cycle.push_back(c);
        }
    }
    return run;
}

/*
 * Referencia de plomeria: dos modelos 1D sueltos, cada uno en su propia
 * simulacion. Filas de todas las bobinas por el nucleo 1; despues, por bobina,
 * las columnas (slot t1, k' natural) por el nucleo 2.
 */
std::vector<cd> run_1d_stream(const fft1d_r22sdf::Params& p1, const std::vector<std::pair<in_t, in_t>>& in)
{
    rtl::ClockDomain                 clk;
    fft1d_r22sdf::fft1d_r22sdf_model dut(p1);
    rtl::Simulator                   sim(clk);
    sim.add(dut);
    sim.init();

    std::vector<cd> out;
    const long n = static_cast<long>(in.size());
    for (long c = 0; c < n + p1.latency() + 8 && static_cast<long>(out.size()) < n; ++c) {
        dut.i_valid = fft1d_r22sdf::bit_t(c < n);
        dut.i_re    = c < n ? in[c].first  : in_t(0);
        dut.i_im    = c < n ? in[c].second : in_t(0);
        sim.cycle();
        if (dut.o_valid.value().to_bool()) {
            out.emplace_back(static_cast<double>(dut.o_re.value()),
                             static_cast<double>(dut.o_im.value()));
        }
    }
    return out;
}

std::vector<cd> composed_reference(const fft2d::Params& p, const std::vector<std::vector<in_t>>& ks)
{
    const int n1 = p.n1(), n2 = p.n2();
    const long t = p.frame();

    std::vector<std::pair<in_t, in_t>> rows;
    for (const auto& coil : ks) {
        for (long i = 0; i < t; ++i) rows.emplace_back(coil[2 * i], coil[2 * i + 1]);
    }
    const std::vector<cd> y1 = run_1d_stream(p.pass1, rows);     // [l][k'][slot]

    std::vector<std::pair<in_t, in_t>> cols;
    for (int l = 0; l < N_COILS; ++l) {
        for (int t1 = 0; t1 < n1; ++t1) {
            for (int k = 0; k < n2; ++k) {
                const cd v = y1[l * t + static_cast<long>(k) * n1 + t1];
                cols.emplace_back(in_t(v.real()), in_t(v.imag()));   // out_t -> in_t, exacto
            }
        }
    }
    return run_1d_stream(p.pass2, cols);                          // [l][t1][t2]
}

/*
 * IDFT 2D en double, separable y en el orden del hardware:
 *   ref[l][t1 * N2 + t2] = 2^-S sum_{k', r} K[k', r] e^{+j2pi(r x / N1 + k' p / N2)}
 * con x = br(t1), p = br(t2). Angulos con (indice * k) mod N para no perder
 * precision.
 */
std::vector<cd> idft2d_reference(const fft2d::Params& p, const std::vector<std::vector<in_t>>& ks)
{
    const int n1 = p.n1(), n2 = p.n2();
    const int b1 = ilog2(n1), b2 = ilog2(n2);
    const double gain = std::ldexp(1.0, -p.scale_shift());
    const double sgn  = p.pass1.inverse ? +1.0 : -1.0;

    std::vector<cd> ref;
    ref.reserve(static_cast<size_t>(N_COILS) * p.frame());
    for (const auto& coil : ks) {
        // pasada 1: h[k'][x]
        std::vector<cd> h(static_cast<size_t>(n2) * n1);
        for (int k = 0; k < n2; ++k) {
            for (int x = 0; x < n1; ++x) {
                cd acc = 0.0;
                for (int r = 0; r < n1; ++r) {
                    const long i = static_cast<long>(k) * n1 + r;
                    const cd   v(static_cast<double>(coil[2 * i]), static_cast<double>(coil[2 * i + 1]));
                    const double ang = sgn * 2.0 * PI * static_cast<double>((static_cast<long>(r) * x) % n1) / n1;
                    acc += v * cd(std::cos(ang), std::sin(ang));
                }
                h[static_cast<size_t>(k) * n1 + x] = acc;
            }
        }
        // pasada 2 y orden del hardware
        for (int t1 = 0; t1 < n1; ++t1) {
            const int x = bitrev(t1, b1);
            for (int t2 = 0; t2 < n2; ++t2) {
                const int pp = bitrev(t2, b2);
                cd acc = 0.0;
                for (int k = 0; k < n2; ++k) {
                    const double ang = sgn * 2.0 * PI * static_cast<double>((static_cast<long>(k) * pp) % n2) / n2;
                    acc += h[static_cast<size_t>(k) * n1 + x] * cd(std::cos(ang), std::sin(ang));
                }
                ref.push_back(acc * gain);
            }
        }
    }
    return ref;
}

/*
 * Las dos corridas del modelo 2D (entrada continua y con huecos) se hacen una
 * sola vez y las comparten los tests. Se hacen la primera vez que un test las
 * pide, DENTRO del test: si el modelo tira (overrun, posicion pisada), la
 * excepcion queda como falla de ese test con su mensaje. En SetUpTestSuite
 * gtest marcaria los tests como skipped.
 */
struct Data {
    fft2d::Params                  p;
    std::vector<std::vector<in_t>> ks;
    SimRun                         cont;
    SimRun                         gaps;
};

const Data& data()
{
    static std::unique_ptr<Data> d;
    static std::string           error;

    if (!d && error.empty()) {
        try {
            auto x  = std::make_unique<Data>();
            x->ks   = make_kspace(x->p);
            x->cont = run_2d(x->p, x->ks, 0);
            x->gaps = run_2d(x->p, x->ks, 12345);
            d = std::move(x);
        }
        catch (const std::exception& e) {
            error = e.what();
        }
    }
    if (!d) {
        throw std::runtime_error(error);
    }
    return *d;
}

TEST(Fft2d, OutputCount)
{
    const Data&  D     = data();
    const size_t total = static_cast<size_t>(N_COILS) * D.p.frame();
    ASSERT_EQ(D.cont.out.size(), total);
    ASSERT_EQ(D.gaps.out.size(), total);
}

TEST(Fft2d, MatchesCoreComposition)
{
    const Data&           D   = data();
    const std::vector<cd> ref = composed_reference(D.p, D.ks);
    ASSERT_EQ(ref.size(), D.cont.out.size());

    long bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != D.cont.out[i]) {
            if (bad < 5) ADD_FAILURE() << "muestra " << i << ": 2D " << D.cont.out[i] << " vs 1D+1D " << ref[i];
            ++bad;
        }
    }
    EXPECT_EQ(bad, 0) << "muestras distintas de la composicion de los nucleos";
}

TEST(Fft2d, MatchesDirect2DIdft)
{
    const Data&           D   = data();
    const std::vector<cd> ref = idft2d_reference(D.p, D.ks);
    ASSERT_EQ(ref.size(), D.cont.out.size());

    double e2 = 0.0, s2 = 0.0, emax = 0.0, smax = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::abs(D.cont.out[i] - ref[i]);
        e2 += e * e;
        s2 += std::norm(ref[i]);
        emax = std::max(emax, e);
        smax = std::max(smax, std::abs(ref[i]));
    }
    const double sqnr = 10.0 * std::log10(s2 / std::max(e2, 1e-300));
    RecordProperty("sqnr_db", std::to_string(sqnr));
    std::printf("[ fft2d    ] S = %d, SQNR contra la IDFT 2D = %.2f dB, error max = %.3g (pico %.3g)\n",
                D.p.scale_shift(), sqnr, emax, smax);

#ifdef DOUBLE
    EXPECT_LE(emax, 1e-9 * smax) << "modo double: el error tiene que ser de redondeo";
#else
    EXPECT_GE(sqnr, static_cast<double>(FFT2D_MIN_SQNR_DB));
#endif
}

TEST(Fft2d, LatencyAndFraming)
{
    const Data&   D = data();
    const SimRun& r = D.cont;
    const long    t = D.p.frame();

    ASSERT_FALSE(r.valid_cycle.empty());
    EXPECT_EQ(r.valid_cycle.front(), D.p.latency()) << "primer o_valid";

    // con entrada continua, la salida es continua en todas las bobinas
    for (size_t i = 1; i < r.valid_cycle.size(); ++i) {
        ASSERT_EQ(r.valid_cycle[i], r.valid_cycle[i - 1] + 1) << "hueco en la salida, muestra " << i;
    }

    ASSERT_EQ(r.last_cycle.size(), static_cast<size_t>(N_COILS));
    for (int l = 0; l < N_COILS; ++l) {
        EXPECT_EQ(r.last_cycle[l], r.valid_cycle[(l + 1) * t - 1]) << "o_last de la bobina " << l;
    }
}

TEST(Fft2d, GapsBetweenRows)
{
    const Data&   D = data();
    const SimRun& g = D.gaps;
    const long    t = D.p.frame();

    ASSERT_EQ(g.out.size(), D.cont.out.size());
    EXPECT_TRUE(g.out == D.cont.out) << "los huecos a la entrada cambiaron algun valor";

    // dentro de cada bobina la salida sigue siendo continua
    for (int l = 0; l < N_COILS; ++l) {
        EXPECT_EQ(g.valid_cycle[(l + 1) * t - 1] - g.valid_cycle[l * t], t - 1)
            << "hueco dentro de la bobina " << l;
    }
    ASSERT_EQ(g.last_cycle.size(), static_cast<size_t>(N_COILS));
    for (int l = 0; l < N_COILS; ++l) {
        EXPECT_EQ(g.last_cycle[l], g.valid_cycle[(l + 1) * t - 1]) << "o_last de la bobina " << l;
    }
}

TEST(Fft2dElaboration, Guards)
{
    fft2d::Params p;
    p.pass1.n = 12;
    EXPECT_THROW(fft2d::fft2d_model m(p), std::invalid_argument);

    fft2d::Params q;
    q.pass2.n = 2;
    EXPECT_THROW(fft2d::fft2d_model m(q), std::invalid_argument);
}

} // namespace

// fft1d_r22sdf_gtest.cpp
//
// Tests unitarios del MODELO C++ del R2^2SDF (fft1d_r22sdf_model).
//
//   double FALLA             -> bug ESTRUCTURAL (formula, signo, indice, orden).
//   double PASA, fixed FALLA -> bug NUMERICO (formato, redondeo, saturacion).
//
// La referencia es la DFT directa en double con cota de error. El modelo emite
// en orden BIT-REVERSED (como el pipeline SDF), asi que run_dut() reordena a
// natural antes de comparar.
//
// ESCALADO SELECTIVO (SCALE_MASK). Con S etapas escaladas, la salida es
//
//     X[k] = 2^-S * sum_n x[n] W^(nk),    S = popcount(SCALE_MASK & (N-1))
//
// y la referencia divide por 2^S, no por N. Para que ningun estimulo sature,
// las amplitudes nominales se multiplican por INPUT_GAIN = 2^(S - log2 N): con
// eso la salida queda en la amplitud nominal sea cual sea la mascara (ver
// apply_headroom). Con la mascara completa S = log2 N e INPUT_GAIN = 1: los
// estimulos son los mismos de siempre.
//
// La referencia se calcula sobre la entrada YA cuantizada al formato del
// puerto (quantize_input): el test mide el error de la FFT, no el de
// representar el estimulo en NB_IN bits.
//
// NOTA DE ESTADO (ver fft1d_r22sdf.hpp): este gtest valida que el modelo
// computa la FFT radix-2^2 correcta. NO es una comparacion bit-exacta contra
// el RTL (eso es el vm, pendiente de cerrar el control del pipeline SDF).

#include "fft1d_r22sdf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

using fft1d_r22sdf::N;
using fft1d_r22sdf::LOG2N;
using fft1d_r22sdf::LATENCY;
using fft1d_r22sdf::SCALE_MASK;
using fft1d_r22sdf::in_t;
using fft1d_r22sdf::out_t;
using fft1d_r22sdf::fft1d_r22sdf_model;

constexpr int popcount_u32(std::uint32_t v)
{
    return v ? static_cast<int>(v & 1u) + popcount_u32(v >> 1) : 0;
}

// Etapas que escalan x1/2. Los bits de SCALE_MASK por encima de log2(N)-1 no
// corresponden a ninguna etapa y se ignoran, igual que en el modelo y el RTL.
constexpr int S = popcount_u32(SCALE_MASK & ((1u << LOG2N) - 1u));

/*
 * Ganancia de entrada para que nada sature.
 *
 * Un tono de amplitud A crece x2 en cada etapa sin escalar: despues de la etapa
 * i vale A * 2^(i - s_i), con s_i las etapas escaladas hasta ahi. Como i - s_i
 * nunca decrece, el maximo esta en la salida, A * 2^(log2N - S), y hace falta
 * A * 2^(log2N - S) < 1. Multiplicando la amplitud nominal por 2^(S - log2N),
 * la salida queda en la amplitud nominal (<= 0.65 en estos tests).
 *
 * Es potencia de 2: en double el escalado es exacto.
 */
const double INPUT_GAIN = std::ldexp(1.0, S - LOG2N);

#ifdef DOUBLE
constexpr double TOL = 1e-12;
#else
constexpr double TOL = 5e-3;
#endif

int bitrev(int i)
{
    int r = 0;
    for (int b = 0; b < LOG2N; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

std::vector<std::complex<double>> dft_golden(const std::vector<std::complex<double>>& x)
{
    const int n = static_cast<int>(x.size());
    std::vector<std::complex<double>> out(n);
    for (int k = 0; k < n; ++k) {
        std::complex<double> acc(0.0, 0.0);
        for (int i = 0; i < n; ++i) {
            /*
             * (k*i) mod n ANTES de pasar a angulo: W^(k*i) es periodica en n,
             * y asi el argumento de cos/sin queda en [0, 2 pi). Sin reducir,
             * k*i llega a ~1e6 para N=1024 y cos/sin pierden ~1e-13 de
             * precision, que pasaria a ser el error dominante del modo double.
             */
            const long long ki = (static_cast<long long>(k) * i) % n;
            const double angle = -2.0 * PI * static_cast<double>(ki)
                                 / static_cast<double>(n);
            acc += x[i] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[k] = acc * std::ldexp(1.0, -S);   // 2^-S (= 1/N con la mascara completa)
    }
    return out;
}

/*
 * Corre un frame por el DUT streaming y devuelve las N muestras de salida ya
 * reordenadas a orden NATURAL de frecuencia.
 */
std::vector<std::complex<double>> run_dut(const std::vector<std::complex<double>>& x)
{
    rtl::ClockDomain clk;
    fft1d_r22sdf_model dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    std::vector<std::complex<double>> bitrev_out;
    bitrev_out.reserve(N);

    /*
     * El pipeline esta SEGMENTADO: el primer o_valid llega en el ciclo LATENCY
     * (70 para N=64, 265 para N=256, 1036 para N=1024; con CMUL_PIPE=1, 72,
     * 268 y 1040). La ventana debe cubrir
     * LATENCY + N para capturar el frame completo; se agrega margen.
     */
    const int total_cycles = LATENCY + N + 16;

    for (int c = 0; c < total_cycles; ++c) {
        if (c < N) {
            dut.i_valid = true;
            dut.i_re    = in_t(x[c].real());
            dut.i_im    = in_t(x[c].imag());
        }
        else {
            dut.i_valid = false;
            dut.i_re    = in_t(0);
            dut.i_im    = in_t(0);
        }

        sim.cycle();

        if (dut.o_valid.value().to_bool() && static_cast<int>(bitrev_out.size()) < N) {
            bitrev_out.emplace_back(
                static_cast<double>(dut.o_re.value()),
                static_cast<double>(dut.o_im.value())
            );
        }
    }

    std::vector<std::complex<double>> out(bitrev_out.size());
    for (int j = 0; j < static_cast<int>(bitrev_out.size()); ++j) {
        out[bitrev(j)] = bitrev_out[j];
    }
    return out;
}

std::vector<std::complex<double>> gen_complex_sine(double amp, int k0)
{
    std::vector<std::complex<double>> x(N);
    for (int i = 0; i < N; ++i) {
        const double angle = 2.0 * PI * static_cast<double>(k0) * static_cast<double>(i)
                             / static_cast<double>(N);
        x[i] = amp * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    return x;
}

std::vector<std::complex<double>> gen_real_cosine(double amp, int k0)
{
    std::vector<std::complex<double>> x(N);
    for (int i = 0; i < N; ++i) {
        const double angle = 2.0 * PI * static_cast<double>(k0) * static_cast<double>(i)
                             / static_cast<double>(N);
        x[i] = std::complex<double>(amp * std::cos(angle), 0.0);
    }
    return x;
}

// Amplitud nominal -> amplitud de entrada sin saturacion (ver INPUT_GAIN).
std::vector<std::complex<double>> apply_headroom(const std::vector<std::complex<double>>& x)
{
    std::vector<std::complex<double>> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = INPUT_GAIN * x[i];
    }
    return y;
}

/*
 * Lo que efectivamente ve el DUT: cada componente cuantizada a in_t con la
 * misma conversion que run_dut(). En modo double in_t es double y esto es la
 * identidad.
 */
std::vector<std::complex<double>> quantize_input(const std::vector<std::complex<double>>& x)
{
    std::vector<std::complex<double>> q(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        q[i] = std::complex<double>(
            static_cast<double>(in_t(x[i].real())),
            static_cast<double>(in_t(x[i].imag())));
    }
    return q;
}

void check_against_dft(const std::vector<std::complex<double>>& x_nominal)
{
    const std::vector<std::complex<double>> x        = apply_headroom(x_nominal);
    const std::vector<std::complex<double>> got      = run_dut(x);
    const std::vector<std::complex<double>> expected = dft_golden(quantize_input(x));

    ASSERT_EQ(got.size(), expected.size());
    for (int k = 0; k < static_cast<int>(got.size()); ++k) {
        EXPECT_NEAR(got[k].real(), expected[k].real(), TOL) << "  re, bin k=" << k;
        EXPECT_NEAR(got[k].imag(), expected[k].imag(), TOL) << "  im, bin k=" << k;
    }
}

// ---------------------------------------------------------------------- tests

TEST(Fft1dR22Sdf, ComplexSineLowFreq)
{
    check_against_dft(gen_complex_sine(0.5, 1));
}

TEST(Fft1dR22Sdf, ComplexSineMidFreq)
{
    check_against_dft(gen_complex_sine(0.5, N / 4));
}

TEST(Fft1dR22Sdf, ComplexSineHighFreq)
{
    check_against_dft(gen_complex_sine(0.5, N - 3));
}

TEST(Fft1dR22Sdf, RealCosine)
{
    check_against_dft(gen_real_cosine(0.5, 5));
}

TEST(Fft1dR22Sdf, DcInput)
{
    std::vector<std::complex<double>> x(N, std::complex<double>(0.5, 0.0));
    check_against_dft(x);
}

TEST(Fft1dR22Sdf, ZeroInput)
{
    std::vector<std::complex<double>> x(N, std::complex<double>(0.0, 0.0));
    check_against_dft(x);
}

TEST(Fft1dR22Sdf, Multitone)
{
    std::vector<std::complex<double>> x(N, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < N; ++i) {
        const double a1 = 2.0 * PI * 3.0  * i / N;
        const double a2 = 2.0 * PI * 7.0  * i / N;
        const double a3 = 2.0 * PI * 12.0 * i / N;
        x[i] = 0.3 * std::complex<double>(std::cos(a1), std::sin(a1))
             + 0.2 * std::complex<double>(std::cos(a2), std::sin(a2))
             + 0.15 * std::complex<double>(std::cos(a3), std::sin(a3));
    }
    check_against_dft(x);
}

TEST(Fft1dR22Sdf, LatencyMatchesFormula)
{
    /*
     * El primer o_valid tiene que salir exactamente en LATENCY (contado en
     * ciclos desde la primera muestra de entrada, con la misma convencion que
     * run_dut). LATENCY depende de CMUL_PIPE: N + (3 + CMUL_PIPE)(log4 N - 1).
     * Ademas: N salidas validas seguidas y un solo o_last, en la ultima.
     *
     * El vm compara el RTL contra el modelo ciclo a ciclo, asi que este test y
     * el vm juntos atan la formula, el modelo y el RTL.
     */
    const auto x = apply_headroom(gen_complex_sine(0.5, 3));

    rtl::ClockDomain clk;
    fft1d_r22sdf_model dut;
    rtl::Simulator sim(clk);
    sim.add(dut);
    sim.init();

    int first_valid = -1;
    int last_valid  = -1;
    int n_valid     = 0;
    int last_at     = -1;
    int n_last      = 0;

    for (int c = 0; c < LATENCY + N + 16; ++c) {
        if (c < N) {
            dut.i_valid = true;
            dut.i_re    = in_t(x[c].real());
            dut.i_im    = in_t(x[c].imag());
        }
        else {
            dut.i_valid = false;
            dut.i_re    = in_t(0);
            dut.i_im    = in_t(0);
        }

        sim.cycle();

        if (dut.o_valid.value().to_bool()) {
            if (first_valid < 0) first_valid = c;
            last_valid = c;
            ++n_valid;
        }
        if (dut.o_last.value().to_bool()) {
            last_at = c;
            ++n_last;
        }
    }

    EXPECT_EQ(first_valid, static_cast<int>(LATENCY));
    EXPECT_EQ(n_valid, N);
    EXPECT_EQ(last_valid - first_valid + 1, N) << "  las salidas validas no son contiguas";
    EXPECT_EQ(n_last, 1);
    EXPECT_EQ(last_at, first_valid + N - 1);
}

TEST(Fft1dR22Sdf, TwoConsecutiveFrames)
{
    // El segundo frame debe dar el mismo resultado que uno aislado: verifica
    // que el estado interno se limpia correctamente entre frames.
    const auto x1 = apply_headroom(gen_complex_sine(0.5, 3));
    const auto expected = dft_golden(quantize_input(x1));

    rtl::ClockDomain clk;
    fft1d_r22sdf_model dut;
    rtl::Simulator sim(clk);
    sim.add(dut);
    sim.init();

    std::vector<std::complex<double>> frame2_bitrev;
    const int total = LATENCY + 2 * N + 16;
    int captured = 0;

    for (int c = 0; c < total; ++c) {
        const int local = c % N;
        if (c < 2 * N) {
            dut.i_valid = true;
            dut.i_re = in_t(x1[local].real());
            dut.i_im = in_t(x1[local].imag());
        } else {
            dut.i_valid = false;
            dut.i_re = in_t(0);
            dut.i_im = in_t(0);
        }
        sim.cycle();
        // capturar la salida del SEGUNDO frame (aparece durante el 3er)
        if (dut.o_valid.value().to_bool()) {
            ++captured;
            if (captured > N && static_cast<int>(frame2_bitrev.size()) < N) {
                frame2_bitrev.emplace_back(
                    static_cast<double>(dut.o_re.value()),
                    static_cast<double>(dut.o_im.value()));
            }
        }
    }

    ASSERT_EQ(static_cast<int>(frame2_bitrev.size()), N);
    std::vector<std::complex<double>> frame2(N);
    for (int j = 0; j < N; ++j) frame2[bitrev(j)] = frame2_bitrev[j];

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(frame2[k].real(), expected[k].real(), TOL) << "  re, bin k=" << k;
        EXPECT_NEAR(frame2[k].imag(), expected[k].imag(), TOL) << "  im, bin k=" << k;
    }
}

} // namespace
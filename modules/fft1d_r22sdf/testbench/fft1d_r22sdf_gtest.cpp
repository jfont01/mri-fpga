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
// NOTA DE ESTADO (ver fft1d_r22sdf.hpp): este gtest valida que el modelo
// computa la FFT radix-2^2 correcta. NO es una comparacion bit-exacta contra
// el RTL (eso es el vm, pendiente de cerrar el control del pipeline SDF).

#include "fft1d_r22sdf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

using fft1d_r22sdf::N;
using fft1d_r22sdf::LOG2N;
using fft1d_r22sdf::LATENCY;
using fft1d_r22sdf::in_t;
using fft1d_r22sdf::out_t;
using fft1d_r22sdf::fft1d_r22sdf_model;

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
            const double angle = -2.0 * PI * static_cast<double>(k) * static_cast<double>(i)
                                 / static_cast<double>(n);
            acc += x[i] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[k] = acc / static_cast<double>(n);
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
     * (70 para N=64, 265 para N=256, 1036 para N=1024). La ventana debe cubrir
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

void check_against_dft(const std::vector<std::complex<double>>& x)
{
    const std::vector<std::complex<double>> got      = run_dut(x);
    const std::vector<std::complex<double>> expected = dft_golden(x);

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

TEST(Fft1dR22Sdf, TwoConsecutiveFrames)
{
    // El segundo frame debe dar el mismo resultado que uno aislado: verifica
    // que el estado interno se limpia correctamente entre frames.
    const auto x1 = gen_complex_sine(0.5, 3);
    const auto expected = dft_golden(x1);

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
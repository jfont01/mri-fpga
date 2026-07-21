// fft1d_r2_tb.cpp
#include "fft1d_r2.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

using fft1d_r2::N;
using fft1d_r2::in_t;
using fft1d_r2::out_t;
using fft1d_r2::fft1d_r2_model;

/*
 * Tolerancia:
 *   - Modo DOUBLE : el modelo y la DFT calculan lo mismo, pero con distinto
 *                   orden de operaciones -> solo difieren por redondeo de
 *                   punto flotante (~1e-15). 1e-12 es holgado pero durisimo.
 *   - Modo fixed  : el error lo domina la cuantizacion de Q1.15 (~3e-5 por
 *                   muestra, amplificado por las LOG2N etapas). La tolerancia
 *                   es empirica: sirve para detectar bugs de logica, NO para
 *                   caracterizar el ruido de cuantizacion (eso lo hace el
 *                   barrido de resoluciones, que reporta SNR/RMSE).
 */
#ifdef DOUBLE
constexpr double TOL = 1e-12;
#else
constexpr double TOL = 5e-3;
#endif

/*
 * DFT directa de referencia, en double.
 * Escala por 1/N para que coincida con el modelo (que escala 1/2 por etapa).
 */
std::vector<std::complex<double>> dft_golden(const std::vector<std::complex<double>>& x)
{
    const int n = static_cast<int>(x.size());
    std::vector<std::complex<double>> out(n);

    for (int k = 0; k < n; ++k) {
        std::complex<double> acc(0.0, 0.0);

        for (int i = 0; i < n; ++i) {
            const double angle = -2.0 * PI * static_cast<double>(k) * static_cast<double>(i) / static_cast<double>(n);
            acc += x[i] * std::complex<double>(std::cos(angle), std::sin(angle));
        }

        out[k] = acc / static_cast<double>(n);
    }

    return out;
}

/*
 * Corre un frame completo por el DUT y devuelve las N muestras de salida.
 */
std::vector<std::complex<double>> run_dut(const std::vector<std::complex<double>>& x)
{
    rtl::ClockDomain clk;
    fft1d_r2_model   dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    std::vector<std::complex<double>> out;
    out.reserve(N);

    // LOADING: N ciclos, una muestra por ciclo
    for (int i = 0; i < N; ++i) {
        dut.i_valid = true;
        dut.i_re    = in_t(x[i].real());
        dut.i_im    = in_t(x[i].imag());
        sim.cycle();
    }

    dut.i_valid = false;

    // COMPUTE + OUTPUT: drenamos con margen suficiente
    //   COMPUTE = LOG2N * N/2 ciclos, OUTPUT = N ciclos
    const int drain_cycles = fft1d_r2::LOG2N * (N / 2) + N + 16;

    for (int c = 0; c < drain_cycles; ++c) {
        sim.cycle();

        if (dut.o_valid.value().to_bool()) {
            out.emplace_back(
                static_cast<double>(dut.o_re.value()),
                static_cast<double>(dut.o_im.value())
            );
        }
    }

    return out;
}

/*
 * Estimulo: senoidal compleja (exponencial) de frecuencia k0.
 * Su DFT es un unico bin distinto de cero -> pico limpio en k0.
 */
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

/*
 * Estimulo: senoidal real (coseno).
 * Su DFT tiene dos bines (k0 y N-k0), cada uno con la mitad de la energia.
 */
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

/*
 * Compara la salida del DUT contra la DFT de referencia, muestra por muestra.
 */
void check_against_dft(const std::vector<std::complex<double>>& x)
{
    const std::vector<std::complex<double>> got      = run_dut(x);
    const std::vector<std::complex<double>> expected = dft_golden(x);

    ASSERT_EQ(got.size(), static_cast<size_t>(N)) << "el DUT no emitio N muestras";

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(got[k].real(), expected[k].real(), TOL) << "parte real, bin k=" << k;
        EXPECT_NEAR(got[k].imag(), expected[k].imag(), TOL) << "parte imag, bin k=" << k;
    }
}

} // namespace

TEST(Fft1dR2, ComplexSineLowFreq)
{
    check_against_dft(gen_complex_sine(0.5, 1));
}

TEST(Fft1dR2, ComplexSineMidFreq)
{
    check_against_dft(gen_complex_sine(0.5, N / 4));
}

TEST(Fft1dR2, RealCosine)
{
    check_against_dft(gen_real_cosine(0.5, 5));
}

TEST(Fft1dR2, DcInput)
{
    // Constante: toda la energia en el bin 0.
    std::vector<std::complex<double>> x(N, std::complex<double>(0.5, 0.0));
    check_against_dft(x);
}

TEST(Fft1dR2, ZeroInput)
{
    std::vector<std::complex<double>> x(N, std::complex<double>(0.0, 0.0));
    check_against_dft(x);
}

/*
 * Verifica que el DUT vuelva a LOADING y pueda procesar otro frame seguido.
 */
TEST(Fft1dR2, TwoConsecutiveFrames)
{
    rtl::ClockDomain clk;
    fft1d_r2_model   dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    const std::vector<std::complex<double>> x1 = gen_complex_sine(0.5, 3);
    const std::vector<std::complex<double>> x2 = gen_real_cosine(0.5, 7);

    const std::vector<std::complex<double>> e1 = dft_golden(x1);
    const std::vector<std::complex<double>> e2 = dft_golden(x2);

    const int drain_cycles = fft1d_r2::LOG2N * (N / 2) + N + 16;

    std::vector<std::complex<double>> out;

    for (const std::vector<std::complex<double>>* x : {&x1, &x2}) {
        for (int i = 0; i < N; ++i) {
            dut.i_valid = true;
            dut.i_re    = in_t((*x)[i].real());
            dut.i_im    = in_t((*x)[i].imag());
            sim.cycle();
        }

        dut.i_valid = false;

        for (int c = 0; c < drain_cycles; ++c) {
            sim.cycle();

            if (dut.o_valid.value().to_bool()) {
                out.emplace_back(
                    static_cast<double>(dut.o_re.value()),
                    static_cast<double>(dut.o_im.value())
                );
            }
        }
    }

    ASSERT_EQ(out.size(), static_cast<size_t>(2 * N)) << "faltan muestras de algun frame";

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(out[k].real(), e1[k].real(), TOL) << "frame 1, real, k=" << k;
        EXPECT_NEAR(out[k].imag(), e1[k].imag(), TOL) << "frame 1, imag, k=" << k;
    }

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(out[N + k].real(), e2[k].real(), TOL) << "frame 2, real, k=" << k;
        EXPECT_NEAR(out[N + k].imag(), e2[k].imag(), TOL) << "frame 2, imag, k=" << k;
    }
}
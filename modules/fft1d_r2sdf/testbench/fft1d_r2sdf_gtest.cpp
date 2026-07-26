// fft1d_r2sdf_gtest.cpp
//
// Tests unitarios del MODELO C++ del R2SDF (fft1d_r2sdf_model), en los dos
// modos de compilacion (fixed / double), igual que fft1d_r2_gtest.cpp:
//
//   double FALLA             -> bug ESTRUCTURAL (formula, signo, indice, orden).
//   double PASA, fixed FALLA -> bug NUMERICO (formato, redondeo, saturacion).
//
// La referencia es la DFT directa en double con cota de error (NO comparacion
// bit a bit contra el iterativo: el R2SDF multiplica DESPUES de la mariposa y
// los redondeos caen en lugares distintos).
//
// PARTICULARIDADES DEL R2SDF vs el iterativo:
//   - Es STREAMING: se inyecta una muestra por ciclo sin pausa. No hay fase de
//     LOADING/COMPUTE/OUTPUT.
//   - LATENCIA = N ciclos (N-1 de la cadena + 1 del registro de salida). La
//     primera salida valida aparece en el ciclo N-esimo tras la primera entrada.
//   - ORDEN DE SALIDA = BIT-REVERSED. run_dut() reordena a natural antes de
//     comparar contra la DFT.

#include "fft1d_r2sdf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

using fft1d_r2sdf::N;
using fft1d_r2sdf::LOG2N;
using fft1d_r2sdf::in_t;
using fft1d_r2sdf::out_t;
using fft1d_r2sdf::fft1d_r2sdf_model;

/*
 * Tolerancia:
 *   - DOUBLE : el modelo y la DFT calculan lo mismo con distinto orden de
 *              operaciones -> difieren por redondeo de punto flotante (~1e-15).
 *              1e-12 es holgado pero durisimo.
 *   - fixed  : el error lo domina la cuantizacion Q1.15 acumulada por las
 *              log2(N) etapas. La tolerancia es empirica: detecta bugs de
 *              logica, NO caracteriza el ruido de cuantizacion (eso es el
 *              barrido de resoluciones, que reporta SNR/RMSE).
 */
#ifdef DOUBLE
constexpr double TOL = 1e-12;
#else
constexpr double TOL = 5e-3;
#endif

/*
 * Bit-reversal de log2(N) bits.
 */
int bitrev(int i)
{
    int r = 0;
    for (int b = 0; b < LOG2N; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

/*
 * DFT directa de referencia, en double, escalada 1/N para coincidir con el
 * modelo (que acumula x1/2 por etapa = 1/N total).
 */
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
 *
 * Semantica de muestreo (identica a los tb del proyecto): se fija la entrada,
 * se avanza sim.cycle(), y se lee la salida DESPUES del flanco. Con la salida
 * registrada, o_valid/o_re/o_im son funcion del estado, asi que la observacion
 * no es ambigua.
 */
std::vector<std::complex<double>> run_dut(const std::vector<std::complex<double>>& x)
{
    rtl::ClockDomain clk;
    fft1d_r2sdf_model dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    std::vector<std::complex<double>> bitrev_out;
    bitrev_out.reserve(N);

    // Streaming: N muestras de entrada seguidas + drenado. La latencia es N,
    // asi que con 2N ciclos alcanza para capturar el frame completo. Margen +8.
    const int total_cycles = 2 * N + 8;

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

    // Reordenar bit-reversed -> natural.
    std::vector<std::complex<double>> out(bitrev_out.size());
    for (int j = 0; j < static_cast<int>(bitrev_out.size()); ++j) {
        out[bitrev(j)] = bitrev_out[j];
    }
    return out;
}

/*
 * Estimulo: senoidal compleja (exponencial) de frecuencia k0.
 * Su DFT es un unico bin no nulo en k0.
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
 * Estimulo: coseno real. DFT con dos bines (k0 y N-k0), mitad de energia cada uno.
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
 * Compara la salida del DUT (ya en orden natural) contra la DFT de referencia.
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

TEST(Fft1dR2sdf, ComplexSineLowFreq)
{
    check_against_dft(gen_complex_sine(0.5, 1));
}

TEST(Fft1dR2sdf, ComplexSineMidFreq)
{
    check_against_dft(gen_complex_sine(0.5, N / 4));
}

TEST(Fft1dR2sdf, ComplexSineArbFreq)
{
    check_against_dft(gen_complex_sine(0.5, 3));
}

TEST(Fft1dR2sdf, RealCosine)
{
    check_against_dft(gen_real_cosine(0.5, 5));
}

TEST(Fft1dR2sdf, DcInput)
{
    // Constante: toda la energia en el bin 0.
    std::vector<std::complex<double>> x(N, std::complex<double>(0.5, 0.0));
    check_against_dft(x);
}

TEST(Fft1dR2sdf, ZeroInput)
{
    std::vector<std::complex<double>> x(N, std::complex<double>(0.0, 0.0));
    check_against_dft(x);
}

/*
 * Streaming continuo: dos frames back-to-back SIN idle entre medio. El R2SDF,
 * a diferencia del iterativo, admite un frame nuevo mientras drena el anterior,
 * porque no tiene fase de carga/descarga: es un pipeline continuo. Verifica que
 * las salidas de ambos frames sean correctas y no se contaminen.
 */
TEST(Fft1dR2sdf, TwoStreamedFrames)
{
    rtl::ClockDomain clk;
    fft1d_r2sdf_model dut;
    rtl::Simulator   sim(clk);

    sim.add(dut);
    sim.init();

    const std::vector<std::complex<double>> x1 = gen_complex_sine(0.5, 3);
    const std::vector<std::complex<double>> x2 = gen_real_cosine(0.5, 7);

    const std::vector<std::complex<double>> e1 = dft_golden(x1);
    const std::vector<std::complex<double>> e2 = dft_golden(x2);

    // Inyectamos 2N muestras seguidas (frame1 || frame2), sin huecos, y drenamos.
    std::vector<std::complex<double>> raw;
    const int total = 3 * N + 8;

    for (int c = 0; c < total; ++c) {
        if (c < N) {
            dut.i_valid = true;
            dut.i_re = in_t(x1[c].real());
            dut.i_im = in_t(x1[c].imag());
        }
        else if (c < 2 * N) {
            dut.i_valid = true;
            dut.i_re = in_t(x2[c - N].real());
            dut.i_im = in_t(x2[c - N].imag());
        }
        else {
            dut.i_valid = false;
            dut.i_re = in_t(0);
            dut.i_im = in_t(0);
        }

        sim.cycle();

        if (dut.o_valid.value().to_bool()) {
            raw.emplace_back(
                static_cast<double>(dut.o_re.value()),
                static_cast<double>(dut.o_im.value())
            );
        }
    }

    ASSERT_GE(raw.size(), static_cast<size_t>(2 * N)) << "faltan muestras de algun frame";

    // Reordenar cada bloque de N muestras (bit-reversed -> natural) y comparar.
    auto reorder = [&](int base) {
        std::vector<std::complex<double>> nat(N);
        for (int j = 0; j < N; ++j) nat[bitrev(j)] = raw[base + j];
        return nat;
    };

    const std::vector<std::complex<double>> got1 = reorder(0);
    const std::vector<std::complex<double>> got2 = reorder(N);

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(got1[k].real(), e1[k].real(), TOL) << "frame 1, real, k=" << k;
        EXPECT_NEAR(got1[k].imag(), e1[k].imag(), TOL) << "frame 1, imag, k=" << k;
        EXPECT_NEAR(got2[k].real(), e2[k].real(), TOL) << "frame 2, real, k=" << k;
        EXPECT_NEAR(got2[k].imag(), e2[k].imag(), TOL) << "frame 2, imag, k=" << k;
    }
}
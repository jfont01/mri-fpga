// cmul_karatsuba_gtest.cpp
//
// Verificacion del MODELO de punto fijo contra una referencia independiente en
// double (std::complex). Esta capa es complementaria al pipeline vm:
//
//   vm     : RTL  ==  modelo C++      (bit a bit)
//   gtest  : modelo C++  ~=  matematica exacta   (con cota de error demostrada)
//
// Sin esta capa, un modelo auto-consistente pero con la convencion equivocada
// (p.ej. AP_TRN/AP_WRAP en vez de AP_RND/AP_SAT) pasaria el vm en verde.
//
// ---------------------------------------------------------------------------
// Por que la tolerancia NO es empirica
// ---------------------------------------------------------------------------
//
// 1) Los estimulos se cargan por valor CRUDO, asi que la entrada del modelo es
//    exactamente representable: no hay error de cuantizacion de entrada.
//
// 2) La referencia se calcula con esos MISMOS valores ya cuantizados. Para
//    NB_IN <= 26 los productos y sumas son enteros de menos de 53 bits, o sea
//    EXACTOS en double (la mantissa alcanza), y el escalado por 2^-NBF es una
//    potencia de dos, tambien exacta. La referencia no tiene error.
//
// 3) El datapath del modelo mantiene todos los intermedios exactos y cuantiza
//    UNA sola vez, al final, con AP_RND (round-half-up).
//
// => el unico error posible es ese redondeo final, acotado por medio LSB:
//
//        |error| <= 2^-(NBF_OUT + 1)
//
// Esa es la cota que se verifica (HALF_LSB), no un numero elegido a ojo. Si el
// test falla, o hay un bug, o alguien cambio la convencion de redondeo.
//
// Cuando el resultado exacto se sale del rango de out_t, AP_SAT satura; para
// que la comparacion siga siendo valida, la referencia se satura igual
// (clamp_to_out_range) antes de comparar.
//
// ---------------------------------------------------------------------------
// Como apuntarlo a otro modulo
// ---------------------------------------------------------------------------
// Cambiar el #include y el alias 'dut' de abajo (p.ej. cmul_karatsuba, o el
// cmul de 4 multiplicadores). El resto del archivo no depende del nombre.

#include "cmul_karatsuba.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace dut = cmul_karatsuba;
using model_t = dut::cmul_karatsuba_model;

namespace {

/*
 * --------------------------------------------------------------------------
 * Constantes derivadas del formato
 * --------------------------------------------------------------------------
 */

constexpr int NB_IN   = dut::NB_IN;
constexpr int NBF_IN  = dut::NBF_IN;
constexpr int NB_OUT  = dut::NB_OUT;
constexpr int NBF_OUT = dut::NBF_OUT;

// LSB del formato de salida y la cota de error del redondeo
const double LSB_OUT  = std::ldexp(1.0, -NBF_OUT);        // 2^-NBF_OUT
const double HALF_LSB = std::ldexp(1.0, -(NBF_OUT + 1));  // 2^-(NBF_OUT+1)

/*
 * Tolerancia segun el modo de compilacion:
 *
 *   punto fijo : medio LSB. Cota DEMOSTRADA (ver cabecera), no empirica.
 *   DOUBLE     : el modelo y la referencia hacen la misma matematica, pero en
 *                distinto orden de operaciones (Gauss recombina k1,k2,k3),
 *                asi que solo difieren por redondeo de punto flotante (~1e-16
 *                relativo). 1e-12 es holgado pero durisimo para cazar un bug
 *                de formula o de signo.
 */
#ifdef DOUBLE
const double TOL = 1e-12;
#else
const double TOL = HALF_LSB;
#endif

// rango representable de la entrada (raw)
constexpr int IN_RAW_MAX =  (1 << (NB_IN - 1)) - 1;
constexpr int IN_RAW_MIN = -(1 << (NB_IN - 1));

// rango representable de la salida (valor real)
const double OUT_MAX = std::ldexp(static_cast<double>((1 << (NB_OUT - 1)) - 1), -NBF_OUT);
const double OUT_MIN = std::ldexp(static_cast<double>(-(1 << (NB_OUT - 1))),    -NBF_OUT);

/*
 * --------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------
 */

/*
 * Construye un in_t desde su representacion cruda (entero con signo).
 * En modo DOUBLE no hay bits, asi que se escala el raw por 2^-NBF_IN: de esa
 * forma AMBOS modos ven exactamente el mismo VALOR de entrada y los tests se
 * escriben una sola vez.
 */
dut::in_t in_from_raw(int raw)
{
#ifdef DOUBLE
    return std::ldexp(static_cast<double>(raw), -NBF_IN);
#else
    ap_int<NB_IN> r = raw;
    dut::in_t v;
    v.range(NB_IN - 1, 0) = r.range(NB_IN - 1, 0);
    return v;
#endif
}

// Convierte un valor real al raw mas cercano del formato de entrada.
int raw_from_double(double x)
{
    const double scaled = std::round(std::ldexp(x, NBF_IN));

    if (scaled > IN_RAW_MAX) {
        return IN_RAW_MAX;
    }
    if (scaled < IN_RAW_MIN) {
        return IN_RAW_MIN;
    }
    return static_cast<int>(scaled);
}

/*
 * Satura un valor real al rango de out_t (lo mismo que hace AP_SAT).
 * En modo DOUBLE no hay saturacion: la funcion es la identidad.
 */
double clamp_to_out_range(double x)
{
#ifdef DOUBLE
    return x;
#else
    if (x > OUT_MAX) {
        return OUT_MAX;
    }
    if (x < OUT_MIN) {
        return OUT_MIN;
    }
    return x;
#endif
}

struct dut_result {
    std::complex<double> got;       // salida del modelo
    std::complex<double> exact;     // referencia en double, sin saturar
    std::complex<double> expected;  // referencia saturada al rango de salida
};

/*
 * Corre un producto por el modelo y arma la referencia con std::complex,
 * usando los MISMOS valores cuantizados que ve el modelo.
 */
dut_result run_one(int a_raw, int b_raw, int c_raw, int d_raw)
{
    rtl::ClockDomain clk;
    model_t          m;
    rtl::Simulator   sim(clk);

    sim.add(m);
    sim.init();

    const dut::in_t a = in_from_raw(a_raw);
    const dut::in_t b = in_from_raw(b_raw);
    const dut::in_t c = in_from_raw(c_raw);
    const dut::in_t d = in_from_raw(d_raw);

    m.i_1_re = a;
    m.i_1_im = b;
    m.i_2_re = c;
    m.i_2_im = d;

    sim.cycle();

    dut_result r;

    r.got = std::complex<double>(
        static_cast<double>(m.o_re.value()),
        static_cast<double>(m.o_im.value())
    );

    // referencia: producto complejo en double sobre las entradas cuantizadas
    const std::complex<double> z1(static_cast<double>(a), static_cast<double>(b));
    const std::complex<double> z2(static_cast<double>(c), static_cast<double>(d));

    r.exact = z1 * z2;

    r.expected = std::complex<double>(
        clamp_to_out_range(r.exact.real()),
        clamp_to_out_range(r.exact.imag())
    );

    return r;
}

// Chequeo estandar: la salida cae dentro de medio LSB de la referencia saturada.
void expect_within_half_lsb(int a, int b, int c, int d)
{
    const dut_result r = run_one(a, b, c, d);

    EXPECT_NEAR(r.got.real(), r.expected.real(), TOL)
        << "parte real | raw a=" << a << " b=" << b << " c=" << c << " d=" << d
        << " | exacto=" << r.exact.real();

    EXPECT_NEAR(r.got.imag(), r.expected.imag(), TOL)
        << "parte imag | raw a=" << a << " b=" << b << " c=" << c << " d=" << d
        << " | exacto=" << r.exact.imag();
}

} // namespace

/*
 * ==========================================================================
 * 1. Casos triviales y de identidad
 * ==========================================================================
 */

TEST(CmulModel, ZeroTimesAnything)
{
    expect_within_half_lsb(0, 0, 0, 0);
    expect_within_half_lsb(0, 0, IN_RAW_MAX, IN_RAW_MIN);
    expect_within_half_lsb(IN_RAW_MAX, IN_RAW_MIN, 0, 0);
}

TEST(CmulModel, MultiplyByOneHalf)
{
    // 0.5 siempre es representable mientras NBF_IN >= 1
    const int half = raw_from_double(0.5);

    // (0.25 + 0.25j) * 0.5 = 0.125 + 0.125j
    const int q = raw_from_double(0.25);

    const dut_result r = run_one(q, q, half, 0);

    EXPECT_NEAR(r.got.real(), 0.125, TOL);
    EXPECT_NEAR(r.got.imag(), 0.125, TOL);
}

TEST(CmulModel, JTimesJIsMinusOne)
{
    // (0 + 0.5j) * (0 + 0.5j) = -0.25
    const int half = raw_from_double(0.5);

    const dut_result r = run_one(0, half, 0, half);

    EXPECT_NEAR(r.got.real(), -0.25, TOL);
    EXPECT_NEAR(r.got.imag(),  0.0,  TOL);
}

TEST(CmulModel, ConjugateProductIsRealAndNonNegative)
{
    // z * conj(z) = |z|^2  -> parte imaginaria exactamente cero
    const int re = raw_from_double(0.6);
    const int im = raw_from_double(-0.3);

    const dut_result r = run_one(re, im, re, -im);

    /*
     * OJO: 0.6 y 0.3 NO son representables en punto fijo, asi que la
     * referencia NO puede escribirse como 0.6*0.6 + 0.3*0.3: eso mezclaria el
     * error de cuantizacion de ENTRADA con el de salida y la cota de medio LSB
     * dejaria de valer. Hay que comparar contra r.expected, que se calcula con
     * las entradas ya cuantizadas (mismo criterio que el resto del archivo).
     */
    EXPECT_NEAR(r.got.real(), r.expected.real(), TOL);
    EXPECT_DOUBLE_EQ(r.got.imag(), 0.0) << "z*conj(z) debe ser puramente real";
    EXPECT_GE(r.got.real(), 0.0);
}

/*
 * ==========================================================================
 * 2. Resultados exactamente representables -> error CERO
 * ==========================================================================
 */

TEST(CmulModel, ExactlyRepresentableProductsHaveZeroError)
{
    /*
     * Si ambos operandos son potencias de dos chicas, el producto cae justo en
     * la grilla de salida y el redondeo no tiene nada que hacer: el error debe
     * ser exactamente 0, no "casi 0".
     */
    for (int e1 = 1; e1 <= 4; ++e1) {
        for (int e2 = 1; e2 <= 4; ++e2) {
            if (e1 + e2 > NBF_OUT) {
                continue;
            }

            const double x = std::ldexp(1.0, -e1);   // 2^-e1
            const double y = std::ldexp(1.0, -e2);   // 2^-e2

            const dut_result r = run_one(raw_from_double(x), 0,
                                         raw_from_double(y), 0);

            EXPECT_DOUBLE_EQ(r.got.real(), x * y)
                << "2^-" << e1 << " * 2^-" << e2 << " deberia ser exacto";
            EXPECT_DOUBLE_EQ(r.got.imag(), 0.0);
        }
    }
}

/*
 * ==========================================================================
 * 3. Bordes del formato de entrada
 * ==========================================================================
 */

TEST(CmulModel, InputExtremes)
{
    const int vals[] = {0, 1, -1, IN_RAW_MAX, IN_RAW_MIN, IN_RAW_MAX - 1, IN_RAW_MIN + 1};

    for (int a : vals) {
        for (int b : vals) {
            for (int c : vals) {
                for (int d : vals) {
                    expect_within_half_lsb(a, b, c, d);
                }
            }
        }
    }
}

/*
 * ==========================================================================
 * 4. Barrido aleatorio sin saturacion (la cota mas estricta)
 * ==========================================================================
 */

TEST(CmulModel, RandomWithoutSaturationStaysWithinHalfLsb)
{
    /*
     * Se limitan las entradas para que |ac-bd| y |ad+bc| entren siempre en el
     * rango de salida. Asi AP_SAT nunca actua y el unico error posible es el
     * redondeo final.
     */
    const double bound = std::sqrt(OUT_MAX / 2.0) * 0.98;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-bound, bound);

    double max_err = 0.0;

    for (int i = 0; i < 3000; ++i) {
        const int a = raw_from_double(dist(rng));
        const int b = raw_from_double(dist(rng));
        const int c = raw_from_double(dist(rng));
        const int d = raw_from_double(dist(rng));

        const dut_result r = run_one(a, b, c, d);

        ASSERT_LE(std::abs(r.exact.real()), OUT_MAX) << "el caso saturo: revisar 'bound'";
        ASSERT_LE(std::abs(r.exact.imag()), OUT_MAX) << "el caso saturo: revisar 'bound'";

        EXPECT_NEAR(r.got.real(), r.exact.real(), TOL);
        EXPECT_NEAR(r.got.imag(), r.exact.imag(), TOL);

        max_err = std::max({max_err,
                            std::abs(r.got.real() - r.exact.real()),
                            std::abs(r.got.imag() - r.exact.imag())});
    }

    // el peor error observado no puede superar medio LSB
    EXPECT_LE(max_err, TOL);

    std::cout << "[          ] max |error| = " << max_err
              << "  (medio LSB = " << HALF_LSB << ")\n";
}

/*
 * ==========================================================================
 * 5. Barrido aleatorio de rango completo (con saturacion)
 * ==========================================================================
 */

TEST(CmulModel, RandomFullRangeMatchesSaturatedReference)
{
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> dist(IN_RAW_MIN, IN_RAW_MAX);

    int saturated = 0;

    for (int i = 0; i < 3000; ++i) {
        const int a = dist(rng);
        const int b = dist(rng);
        const int c = dist(rng);
        const int d = dist(rng);

        const dut_result r = run_one(a, b, c, d);

        if (r.exact != r.expected) {
            ++saturated;
        }

        EXPECT_NEAR(r.got.real(), r.expected.real(), TOL)
            << "raw a=" << a << " b=" << b << " c=" << c << " d=" << d;
        EXPECT_NEAR(r.got.imag(), r.expected.imag(), TOL)
            << "raw a=" << a << " b=" << b << " c=" << c << " d=" << d;
    }

#ifndef DOUBLE
    // el barrido tiene que ejercitar saturacion, si no no prueba AP_SAT
    EXPECT_GT(saturated, 0) << "ningun caso saturo: el test no cubre AP_SAT";
#endif

    std::cout << "[          ] casos que saturaron: " << saturated << " / 3000\n";
}

/*
 * ==========================================================================
 * 6. Saturacion: satura, no envuelve
 * ==========================================================================
 */

TEST(CmulModel, OverflowSaturatesInsteadOfWrapping)
{
#ifdef DOUBLE
    GTEST_SKIP() << "modo DOUBLE: no hay saturacion (el rango es el de double)";
#else
    /*
     * Este es el test que distingue AP_SAT de AP_WRAP. Con AP_WRAP el
     * resultado cambiaria de signo; con AP_SAT queda pegado al extremo.
     *
     * OJO con la eleccion de estimulos: tienen que desbordar en CUALQUIER
     * formato, no solo en el default. Por ejemplo max*max desborda en Q2.14
     * (0.99994*... -> 3.99976 > 1.99994) pero NO en Q1.15
     * (0.99997^2 = 0.99994 < 0.99997). En cambio min*min = (-1)^2 = 1.0
     * siempre queda 1 LSB por encima de OUT_MAX, en todo Qn.m.
     *
     * Los EXPECT_GT/LT sobre 'exact' son la guarda: si algun dia el estimulo
     * deja de desbordar, el test avisa en vez de pasar sin probar nada.
     */

    // desborde positivo: min * min = +|min|^2  >  OUT_MAX
    const dut_result pos = run_one(IN_RAW_MIN, 0, IN_RAW_MIN, 0);

    EXPECT_GT(pos.exact.real(), OUT_MAX) << "el estimulo deberia desbordar por arriba";
    EXPECT_DOUBLE_EQ(pos.got.real(), OUT_MAX) << "deberia saturar al maximo";

    // desborde negativo: min*max - max*max  <  OUT_MIN
    const dut_result neg = run_one(IN_RAW_MIN, IN_RAW_MAX, IN_RAW_MAX, IN_RAW_MAX);

    EXPECT_LT(neg.exact.real(), OUT_MIN) << "el estimulo deberia desbordar por abajo";
    EXPECT_DOUBLE_EQ(neg.got.real(), OUT_MIN) << "deberia saturar al minimo";
#endif
}

/*
 * ==========================================================================
 * 7. Estadistica del error de cuantizacion
 * ==========================================================================
 */

TEST(CmulModel, QuantizationErrorStatistics)
{
    const double bound = std::sqrt(OUT_MAX / 2.0) * 0.98;

    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> dist(-bound, bound);

    std::vector<double> errs;
    errs.reserve(4000);

    for (int i = 0; i < 2000; ++i) {
        const int a = raw_from_double(dist(rng));
        const int b = raw_from_double(dist(rng));
        const int c = raw_from_double(dist(rng));
        const int d = raw_from_double(dist(rng));

        const dut_result r = run_one(a, b, c, d);

        errs.push_back(r.got.real() - r.exact.real());
        errs.push_back(r.got.imag() - r.exact.imag());
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    double worst = 0.0;

    for (double e : errs) {
        sum    += e;
        sum_sq += e * e;
        worst   = std::max(worst, std::abs(e));
    }

    const double mean = sum / static_cast<double>(errs.size());
    const double rms  = std::sqrt(sum_sq / static_cast<double>(errs.size()));

#ifdef DOUBLE
    /*
     * En DOUBLE no hay cuantizacion: el error debe ser ruido de punto
     * flotante, ordenes de magnitud por debajo de un LSB del formato fijo.
     */
    EXPECT_LE(worst, TOL) << "en modo DOUBLE el error deberia ser ~1e-16";

    std::cout << "[          ] error (DOUBLE): media=" << mean
              << ", rms=" << rms
              << ", max=" << worst << "\n";
#else
    /*
     * Para redondeo (no truncamiento) el error es aprox uniforme en
     * [-LSB/2, +LSB/2]: media ~0 y RMS ~ LSB/sqrt(12) = 0.2887*LSB.
     * Si el modelo truncara, la media se iria a -LSB/2 y este test lo cazaria.
     */
    EXPECT_LE(worst, HALF_LSB);
    EXPECT_NEAR(mean, 0.0, 0.05 * LSB_OUT) << "media muy lejos de 0: parece truncamiento, no redondeo";
    EXPECT_NEAR(rms, LSB_OUT / std::sqrt(12.0), 0.10 * LSB_OUT);

    std::cout << "[          ] error: media=" << mean / LSB_OUT << " LSB"
              << ", rms=" << rms / LSB_OUT << " LSB"
              << ", max=" << worst / LSB_OUT << " LSB\n";
#endif
}
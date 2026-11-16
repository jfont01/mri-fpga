// ps_sense_gtest.cpp -- ps_sense contra model/unfold_ref.py, bit a bit.
//
// Los vectores los escribe model/gen_ps_sense_vectors.py (ver su docstring para
// el formato). La carpeta del caso sale, en este orden, de:
//   - la variable de entorno PS_SENSE_VEC (la usa el Makefile del modulo para
//     correr el mismo binario sobre todos los casos, en la PC y en qemu);
//   - el define PS_SENSE_VEC_DIR (clave VEC_DIR del JSON; run_gtest corre el
//     binario desde la carpeta del modulo).
//
// Los vectores se cargan una sola vez, adentro de los tests (no en
// SetUpTestSuite: una excepcion ahi deja los tests en skipped, ver 08_trampas).
#include "ps_sense.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace ps_sense;

#define PS_SENSE_STR2(x) #x
#define PS_SENSE_STR(x) PS_SENSE_STR2(x)

std::uint64_t bits(double v)
{
    std::uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

std::string hexf(double v)
{
    char s[64];
    std::snprintf(s, sizeof s, "%a", v);
    return s;
}

struct Vec {
    bool ok = false;
    std::string err;
    std::string dir;
    dims dm;
    config cfg;
    int k_iters = 0;
    // resultados esperados
    int E = 0;
    double max_abs = 0.0, d_min = 0.0, d_max = 0.0;
    std::uint32_t n_invalid = 0, n_sat = 0;
    std::size_t words = 0;
    // datos
    std::vector<double> smaps, g, u, d;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint32_t> buf;
};

std::string vec_dir()
{
    const char* e = std::getenv("PS_SENSE_VEC");
    if (e != nullptr && *e != '\0') {
        return e;
    }
#ifdef PS_SENSE_VEC_DIR
    return PS_SENSE_STR(PS_SENSE_VEC_DIR);
#else
    return "";
#endif
}

template <class T>
bool read_bin(const std::string& path, std::vector<T>& v, std::size_t n, std::string& err)
{
    v.assign(n, T {});
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "no se pudo abrir " + path;
        return false;
    }
    const std::size_t got = std::fread(v.data(), sizeof(T), n, f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != n || extra) {
        err = path + ": tamano distinto del esperado (" + std::to_string(n) + " elementos)";
        return false;
    }
    return true;
}

Vec load()
{
    Vec v;
    v.dir = vec_dir();
    if (v.dir.empty()) {
        v.err = "sin carpeta de vectores: definir PS_SENSE_VEC o compilar con -DPS_SENSE_VEC_DIR";
        return v;
    }
    FILE* f = std::fopen((v.dir + "/case.txt").c_str(), "r");
    if (f == nullptr) {
        v.err = "no se pudo abrir " + v.dir + "/case.txt (generar con model/gen_ps_sense_vectors.py)";
        return v;
    }
    char key[64], val[128];
    int found = 0;
    while (std::fscanf(f, "%63s %127s", key, val) == 2) {
        const std::string k = key;
        const double x = std::strtod(val, nullptr);  // enteros y hex floats, exacto
        const long i = std::strtol(val, nullptr, 10);
        if (k == "n_ro") { v.dm.n_ro = static_cast<int>(i); ++found; }
        else if (k == "n_pe") { v.dm.n_pe = static_cast<int>(i); ++found; }
        else if (k == "L") { v.dm.L = static_cast<int>(i); ++found; }
        else if (k == "af") { v.dm.af = static_cast<int>(i); ++found; }
        else if (k == "lam") { v.cfg.lam = x; ++found; }
        else if (k == "k_iters") { v.k_iters = static_cast<int>(i); ++found; }
        else if (k == "k_cordic") { v.cfg.k_cordic = x; ++found; }
        else if (k == "nb") { v.cfg.nb = static_cast<int>(i); ++found; }
        else if (k == "layout") { v.cfg.lay = static_cast<layout>(i); ++found; }
        else if (k == "E") { v.E = static_cast<int>(i); ++found; }
        else if (k == "max_abs") { v.max_abs = x; ++found; }
        else if (k == "d_min") { v.d_min = x; ++found; }
        else if (k == "d_max") { v.d_max = x; ++found; }
        else if (k == "n_invalid") { v.n_invalid = static_cast<std::uint32_t>(i); ++found; }
        else if (k == "n_sat") { v.n_sat = static_cast<std::uint32_t>(i); ++found; }
        else if (k == "words") { v.words = static_cast<std::size_t>(i); ++found; }
    }
    std::fclose(f);
    if (found != 16) {
        v.err = v.dir + "/case.txt incompleto";
        return v;
    }
    if (check_dims(v.dm) != status::ok) {
        v.err = v.dir + ": dimensiones invalidas para ps_sense";
        return v;
    }
    const std::size_t T = v.dm.positions();
    const std::size_t L = static_cast<std::size_t>(v.dm.L);
    const std::size_t af = static_cast<std::size_t>(v.dm.af);
    const std::size_t n_maps = 2 * L * static_cast<std::size_t>(v.dm.n_ro) * static_cast<std::size_t>(v.dm.n_pe);
    if (!read_bin(v.dir + "/smaps.bin", v.smaps, n_maps, v.err) ||
        !read_bin(v.dir + "/g.bin", v.g, L, v.err) ||
        !read_bin(v.dir + "/u_ref.bin", v.u, 2 * T * af * L, v.err) ||
        !read_bin(v.dir + "/d_ref.bin", v.d, T * af, v.err) ||
        !read_bin(v.dir + "/valid_ref.bin", v.valid, T, v.err) ||
        !read_bin(v.dir + "/buf_ref.bin", v.buf, v.dm.buffer_words(), v.err)) {
        return v;
    }
    v.ok = true;
    return v;
}

const Vec& vec()
{
    static const Vec v = load();
    return v;
}

#define REQUIRE_VEC()                                  \
    const Vec& V = vec();                              \
    ASSERT_TRUE(V.ok) << V.err

// ------------------------------------------------------------------ tests

// Sin esto nada de lo que sigue es exigible bit a bit (H29).
TEST(PsSense, CompiledWithoutFpContraction)
{
    EXPECT_TRUE(fp_contract_off()) << "compilar con -ffp-contract=off";
}

TEST(PsSense, CaseFileIsConsistent)
{
    REQUIRE_VEC();
    EXPECT_EQ(V.words, V.dm.buffer_words());
    EXPECT_EQ(check_config(V.dm, V.cfg, V.g.data()), status::ok);
}

// Nucleo por posicion (gram, ldlh, solve_u): U, d y la validez, bit a bit.
TEST(PsSense, KernelMatchesReferenceBitwise)
{
    REQUIRE_VEC();
    const dims& dm = V.dm;
    const int L = dm.L, af = dm.af, n2 = dm.n2();
    double Sr[MAX_L * MAX_AF], Si[MAX_L * MAX_AF], Ur[MAX_AF * MAX_L], Ui[MAX_AF * MAX_L], d[MAX_AF];
    std::size_t bad_valid = 0, bad_d = 0, bad_u = 0, n_valid = 0;
    std::string first;
    for (int x = 0; x < dm.n_ro; ++x) {
        for (int p = 0; p < n2; ++p) {
            const std::size_t pos = static_cast<std::size_t>(x) * static_cast<std::size_t>(n2) + static_cast<std::size_t>(p);
            gather_s(V.smaps.data(), dm, x, p, Sr, Si);
            const bool ok = unfold_position(Sr, Si, L, af, V.cfg.lam, Ur, Ui, d);
            if (ok != (V.valid[pos] != 0)) {
                if (bad_valid++ == 0 && first.empty()) first = "validez en (" + std::to_string(x) + ", " + std::to_string(p) + ")";
                continue;
            }
            for (int j = 0; j < af; ++j) {
                const double r = V.d[pos * static_cast<std::size_t>(af) + static_cast<std::size_t>(j)];
                const bool same = std::isfinite(r) ? bits(d[j]) == bits(r) : !std::isfinite(d[j]);
                if (!same && bad_d++ == 0 && first.empty()) {
                    first = "d[" + std::to_string(j) + "] en (" + std::to_string(x) + ", " + std::to_string(p) + "): " + hexf(d[j]) + " != " + hexf(r);
                }
            }
            if (!ok) {
                continue;  // U con NaN: no entra al buffer
            }
            ++n_valid;
            for (int a = 0; a < af; ++a) {
                for (int l = 0; l < L; ++l) {
                    const std::size_t k = 2 * ((pos * static_cast<std::size_t>(af) + static_cast<std::size_t>(a)) * static_cast<std::size_t>(L) + static_cast<std::size_t>(l));
                    if (bits(Ur[a * L + l]) != bits(V.u[k]) || bits(Ui[a * L + l]) != bits(V.u[k + 1])) {
                        if (bad_u++ == 0 && first.empty()) {
                            first = "U[a=" + std::to_string(a) + "][l=" + std::to_string(l) + "] en (" + std::to_string(x) + ", " + std::to_string(p) + "): " +
                                    hexf(Ur[a * L + l]) + " " + hexf(Ui[a * L + l]) + " != " + hexf(V.u[k]) + " " + hexf(V.u[k + 1]);
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(bad_valid, 0u) << first;
    EXPECT_EQ(bad_d, 0u) << first;
    EXPECT_EQ(bad_u, 0u) << first;
    EXPECT_EQ(dm.positions() - n_valid, static_cast<std::size_t>(V.n_invalid));
}

// La sesion completa: E, el reporte y el buffer de U'', bit a bit.
TEST(PsSense, SessionMatchesReferenceBitwise)
{
    REQUIRE_VEC();
    std::vector<std::uint32_t> out(V.dm.buffer_words(), 0xDEADBEEFu);
    report rep;
    ASSERT_EQ(build_buffer(V.dm, V.smaps.data(), V.g.data(), V.cfg, out.data(), &rep), status::ok);
    EXPECT_EQ(rep.E, V.E);
    EXPECT_EQ(bits(rep.max_abs), bits(V.max_abs)) << hexf(rep.max_abs) << " != " << hexf(V.max_abs);
    EXPECT_EQ(bits(rep.d_min), bits(V.d_min)) << hexf(rep.d_min) << " != " << hexf(V.d_min);
    EXPECT_EQ(bits(rep.d_max), bits(V.d_max)) << hexf(rep.d_max) << " != " << hexf(V.d_max);
    EXPECT_EQ(rep.n_invalid, V.n_invalid);
    EXPECT_EQ(rep.n_sat, V.n_sat);
    std::size_t n_bad = 0, first = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] != V.buf[i] && n_bad++ == 0) first = i;
    }
    EXPECT_EQ(n_bad, 0u) << "primera palabra distinta: " << first << " (de " << out.size() << ")";
}

TEST(PsSense, CordicGainMatchesReference)
{
    REQUIRE_VEC();
    if (V.k_iters > 0) {
        EXPECT_EQ(bits(cordic_gain(V.k_iters)), bits(V.cfg.k_cordic));
    }
    EXPECT_NEAR(cordic_gain(12), 1.646760, 5e-7);
    EXPECT_EQ(cordic_gain(0), 1.0);
}

// word_index recorre el buffer entero sin repetir, en los dos layouts.
TEST(PsSense, WordIndexIsAPermutation)
{
    REQUIRE_VEC();
    const dims& dm = V.dm;
    for (layout lay : {layout::pos_coil_a, layout::coil_pos_a}) {
        std::vector<std::uint8_t> hit(dm.buffer_words(), 0);
        std::size_t bad = 0;
        for (std::size_t o = 0; o < dm.positions(); ++o) {
            for (int l = 0; l < dm.L; ++l) {
                for (int a = 0; a < dm.af; ++a) {
                    const std::size_t w = word_index(dm, lay, o, l, a);
                    if (w >= hit.size() || hit[w]++ != 0) ++bad;
                }
            }
        }
        EXPECT_EQ(bad, 0u) << "layout " << static_cast<int>(lay);
    }
}

TEST(PsSense, BitrevIsAnInvolution)
{
    for (int b = 0; b <= 12; ++b) {
        for (std::uint32_t i = 0; i < (1u << b); ++i) {
            ASSERT_EQ(bitrev(bitrev(i, b), b), i) << "bits " << b;
        }
    }
    EXPECT_EQ(bitrev(1, 10), 512u);
    EXPECT_EQ(log2_exact(1024), 10);
    EXPECT_EQ(log2_exact(48), -1);
    EXPECT_EQ(log2_exact(0), -1);
}

TEST(PsSense, QuantizeRoundsToEvenAndSaturates)
{
    std::uint32_t sat = 0;
    EXPECT_EQ(quantize(0.5, 0, 16, &sat), 0);       // empate: al par
    EXPECT_EQ(quantize(1.5, 0, 16, &sat), 2);
    EXPECT_EQ(quantize(-2.5, 0, 16, &sat), -2);
    EXPECT_EQ(quantize(0x1p-15 * 3.0, 15, 16, &sat), 3);
    EXPECT_EQ(sat, 0u);
    EXPECT_EQ(quantize(32767.5, 0, 16, &sat), 32767);   // rint da 32768: satura
    EXPECT_EQ(quantize(-32768.4, 0, 16, &sat), -32768); // rint da -32768: no satura
    EXPECT_EQ(quantize(-40000.0, 0, 16, &sat), -32768);
    EXPECT_EQ(sat, 2u);
    EXPECT_EQ(pack_word(-1, 1), 0x0001FFFFu);
    EXPECT_EQ(pack_word(-32768, 32767), 0x7FFF8000u);
}

TEST(PsSense, RejectsInvalidInputs)
{
    double smaps[2 * 8 * 4 * 4] = {};
    double g[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::uint32_t out[4 * 2 * 2 * 8];
    dims dm;
    dm.L = 8; dm.af = 2; dm.n_ro = 4; dm.n_pe = 4;
    config cfg;
    EXPECT_EQ(build_buffer(dm, nullptr, g, cfg, out), status::null_ptr);
    EXPECT_EQ(build_buffer(dm, smaps, nullptr, cfg, out), status::null_ptr);
    EXPECT_EQ(build_buffer(dm, smaps, g, cfg, nullptr), status::null_ptr);

    dims bad = dm;
    bad.n_ro = 3;
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);
    bad = dm; bad.n_pe = 6;                 // n2 = 3: no es potencia de 2
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);
    bad = dm; bad.n_pe = 5;                 // no divisible por Af
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);
    bad = dm; bad.af = MAX_AF + 1;
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);
    bad = dm; bad.L = 0;
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);
    bad = dm; bad.L = MAX_L + 1;
    EXPECT_EQ(build_buffer(bad, smaps, g, cfg, out), status::bad_dims);

    config c = cfg; c.lam = -1e-9;
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    c = cfg; c.lam = NAN;
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    c = cfg; c.k_cordic = -1.0;
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    c = cfg; c.nb = 1;
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    c = cfg; c.nb = 17;
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    c = cfg; c.lay = static_cast<layout>(2);
    EXPECT_EQ(build_buffer(dm, smaps, g, c, out), status::bad_param);
    double g_bad[8] = {1, 1, 1, 0.0, 1, 1, 1, 1};
    EXPECT_EQ(build_buffer(dm, smaps, g_bad, cfg, out), status::bad_param);
    g_bad[3] = INFINITY;
    EXPECT_EQ(build_buffer(dm, smaps, g_bad, cfg, out), status::bad_param);

    // mapas en cero con lam = 0: todas invalidas, buffer en 0, E = 0
    report rep;
    ASSERT_EQ(build_buffer(dm, smaps, g, cfg, out, &rep), status::ok);
    EXPECT_EQ(rep.n_invalid, static_cast<std::uint32_t>(dm.positions()));
    EXPECT_EQ(rep.E, 0);
    for (std::uint32_t w : out) EXPECT_EQ(w, 0u);
}

}  // namespace

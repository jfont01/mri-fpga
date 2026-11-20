#pragma once

// -----------------------------------------------------------------------------
// unfold_stimulus.hpp -- estimulo de unfold, comun al testbench de sim
// (unfold_tb.cpp) y al gtest (unfold_gtest.cpp): con el mismo caso del JSON los
// dos ven exactamente los mismos ciclos.
//
// Los valores los escribe model/gen_unfold_vectors.py (ver su docstring) en la
// carpeta del caso, clave VEC_DIR del JSON -> UNFOLD_VEC_DIR, relativa a la
// carpeta del modulo (run_gtest y run_regression_sim corren desde ahi):
//
//   case.txt       clave valor: parametros con que se genero, n, n_out, ...
//   stream.bin     int32 little-endian [n][2 + 2 AF + 2]: a_re, a_im,
//                  u_re[0], u_im[0], ..., u_re[AF-1], u_im[AF-1], last, d
//                  (las muestras validas, en orden)
//   p_ref.bin      int64 [n_out][AF][2]   P de cada posicion cerrada (Python)
//   m_ref.bin      int32 [n_out][AF][2]   M (Python)
//   close_ref.bin  int32 [n_out][2]       muestra que cierra la posicion, err
//   con e2e = 1 (caso punta a punta), ademas:
//   mag_ref.bin    int32 [n_out][AF]      salida de cordic_mag (output_stage)
//   xn_ref.bin     int32 [n_out][AF]      x_n del CORDIC
//
// Ciclo a ciclo (build), una muestra valida por ciclo con huecos al azar:
//
//   UNFOLD_GAP_PCT   0: sin huecos, la muestra k entra en el ciclo k.   (0)
//                    p > 0: antes de la primera muestra van N_STAGES ciclos
//                    con i_valid = 0 (llenan el pipeline de basura), y antes
//                    de cada muestra, huecos mientras rng() % 100 < p
//                    (geometrica: media p / (100 - p)). Los huecos caen
//                    tambien entre bobinas de una misma posicion.
//   UNFOLD_GAP_SEED  semilla del generador de los huecos y la basura.   (1)
//
// Con i_valid = 0, todas las entradas llevan basura en rango (A en NB_A bits,
// U en NB_U, i_last 0 o 1, i_d en NB_D bits sin signo): el modelo y el RTL la
// tienen que ignorar (en particular, un i_last basura no cierra nada), y el
// gtest chequea que las salidas no se muevan. Despues de la ultima muestra van
// LATENCY + N_STAGES ciclos de basura (la salida de la ultima y una cola).
//
// Aleatorio: la salida cruda de std::mt19937_64, que el estandar fija bit a
// bit; las distribuciones de <random> no estan fijadas y cambian entre
// bibliotecas, y el vm necesita los mismos vectores en cualquier PC.
// -----------------------------------------------------------------------------

#include "unfold.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef UNFOLD_GAP_PCT
#define UNFOLD_GAP_PCT 0
#endif
#ifndef UNFOLD_GAP_SEED
#define UNFOLD_GAP_SEED 1
#endif
#ifndef UNFOLD_E2E
#define UNFOLD_E2E 0
#endif

#define UNFOLD_STR2(x) #x
#define UNFOLD_STR(x) UNFOLD_STR2(x)

namespace unfold_stimulus {

using unfold::AF;
using unfold::word_t;

constexpr int           GAP_PCT  = UNFOLD_GAP_PCT;
constexpr std::uint64_t GAP_SEED = UNFOLD_GAP_SEED;
constexpr int           E2E      = UNFOLD_E2E;

static_assert(GAP_PCT >= 0 && GAP_PCT <= 99, "unfold_stimulus: GAP_PCT tiene que estar en [0, 99]");
static_assert(E2E == 0 || E2E == 1, "unfold_stimulus: E2E tiene que ser 0 o 1");

using arr_t = std::array<word_t, AF>;

constexpr std::size_t STREAM_COLS = 2 + 2 * AF + 2;   // a_re, a_im, (u_re, u_im) x AF, last, d

// Carpeta del caso (VEC_DIR). Vacia si no se compilo con UNFOLD_VEC_DIR.
inline std::string vec_dir()
{
#ifdef UNFOLD_VEC_DIR
    return UNFOLD_STR(UNFOLD_VEC_DIR);
#else
    return "";
#endif
}

inline std::string regen_hint()
{
    return "regenera los vectores con python3 model/gen_unfold_vectors.py --from-json "
           "modules/unfold/testbench/unfold_tb_regression.json (o revisa el JSON)";
}

// --------------------------------------------------------------------- caso

struct Sample {
    word_t a_re = 0, a_im = 0;
    arr_t  u_re {}, u_im {};
    bool   last = false;
    word_t d = 0;
};

struct Case {
    std::string dir;
    std::map<std::string, std::string> info;    // case.txt tal cual
    std::size_t n = 0;                           // muestras validas
    std::size_t n_out = 0;                       // posiciones cerradas
    std::vector<Sample> stream;
    std::vector<arr_t> p_re, p_im, m_re, m_im;   // [n_out]
    std::vector<long>  close;                    // muestra que cierra la posicion k
    std::vector<int>   err;
    std::vector<arr_t> mag, xn;                  // solo con e2e = 1

    // Entero de case.txt; tira si falta o no es un entero.
    long get(const std::string& key) const
    {
        const auto it = info.find(key);
        if (it == info.end()) {
            throw std::runtime_error(dir + "/case.txt: falta '" + key + "' (" + regen_hint() + ")");
        }
        std::size_t used = 0;
        const long v = std::stol(it->second, &used, 10);
        if (used != it->second.size()) {
            throw std::runtime_error(dir + "/case.txt: '" + key + "' no es un entero: " + it->second);
        }
        return v;
    }
};

template <typename T>
std::vector<T> read_bin(const std::string& path, std::size_t count)
{
    std::vector<T> v(count);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("no se puede abrir " + path + " (" + regen_hint() + ")");
    }
    const std::size_t got = std::fread(v.data(), sizeof(T), count, f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != count || extra) {
        throw std::runtime_error(path + ": no tiene " + std::to_string(count) + " enteros de " +
                                 std::to_string(8 * sizeof(T)) + " bits");
    }
    return v;
}

// [n][AF] de un .bin int32 o int64 a vector<arr_t>; stride = valores por fila, off = primero.
template <typename T>
std::vector<arr_t> to_arrays(const std::vector<T>& raw, std::size_t rows, std::size_t stride, std::size_t off,
                             std::size_t step)
{
    std::vector<arr_t> out(rows);
    for (std::size_t k = 0; k < rows; ++k) {
        for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
            out[k][a] = static_cast<word_t>(raw[k * stride + off + a * step]);
        }
    }
    return out;
}

// Lee el caso de la carpeta dir. Tira std::runtime_error si algo falta.
inline Case load(const std::string& dir)
{
    if (dir.empty()) {
        throw std::runtime_error("sin carpeta de vectores: compilar con -DUNFOLD_VEC_DIR (clave VEC_DIR del JSON)");
    }
    Case cs;
    cs.dir = dir;
    std::ifstream f(dir + "/case.txt");
    if (!f) {
        throw std::runtime_error("no se puede abrir " + dir + "/case.txt (" + regen_hint() + ")");
    }
    std::string key, val;
    while (f >> key >> val) {
        cs.info[key] = val;
    }
    const long n = cs.get("n");
    const long n_out = cs.get("n_out");
    if (n <= 0 || n_out <= 0) {
        throw std::runtime_error(dir + "/case.txt: n = " + std::to_string(n) + ", n_out = " + std::to_string(n_out));
    }
    if (cs.get("af") != AF) {    // sin esto, los .bin se leerian con otro ancho de fila
        throw std::runtime_error(dir + "/case.txt: af = " + std::to_string(cs.get("af")) + " y AF compilado = " +
                                 std::to_string(AF) + ": " + regen_hint());
    }
    cs.n = static_cast<std::size_t>(n);
    cs.n_out = static_cast<std::size_t>(n_out);

    const std::vector<std::int32_t> st = read_bin<std::int32_t>(dir + "/stream.bin", STREAM_COLS * cs.n);
    cs.stream.resize(cs.n);
    for (std::size_t k = 0; k < cs.n; ++k) {
        const std::int32_t* r = &st[k * STREAM_COLS];
        Sample& s = cs.stream[k];
        s.a_re = r[0];
        s.a_im = r[1];
        for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
            s.u_re[a] = r[2 + 2 * a];
            s.u_im[a] = r[3 + 2 * a];
        }
        s.last = r[STREAM_COLS - 2] != 0;
        s.d = r[STREAM_COLS - 1];
    }

    const std::size_t w = 2 * static_cast<std::size_t>(AF);
    const std::vector<std::int64_t> p = read_bin<std::int64_t>(dir + "/p_ref.bin", w * cs.n_out);
    const std::vector<std::int32_t> m = read_bin<std::int32_t>(dir + "/m_ref.bin", w * cs.n_out);
    const std::vector<std::int32_t> c = read_bin<std::int32_t>(dir + "/close_ref.bin", 2 * cs.n_out);
    cs.p_re = to_arrays(p, cs.n_out, w, 0, 2);
    cs.p_im = to_arrays(p, cs.n_out, w, 1, 2);
    cs.m_re = to_arrays(m, cs.n_out, w, 0, 2);
    cs.m_im = to_arrays(m, cs.n_out, w, 1, 2);
    cs.close.resize(cs.n_out);
    cs.err.resize(cs.n_out);
    for (std::size_t k = 0; k < cs.n_out; ++k) {
        cs.close[k] = c[2 * k];
        cs.err[k] = c[2 * k + 1];
        if (cs.close[k] < 0 || cs.close[k] >= n || (k > 0 && cs.close[k] <= cs.close[k - 1])) {
            throw std::runtime_error(dir + "/close_ref.bin: indice de cierre invalido en la salida " + std::to_string(k));
        }
    }
    if (cs.get("e2e") == 1) {
        const std::size_t na = static_cast<std::size_t>(AF);
        cs.mag = to_arrays(read_bin<std::int32_t>(dir + "/mag_ref.bin", na * cs.n_out), cs.n_out, na, 0, 1);
        cs.xn = to_arrays(read_bin<std::int32_t>(dir + "/xn_ref.bin", na * cs.n_out), cs.n_out, na, 0, 1);
    }
    return cs;
}

// Parametros de case.txt distintos de los compilados ("" si coinciden).
inline std::string param_mismatch(const Case& cs)
{
    std::string out;
    const auto chk = [&](const char* key, long compiled) {
        const long v = cs.get(key);
        if (v != compiled) {
            out += std::string(out.empty() ? "" : ", ") + key + " = " + std::to_string(v) + " en case.txt y " +
                   std::to_string(compiled) + " compilado";
        }
    };
    chk("l", unfold::L);
    chk("af", unfold::AF);
    chk("nb_a", unfold::NB_A);
    chk("nb_u", unfold::NB_U);
    chk("g", unfold::G);
    chk("nb_m", unfold::NB_M);
    chk("round_conv", unfold::ROUND_CONV);
    chk("nb_d", unfold::NB_D);
    chk("nb_p", unfold::NB_P);
    chk("e2e", E2E);
    return out;
}

// ------------------------------------------------------------ ciclo a ciclo

struct Cycle {
    bool   valid;
    Sample in;
    long   k;          // indice de la muestra (-1 si i_valid = 0)
};

struct Schedule {
    std::vector<Cycle> cycles;     // entrada de cada ciclo (se pone antes del sim.cycle() numero c)
    std::vector<long>  in_cycle;   // ciclo en que entra la muestra k
    long idle_before_last = 0;     // ciclos con i_valid = 0 antes de la ultima muestra
};

// Estimulo completo: el que usa el gtest y, con N_CYCLES = 0, el tb. min_cycles
// alarga la cola con mas basura (N_CYCLES mayor que el automatico).
inline Schedule build(const Case& cs, std::size_t min_cycles = 0)
{
    using unfold::NB_A;
    using unfold::NB_D;
    using unfold::NB_U;
    std::mt19937_64 rng(GAP_SEED);
    const auto garbage = [&rng](int nb) {
        return unfold::smin(nb) + static_cast<word_t>(rng() & static_cast<std::uint64_t>(unfold::umax(nb)));
    };

    Schedule s;
    const auto idle = [&]() {
        Cycle c {false, Sample {}, -1};
        c.in.a_re = garbage(NB_A);
        c.in.a_im = garbage(NB_A);
        for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
            c.in.u_re[a] = garbage(NB_U);
            c.in.u_im[a] = garbage(NB_U);
        }
        c.in.last = (rng() & 1u) != 0;
        c.in.d = static_cast<word_t>(rng() & static_cast<std::uint64_t>(unfold::umax(NB_D)));
        s.cycles.push_back(c);
    };

    s.in_cycle.resize(cs.n);
    if (GAP_PCT > 0) {
        for (int i = 0; i < unfold::N_STAGES; ++i) idle();
    }
    for (std::size_t k = 0; k < cs.n; ++k) {
        if (GAP_PCT > 0) {
            while (rng() % 100u < static_cast<std::uint64_t>(GAP_PCT)) idle();
        }
        s.in_cycle[k] = static_cast<long>(s.cycles.size());
        s.cycles.push_back(Cycle {true, cs.stream[k], static_cast<long>(k)});
    }
    s.idle_before_last = static_cast<long>(s.cycles.size() - cs.n);
    for (int i = 0; i < unfold::LATENCY + unfold::N_STAGES; ++i) idle();
    while (s.cycles.size() < min_cycles) idle();
    return s;
}

// Pone en los puertos de entrada del modelo la entrada del ciclo.
inline void drive(unfold::unfold_model& dut, const Cycle& c)
{
    dut.i_valid = unfold::bit_t(c.valid);
    dut.i_last  = unfold::bit_t(c.in.last);
    dut.i_a_re  = c.in.a_re;
    dut.i_a_im  = c.in.a_im;
    for (std::size_t a = 0; a < static_cast<std::size_t>(AF); ++a) {
        dut.i_u_re[a] = c.in.u_re[a];
        dut.i_u_im[a] = c.in.u_im[a];
    }
    dut.i_d = c.in.d;
}

} // namespace unfold_stimulus

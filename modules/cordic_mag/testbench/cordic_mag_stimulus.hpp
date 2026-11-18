#pragma once

// -----------------------------------------------------------------------------
// cordic_mag_stimulus.hpp -- estimulo de cordic_mag, comun al testbench de sim
// (cordic_mag_tb.cpp) y al gtest (cordic_mag_gtest.cpp): con el mismo caso del
// JSON los dos ven exactamente los mismos ciclos.
//
// Los valores los escribe model/gen_cordic_mag_vectors.py (ver su docstring) en
// la carpeta del caso, clave VEC_DIR del JSON -> CORDIC_MAG_VEC_DIR, relativa a
// la carpeta del modulo (run_gtest y run_regression_sim corren desde ahi):
//
//   case.txt     clave valor: parametros con que se genero, n, semilla, ...
//   m.bin        int32 little-endian [n][re, im]   entrada
//   xn_ref.bin   int32 [n]                         x_n del Python
//   out_ref.bin  int32 [n]                         o_mag del Python
//
// Ciclo a ciclo (build), una muestra valida por ciclo con huecos al azar:
//
//   CORDIC_MAG_GAP_PCT   0: sin huecos, la muestra k entra en el ciclo k.   (0)
//                        p > 0: antes de la primera muestra van N_STAGES
//                        ciclos con i_valid = 0 (llenan el pipeline de
//                        basura), y antes de cada muestra, huecos mientras
//                        rng() % 100 < p (geometrica: media p / (100 - p)).
//   CORDIC_MAG_GAP_SEED  semilla del generador de los huecos y la basura.   (1)
//
// Con i_valid = 0, i_m_re e i_m_im llevan basura en rango (NB_M bits con
// signo, uniforme): el modelo y el RTL la tienen que ignorar, y el gtest
// chequea que o_mag no se mueva. Despues de la ultima muestra van
// LATENCY + N_STAGES ciclos de basura (la salida de la ultima y una cola).
//
// Aleatorio: la salida cruda de std::mt19937_64, que el estandar fija bit a
// bit; las distribuciones de <random> (uniform_int_distribution, ...) no estan
// fijadas y cambian entre bibliotecas, y el vm necesita los mismos vectores en
// cualquier PC. mt19937_64 y no mt19937 porque la basura llega a 61 bits.
// -----------------------------------------------------------------------------

#include "cordic_mag.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CORDIC_MAG_GAP_PCT
#define CORDIC_MAG_GAP_PCT 0
#endif
#ifndef CORDIC_MAG_GAP_SEED
#define CORDIC_MAG_GAP_SEED 1
#endif

#define CORDIC_MAG_STR2(x) #x
#define CORDIC_MAG_STR(x) CORDIC_MAG_STR2(x)

namespace cordic_mag_stimulus {

using cordic_mag::word_t;

constexpr int           GAP_PCT  = CORDIC_MAG_GAP_PCT;
constexpr std::uint64_t GAP_SEED = CORDIC_MAG_GAP_SEED;

static_assert(GAP_PCT >= 0 && GAP_PCT <= 99, "cordic_mag_stimulus: GAP_PCT tiene que estar en [0, 99]");

// Carpeta del caso (VEC_DIR). Vacia si no se compilo con CORDIC_MAG_VEC_DIR.
inline std::string vec_dir()
{
#ifdef CORDIC_MAG_VEC_DIR
    return CORDIC_MAG_STR(CORDIC_MAG_VEC_DIR);
#else
    return "";
#endif
}

inline std::string regen_hint()
{
    return "regenera los vectores con python3 model/gen_cordic_mag_vectors.py --from-json "
           "modules/cordic_mag/testbench/cordic_mag_tb_regression.json (o revisa el JSON)";
}

// --------------------------------------------------------------------- caso

struct Case {
    std::string dir;
    std::map<std::string, std::string> info;    // case.txt tal cual
    std::size_t n = 0;                           // muestras
    std::vector<word_t> m_re, m_im, xn_ref, out_ref;

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

inline std::vector<std::int32_t> read_i32(const std::string& path, std::size_t count)
{
    std::vector<std::int32_t> v(count);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("no se puede abrir " + path + " (" + regen_hint() + ")");
    }
    const std::size_t got = std::fread(v.data(), sizeof(std::int32_t), count, f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != count || extra) {
        throw std::runtime_error(path + ": no tiene " + std::to_string(count) + " enteros int32");
    }
    return v;
}

// Lee el caso de la carpeta dir. Tira std::runtime_error si algo falta.
inline Case load(const std::string& dir)
{
    if (dir.empty()) {
        throw std::runtime_error("sin carpeta de vectores: compilar con -DCORDIC_MAG_VEC_DIR (clave VEC_DIR del JSON)");
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
    if (n <= 0) {
        throw std::runtime_error(dir + "/case.txt: n = " + std::to_string(n));
    }
    cs.n = static_cast<std::size_t>(n);

    const std::vector<std::int32_t> m   = read_i32(dir + "/m.bin", 2 * cs.n);
    const std::vector<std::int32_t> xn  = read_i32(dir + "/xn_ref.bin", cs.n);
    const std::vector<std::int32_t> out = read_i32(dir + "/out_ref.bin", cs.n);
    cs.m_re.resize(cs.n);
    cs.m_im.resize(cs.n);
    cs.xn_ref.assign(xn.begin(), xn.end());
    cs.out_ref.assign(out.begin(), out.end());
    for (std::size_t k = 0; k < cs.n; ++k) {
        cs.m_re[k] = m[2 * k];
        cs.m_im[k] = m[2 * k + 1];
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
            out += std::string(out.empty() ? "" : ", ") + key + " = " + std::to_string(v) +
                   " en case.txt y " + std::to_string(compiled) + " compilado";
        }
    };
    chk("nb_out", cordic_mag::NB_OUT);
    chk("g", cordic_mag::G);
    chk("n_it", cordic_mag::N_IT);
    chk("nb_m", cordic_mag::NB_M);
    chk("round_conv", cordic_mag::ROUND_CONV);
    return out;
}

// ------------------------------------------------------------ ciclo a ciclo

struct Cycle {
    bool   valid;
    word_t re;
    word_t im;
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
    using cordic_mag::NB_M;
    std::mt19937_64 rng(GAP_SEED);
    const auto garbage = [&rng]() {
        return cordic_mag::smin(NB_M) + static_cast<word_t>(rng() & static_cast<std::uint64_t>(cordic_mag::umax(NB_M)));
    };

    Schedule s;
    const auto idle = [&]() {
        const word_t re = garbage();
        const word_t im = garbage();
        s.cycles.push_back(Cycle {false, re, im, -1});
    };

    s.in_cycle.resize(cs.n);
    if (GAP_PCT > 0) {
        for (int i = 0; i < cordic_mag::N_STAGES; ++i) idle();
    }
    for (std::size_t k = 0; k < cs.n; ++k) {
        if (GAP_PCT > 0) {
            while (rng() % 100u < static_cast<std::uint64_t>(GAP_PCT)) idle();
        }
        s.in_cycle[k] = static_cast<long>(s.cycles.size());
        s.cycles.push_back(Cycle {true, cs.m_re[k], cs.m_im[k], static_cast<long>(k)});
    }
    s.idle_before_last = static_cast<long>(s.cycles.size() - cs.n);
    for (int i = 0; i < cordic_mag::LATENCY + cordic_mag::N_STAGES; ++i) idle();
    while (s.cycles.size() < min_cycles) idle();
    return s;
}

} // namespace cordic_mag_stimulus

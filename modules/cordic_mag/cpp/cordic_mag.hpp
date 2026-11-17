#pragma once

#include "rtlsim.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// cordic_mag.hpp -- modelo ciclo a ciclo y bit a bit de cordic_mag (paso D2):
// magnitud de M = (re, im) por CORDIC en modo vectoring, y redondeo a la salida.
// Referencia: model/mac_cordic_fxp.py (m_to_out); diseno en
// doc/diseno_unfold_cordic.md (secciones 4.3 y 4.5).
//
// Aritmetica (la misma que el Python):
//
//   etapa 0        x = |re|, y = |im|                 (primer cuadrante)
//   iteracion i    si y >= 0:  x += y >> i,  y -= x >> i
//   (i = 0..N_IT-1) si no:      x -= y >> i,  y += x >> i
//                  (los dos lados con los x, y de antes; >> aritmetico = floor)
//   salida         o_mag = sat_u(round(x / 2^G)), NB_OUT bits sin signo
//                  round: half-up, floor((x + 2^(G-1)) / 2^G); con ROUND_CONV = 1,
//                  empates al par, como cast.v
//
// x termina en ~K_n |M| (K_12 = 1,64676). La ganancia no se corrige aca: el PS la
// pliega en U'' (1/K_n), asi que M ya llega dividido por K_n.
//
// Pipeline: N_STAGES = N_IT + 2 registros (|.|, una iteracion por registro, la
// salida). Cada etapa tiene su valid y sus datos se cargan solo con ese valid
// (clock enable): entre salidas validas, o_mag se queda con el ultimo valor.
// El vm compara todos los ciclos, asi que el RTL tiene que hacer lo mismo. Sin
// reset: todo arranca en 0, como en la FFT.
//
// LATENCY, con la convencion de la FFT (gtest LatencyMatchesFormula): si la
// entrada se pone antes del sim.cycle() numero c, la salida se ve despues del
// sim.cycle() numero c + LATENCY. Con N_STAGES registros, LATENCY = N_STAGES - 1.
//
// Tipos: enteros de 64 bits con chequeo de rango contra los anchos del RTL
// (decision P5). Si un valor no entra (entrada fuera de NB_M bits, x o y fuera de
// NB_W bits), el modelo tira std::range_error en vez de envolver sin avisar.
// Anchos internos (bitacora 4.13): x llega a K sqrt(2) 2^(NB_M-1) < 2^(NB_M+1),
// sin signo; y queda en [-2^(NB_M-1), 2^(NB_M-1)], con signo: NB_W = NB_M + 1.
// -----------------------------------------------------------------------------

namespace cordic_mag {

#ifndef CORDIC_MAG_NB_OUT
#define CORDIC_MAG_NB_OUT 16          // bits de la salida, sin signo
#endif

#ifndef CORDIC_MAG_G
#define CORDIC_MAG_G 5                // bits de guarda: M tiene G bits fraccionales mas que la salida
#endif

#ifndef CORDIC_MAG_N_IT
#define CORDIC_MAG_N_IT 12            // iteraciones del CORDIC
#endif

/*
 * Ancho de M con signo. Misma regla que el Python (OutCfg.W_M): NB_OUT + 1 + G.
 * Es suficiente y es el minimo (bitacora 4.13): saturar M en 2^(NB_OUT + G) solo
 * recorta valores cuya salida igual satura.
 */
#ifndef CORDIC_MAG_NB_M
#define CORDIC_MAG_NB_M (CORDIC_MAG_NB_OUT + 1 + CORDIC_MAG_G)
#endif

/*
 * Redondeo de la salida: 0 = half-up (default, decision 3 de D2), 1 = empates al par.
 */
#ifndef CORDIC_MAG_ROUND_CONV
#define CORDIC_MAG_ROUND_CONV 0
#endif

enum parameters : int {
    NB_OUT     = CORDIC_MAG_NB_OUT,
    G          = CORDIC_MAG_G,
    N_IT       = CORDIC_MAG_N_IT,
    NB_M       = CORDIC_MAG_NB_M,
    ROUND_CONV = CORDIC_MAG_ROUND_CONV
};

enum localparameters : int {
    NB_W     = NB_M + 1,             // x (sin signo) e y (con signo) adentro del CORDIC
    N_STAGES = N_IT + 2,             // registros en serie
    LATENCY  = N_STAGES - 1          // convencion de la FFT (ver arriba)
};

static_assert(NB_OUT >= 1 && NB_OUT <= 32, "cordic_mag: NB_OUT tiene que estar en [1, 32]");
static_assert(G >= 0 && G <= 30, "cordic_mag: G tiene que estar en [0, 30]");
static_assert(N_IT >= 1 && N_IT <= 32, "cordic_mag: N_IT tiene que estar en [1, 32]");
static_assert(NB_M >= 2 && NB_W <= 62, "cordic_mag: NB_M tiene que estar en [2, 61] (enteros de 64 bits)");
static_assert(ROUND_CONV == 0 || ROUND_CONV == 1, "cordic_mag: ROUND_CONV tiene que ser 0 o 1");
static_assert((std::int64_t(-3) >> 1) == -2,
              "cordic_mag: el modelo necesita >> aritmetico en enteros con signo (en C++17 lo define el compilador)");

using bit_t  = rtl::bit_t;
using word_t = std::int64_t;

// Rango de un entero de nb bits con signo y sin signo.
constexpr word_t smin(int nb) { return -(word_t(1) << (nb - 1)); }
constexpr word_t smax(int nb) { return (word_t(1) << (nb - 1)) - 1; }
constexpr word_t umax(int nb) { return (word_t(1) << nb) - 1; }

/*
 * x / 2^g redondeado, con x >= 0. half-up: floor((x + 2^(g-1)) / 2^g). conv:
 * m + (r > h) + (r == h) lsb(m), con m = floor(x / 2^g), r = x - m 2^g, h = 2^(g-1)
 * (la misma regla que cast.v con ROUND_CONV = 1 y que mac_cordic_fxp.round_shift).
 */
constexpr word_t round_shift(word_t x, int g, bool conv)
{
    if (g == 0) {
        return x;
    }
    const word_t h = word_t(1) << (g - 1);
    if (!conv) {
        return (x + h) >> g;
    }
    const word_t m = x >> g;
    const word_t r = x - (m << g);
    return m + ((r > h || (r == h && (m & 1) != 0)) ? 1 : 0);
}

class cordic_mag_model final : public rtl::Module {
public:
    cordic_mag_model() = default;

    // --------------------------------------------------------------- puertos
    rtl::InPort<bit_t>   i_valid;
    rtl::InPort<word_t>  i_m_re;          // NB_M bits con signo
    rtl::InPort<word_t>  i_m_im;          // NB_M bits con signo

    rtl::OutPort<bit_t>  o_valid;
    rtl::OutPort<word_t> o_mag;           // NB_OUT bits sin signo

    // Solo para depurar (no existe en el RTL): x_n del dato que esta en o_mag,
    // antes del redondeo. El gtest lo compara contra el x_n del Python.
    word_t dbg_xn() const { return xn_r.o; }

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        for (auto& r : v_r) clk.add(r);
        for (auto& r : x_r) clk.add(r);
        for (auto& r : y_r) clk.add(r);
        clk.add(mag_r);
        clk.add(xn_r);
    }

    void init() override
    {
        i_valid = bit_t(false);
        i_m_re  = word_t(0);
        i_m_im  = word_t(0);
        o_valid = bit_t(false);
        o_mag   = word_t(0);

        for (auto& r : v_r) r.set_initial_value(bit_t(false));
        for (auto& r : x_r) r.set_initial_value(0);
        for (auto& r : y_r) r.set_initial_value(0);
        mag_r.set_initial_value(0);
        xn_r.set_initial_value(0);
        cycle_ = 0;
    }

    void combinational() override
    {
        o_valid = v_r[N_STAGES - 1].o;
        o_mag   = mag_r.o;
    }

    void sequential() override
    {
        // Registros que no se asignan mantienen su valor (el Simulator hace i = o
        // antes de llamar aca): eso es el clock enable.

        // Etapa 0: valor absoluto. La entrada se chequea solo si es valida: con
        // i_valid = 0 el dato no importa (el RTL lo ignora).
        const bool vin = i_valid.value().to_bool();
        v_r[0].i = bit_t(vin);
        if (vin) {
            const word_t re = check(i_m_re.value(), smin(NB_M), smax(NB_M), "i_m_re", -1);
            const word_t im = check(i_m_im.value(), smin(NB_M), smax(NB_M), "i_m_im", -1);
            x_r[0].i = re < 0 ? -re : re;
            y_r[0].i = im < 0 ? -im : im;
        }

        // Iteraciones: el registro k + 1 guarda el resultado de la iteracion k.
        for (int k = 0; k < N_IT; ++k) {
            v_r[k + 1].i = v_r[k].o;
            if (v_r[k].o.to_bool()) {
                const word_t x  = x_r[k].o;
                const word_t y  = y_r[k].o;
                const word_t xs = x >> k;
                const word_t ys = y >> k;
                const word_t xn = (y >= 0) ? x + ys : x - ys;
                const word_t yn = (y >= 0) ? y - xs : y + xs;
                x_r[k + 1].i = check(xn, 0, umax(NB_W), "x", k);
                y_r[k + 1].i = check(yn, smin(NB_W), smax(NB_W), "y", k);
            }
        }

        // Salida: redondeo de G bits y saturacion a NB_OUT bits sin signo.
        v_r[N_IT + 1].i = v_r[N_IT].o;
        if (v_r[N_IT].o.to_bool()) {
            const word_t x = x_r[N_IT].o;
            const word_t r = round_shift(x, G, ROUND_CONV == 1);
            mag_r.i = r > umax(NB_OUT) ? umax(NB_OUT) : r;
            xn_r.i  = x;
        }

        ++cycle_;
    }

private:
    std::array<rtl::Reg<bit_t>,  N_STAGES> v_r {};    // valid de cada registro
    std::array<rtl::Reg<word_t>, N_IT + 1> x_r {};    // [0]: |re|; [k + 1]: tras la iteracion k
    std::array<rtl::Reg<word_t>, N_IT + 1> y_r {};
    rtl::Reg<word_t> mag_r {};                         // o_mag
    rtl::Reg<word_t> xn_r {};                          // depuracion (no esta en el RTL)
    std::uint64_t cycle_ {0};

    word_t check(word_t v, word_t lo, word_t hi, const char* what, int it) const
    {
        if (v < lo || v > hi) {
            throw std::range_error(
                std::string("cordic_mag: ") + what + " = " + std::to_string(v) + " fuera de ["
                + std::to_string(lo) + ", " + std::to_string(hi) + "]"
                + (it >= 0 ? " en la iteracion " + std::to_string(it) : std::string())
                + " (ciclo " + std::to_string(cycle_) + ")");
        }
        return v;
    }
};

} // namespace cordic_mag

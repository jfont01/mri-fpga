#pragma once

#include "rtlsim.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// unfold.hpp -- modelo ciclo a ciclo y bit a bit de unfold (paso D2), un carril:
// MAC de desplegado (U'' * A) y ajuste de P a M para el CORDIC.
// Referencia: model/mac_cordic_fxp.py (mac y p_to_m); diseno en
// doc/diseno_unfold_cordic.md (secciones 4.2 y 4.6, decision 13).
//
// Por cada posicion plegada entran las L bobinas de a una (una muestra valida por
// ciclo, con huecos en cualquier lugar, tambien adentro de una posicion). Cada
// muestra trae A_l y sus AF coeficientes U''_{a,l}. Para a = 0 ... AF-1:
//
//   P_a = sum_l U''_{a,l} * A_l           producto complejo exacto:
//                                         re = ur*ar - ui*ai, im = ur*ai + ui*ar
//   M_a = sat_NB_M(round(P_a / 2^s))      s = D - G, de -G a 2^NB_D - 1 - G
//
//   round, con s > 0: half-up, floor((P + 2^(s-1)) / 2^s), como cast.v; con
//   ROUND_CONV = 1, empates al par. Con s <= 0: corrimiento a la izquierda
//   exacto (y despues la saturacion). Igual que mac_cordic_fxp.p_to_m.
//
// Cierre de una posicion (decision 13). Un contador l cuenta las muestras validas
// de la posicion actual. La posicion se CIERRA con la muestra valida que trae
// i_last = 1 o con la muestra numero L - 1 (l == L - 1), lo que pase primero. Al
// cerrar sale o_valid con los AF valores de M, y o_err = 1 junto con ese o_valid
// si las dos condiciones no coincidieron (i_last con l != L - 1, o l == L - 1
// sin i_last); despues l vuelve a 0. o_err no es pegajoso (el pegajoso va en
// sense_ctrl). i_d se toma con la muestra que cierra la posicion. La primera
// muestra de una posicion carga el acumulador y las siguientes suman.
// Resincronizacion: con una muestra perdida sale mal una posicion (dos si la
// perdida es la del i_last); con una de mas, dos (una si la repetida es la del
// i_last); despues vuelve a estar alineado.
//
// Pipeline: N_STAGES = 4 registros.
//   1. entrada       valid, last, l, d, A, U[AF]
//   2. productos     4 por a, exactos (ur*ar, ui*ai, ur*ai, ui*ar); first,
//                    close y err salen de l y last
//   3. acumulador    P[AF] (NB_P bits); su valid es valid && close
//   4. salida        M[AF] = sat(round(P / 2^s)), o_err, P de depuracion
// Los datos de cada registro se cargan solo con su valid (clock enable): entre
// salidas validas o_m, o_err y el P de depuracion quedan quietos. El vm compara
// todos los ciclos, asi que el RTL tiene que hacer lo mismo. Sin reset: todo
// arranca en 0, como en la FFT y en cordic_mag.
//
// LATENCY, con la convencion de la FFT (gtest LatencyMatchesFormula): si la
// muestra que cierra la posicion se pone antes del sim.cycle() numero c, o_valid
// se ve despues del sim.cycle() numero c + LATENCY. LATENCY = N_STAGES - 1 = 3.
//
// Tipos: enteros de 64 bits con chequeo de rango contra los anchos del RTL
// (decision P5). Si una entrada valida no entra en su ancho (A, U, i_d) o P se
// sale de NB_P bits, el modelo tira std::range_error en vez de envolver.
// NB_P = NB_A + NB_U + 1 + ceil(log2 L) (H31): con A y U'' en el minimo en las L
// bobinas, P_im = +L 2^(NB_A+NB_U-1) (36 bits con los defaults: P_im = +2^34).
// -----------------------------------------------------------------------------

namespace unfold {

#ifndef UNFOLD_L
#define UNFOLD_L 8                    // bobinas por posicion plegada
#endif

#ifndef UNFOLD_AF
#define UNFOLD_AF 2                   // factor de aceleracion: pixeles por posicion
#endif

#ifndef UNFOLD_NB_A
#define UNFOLD_NB_A 16                // A_l (salida de la fft2d), con signo
#endif

#ifndef UNFOLD_NB_U
#define UNFOLD_NB_U 16                // U'' (buffer del PS), con signo
#endif

#ifndef UNFOLD_G
#define UNFOLD_G 5                    // bits de guarda: M tiene G bits fraccionales mas que la salida
#endif

/*
 * Ancho de M con signo: 17 + G, la misma regla que el Python (OutCfg.W_M =
 * W_OUT + 1 + G, con W_OUT = 16) y el mismo default que CORDIC_MAG_NB_M.
 */
#ifndef UNFOLD_NB_M
#define UNFOLD_NB_M (17 + UNFOLD_G)
#endif

/*
 * Redondeo de P a M: 0 = half-up (default, decision 3 de D2), 1 = empates al par.
 */
#ifndef UNFOLD_ROUND_CONV
#define UNFOLD_ROUND_CONV 0
#endif

#ifndef UNFOLD_NB_D
#define UNFOLD_NB_D 6                 // i_d, sin signo (decision 11): s = D - G
#endif

enum parameters : int {
    L          = UNFOLD_L,
    AF         = UNFOLD_AF,
    NB_A       = UNFOLD_NB_A,
    NB_U       = UNFOLD_NB_U,
    G          = UNFOLD_G,
    NB_M       = UNFOLD_NB_M,
    ROUND_CONV = UNFOLD_ROUND_CONV,
    NB_D       = UNFOLD_NB_D
};

// ceil(log2 v), con clog2(1) = 0 (como $clog2 de Verilog).
constexpr int clog2(int v)
{
    int r = 0;
    while ((1L << r) < v) {
        ++r;
    }
    return r;
}

enum localparameters : int {
    NB_P     = NB_A + NB_U + 1 + clog2(L),   // acumulador (H31)
    S_MIN    = -G,                           // s = D - G con D = 0
    S_MAX    = (1 << NB_D) - 1 - G,          // s con D = 2^NB_D - 1
    N_STAGES = 4,                            // registros en serie
    LATENCY  = N_STAGES - 1                  // convencion de la FFT (ver arriba)
};

static_assert(L >= 1 && L <= 256, "unfold: L tiene que estar en [1, 256]");
static_assert(AF >= 1 && AF <= 16, "unfold: AF tiene que estar en [1, 16]");
static_assert(NB_A >= 2 && NB_A <= 31, "unfold: NB_A tiene que estar en [2, 31]");
static_assert(NB_U >= 2 && NB_U <= 31, "unfold: NB_U tiene que estar en [2, 31]");
static_assert(G >= 0 && G <= 30, "unfold: G tiene que estar en [0, 30]");
static_assert(NB_M >= 2 && NB_M <= 61, "unfold: NB_M tiene que estar en [2, 61]");
static_assert(ROUND_CONV == 0 || ROUND_CONV == 1, "unfold: ROUND_CONV tiene que ser 0 o 1");
static_assert(NB_D >= 1 && NB_D <= 7, "unfold: NB_D tiene que estar en [1, 7]");
// Con s = -G, P se corre G bits a la izquierda: |P 2^G| < 2^(NB_P - 1 + G) tiene
// que entrar en int64.
static_assert(NB_P + G <= 63, "unfold: P * 2^G (s minimo = -G) no entra en 64 bits: bajar NB_A, NB_U, L o G");
// s maximo: el round_shift del Python (la referencia) acepta s <= 62; en C++,
// 2^(s-1) y P + 2^(s-1) entran en int64 (|P| < 2^61, por el assert de arriba).
static_assert(S_MAX <= 62, "unfold: s maximo = 2^NB_D - 1 - G tiene que ser <= 62 (con G = 0, NB_D <= 5)");
static_assert((std::int64_t(-3) >> 1) == -2,
              "unfold: el modelo necesita >> aritmetico en enteros con signo (en C++17 lo define el compilador)");

using bit_t  = rtl::bit_t;
using word_t = std::int64_t;

// Rango de un entero de nb bits con signo y sin signo.
constexpr word_t smin(int nb) { return -(word_t(1) << (nb - 1)); }
constexpr word_t smax(int nb) { return (word_t(1) << (nb - 1)) - 1; }
constexpr word_t umax(int nb) { return (word_t(1) << nb) - 1; }

/*
 * x / 2^s redondeado, x con signo (mac_cordic_fxp.round_shift).
 *   s <= 0: x 2^(-s), exacto (multiplicacion: en C++17 << de un negativo es UB).
 *   s > 0, half-up: floor((x + 2^(s-1)) / 2^s).
 *   s > 0, conv: m + (r > h) + (r == h) lsb(m), con m = floor(x / 2^s),
 *          r = x - m 2^s en [0, 2^s) y h = 2^(s-1) (la regla de cast.v con
 *          ROUND_CONV = 1). r son los s bits bajos de x en complemento a 2.
 * Validez: s <= 62, |x| < 2^61 y |x 2^(-s)| < 2^63 (static_assert del modelo).
 */
constexpr word_t round_shift(word_t x, int s, bool conv)
{
    if (s <= 0) {
        return x * (word_t(1) << -s);
    }
    const word_t h = word_t(1) << (s - 1);
    if (!conv) {
        return (x + h) >> s;
    }
    const word_t m = x >> s;
    const word_t r = static_cast<word_t>(static_cast<std::uint64_t>(x) & ((std::uint64_t(1) << s) - 1));
    return m + ((r > h || (r == h && (m & 1) != 0)) ? 1 : 0);
}

constexpr word_t sat(word_t x, int nb)
{
    return x < smin(nb) ? smin(nb) : (x > smax(nb) ? smax(nb) : x);
}

class unfold_model final : public rtl::Module {
public:
    unfold_model() = default;

    // --------------------------------------------------------------- puertos
    rtl::InPort<bit_t>  i_valid;
    rtl::InPort<bit_t>  i_last;                       // ultima bobina de la posicion
    rtl::InPort<word_t> i_a_re;                       // NB_A bits con signo
    rtl::InPort<word_t> i_a_im;
    std::array<rtl::InPort<word_t>, AF> i_u_re;       // NB_U bits con signo, uno por a
    std::array<rtl::InPort<word_t>, AF> i_u_im;
    rtl::InPort<word_t> i_d;                          // NB_D bits sin signo

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<bit_t> o_err;                        // la posicion no cerro con i_last y l == L - 1 juntos
    std::array<rtl::OutPort<word_t>, AF> o_m_re;      // NB_M bits con signo
    std::array<rtl::OutPort<word_t>, AF> o_m_im;

    // Solo para depurar (no existe en el RTL): el P de la posicion que esta en
    // o_m, antes del redondeo. El gtest lo compara contra el P del Python.
    word_t dbg_p_re(int a) const { return pd_re_r[static_cast<std::size_t>(a)].o; }
    word_t dbg_p_im(int a) const { return pd_im_r[static_cast<std::size_t>(a)].o; }

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        clk.add(cnt_r);
        clk.add(v1_r);
        clk.add(last1_r);
        clk.add(l1_r);
        clk.add(d1_r);
        clk.add(a1_re_r);
        clk.add(a1_im_r);
        clk.add(v2_r);
        clk.add(first2_r);
        clk.add(close2_r);
        clk.add(err2_r);
        clk.add(d2_r);
        clk.add(v3_r);
        clk.add(err3_r);
        clk.add(d3_r);
        clk.add(v4_r);
        clk.add(err4_r);
        for (int a = 0; a < AF; ++a) {
            const std::size_t k = static_cast<std::size_t>(a);
            clk.add(u1_re_r[k]);
            clk.add(u1_im_r[k]);
            clk.add(rr2_r[k]);
            clk.add(ii2_r[k]);
            clk.add(ri2_r[k]);
            clk.add(ir2_r[k]);
            clk.add(p_re_r[k]);
            clk.add(p_im_r[k]);
            clk.add(m_re_r[k]);
            clk.add(m_im_r[k]);
            clk.add(pd_re_r[k]);
            clk.add(pd_im_r[k]);
        }
    }

    void init() override
    {
        i_valid = bit_t(false);
        i_last  = bit_t(false);
        i_a_re  = word_t(0);
        i_a_im  = word_t(0);
        i_d     = word_t(0);
        o_valid = bit_t(false);
        o_err   = bit_t(false);
        for (int a = 0; a < AF; ++a) {
            const std::size_t k = static_cast<std::size_t>(a);
            i_u_re[k] = word_t(0);
            i_u_im[k] = word_t(0);
            o_m_re[k] = word_t(0);
            o_m_im[k] = word_t(0);
        }

        cnt_r.set_initial_value(0);
        for (auto* r : {&v1_r, &last1_r, &v2_r, &first2_r, &close2_r, &err2_r, &v3_r, &err3_r, &v4_r, &err4_r}) {
            r->set_initial_value(bit_t(false));
        }
        for (auto* r : {&l1_r, &d1_r, &a1_re_r, &a1_im_r, &d2_r, &d3_r}) {
            r->set_initial_value(0);
        }
        for (int a = 0; a < AF; ++a) {
            const std::size_t k = static_cast<std::size_t>(a);
            for (auto* r : {&u1_re_r[k], &u1_im_r[k], &rr2_r[k], &ii2_r[k], &ri2_r[k], &ir2_r[k], &p_re_r[k],
                            &p_im_r[k], &m_re_r[k], &m_im_r[k], &pd_re_r[k], &pd_im_r[k]}) {
                r->set_initial_value(0);
            }
        }
        cycle_ = 0;
    }

    void combinational() override
    {
        o_valid = v4_r.o;
        o_err   = err4_r.o;
        for (int a = 0; a < AF; ++a) {
            const std::size_t k = static_cast<std::size_t>(a);
            o_m_re[k] = m_re_r[k].o;
            o_m_im[k] = m_im_r[k].o;
        }
    }

    void sequential() override
    {
        // Registros que no se asignan mantienen su valor (el Simulator hace i = o
        // antes de llamar aca): eso es el clock enable.

        // Etapa 1: registro de entrada y contador de bobinas. La entrada se
        // chequea solo si es valida: con i_valid = 0 el dato no importa.
        const bool vin = i_valid.value().to_bool();
        v1_r.i = bit_t(vin);
        if (vin) {
            const bool   last = i_last.value().to_bool();
            const word_t l    = cnt_r.o;
            a1_re_r.i = check(i_a_re.value(), smin(NB_A), smax(NB_A), "i_a_re", -1);
            a1_im_r.i = check(i_a_im.value(), smin(NB_A), smax(NB_A), "i_a_im", -1);
            for (int a = 0; a < AF; ++a) {
                const std::size_t k = static_cast<std::size_t>(a);
                u1_re_r[k].i = check(i_u_re[k].value(), smin(NB_U), smax(NB_U), "i_u_re", a);
                u1_im_r[k].i = check(i_u_im[k].value(), smin(NB_U), smax(NB_U), "i_u_im", a);
            }
            d1_r.i    = check(i_d.value(), 0, umax(NB_D), "i_d", -1);
            last1_r.i = bit_t(last);
            l1_r.i    = l;
            // La posicion se cierra con i_last o con la bobina L - 1.
            cnt_r.i = (last || l == L - 1) ? 0 : l + 1;
        }

        // Etapa 2: productos (exactos) y banderas de la posicion.
        v2_r.i = v1_r.o;
        if (v1_r.o.to_bool()) {
            const bool last  = last1_r.o.to_bool();
            const bool count = l1_r.o == L - 1;
            first2_r.i = bit_t(l1_r.o == 0);
            close2_r.i = bit_t(last || count);
            err2_r.i   = bit_t(last != count);
            d2_r.i     = d1_r.o;
            const word_t ar = a1_re_r.o;
            const word_t ai = a1_im_r.o;
            for (int a = 0; a < AF; ++a) {
                const std::size_t k = static_cast<std::size_t>(a);
                const word_t ur = u1_re_r[k].o;
                const word_t ui = u1_im_r[k].o;
                rr2_r[k].i = ur * ar;
                ii2_r[k].i = ui * ai;
                ri2_r[k].i = ur * ai;
                ir2_r[k].i = ui * ar;
            }
        }

        // Etapa 3: acumulador. La primera bobina carga, las siguientes suman.
        const bool close = v2_r.o.to_bool() && close2_r.o.to_bool();
        v3_r.i = bit_t(close);
        if (v2_r.o.to_bool()) {
            const bool first = first2_r.o.to_bool();
            for (int a = 0; a < AF; ++a) {
                const std::size_t k = static_cast<std::size_t>(a);
                const word_t pre = rr2_r[k].o - ii2_r[k].o;
                const word_t pim = ri2_r[k].o + ir2_r[k].o;
                p_re_r[k].i = check(first ? pre : p_re_r[k].o + pre, smin(NB_P), smax(NB_P), "P_re", a);
                p_im_r[k].i = check(first ? pim : p_im_r[k].o + pim, smin(NB_P), smax(NB_P), "P_im", a);
            }
        }
        if (close) {
            err3_r.i = err2_r.o;
            d3_r.i   = d2_r.o;
        }

        // Etapa 4: redondeo de s = D - G bits y saturacion a NB_M bits.
        v4_r.i = v3_r.o;
        if (v3_r.o.to_bool()) {
            const int s = static_cast<int>(d3_r.o) - G;
            err4_r.i = err3_r.o;
            for (int a = 0; a < AF; ++a) {
                const std::size_t k = static_cast<std::size_t>(a);
                m_re_r[k].i  = sat(round_shift(p_re_r[k].o, s, ROUND_CONV == 1), NB_M);
                m_im_r[k].i  = sat(round_shift(p_im_r[k].o, s, ROUND_CONV == 1), NB_M);
                pd_re_r[k].i = p_re_r[k].o;
                pd_im_r[k].i = p_im_r[k].o;
            }
        }

        ++cycle_;
    }

private:
    using RegB = rtl::Reg<bit_t>;
    using RegW = rtl::Reg<word_t>;
    using RegWA = std::array<RegW, AF>;

    RegW  cnt_r {};                                      // l de la proxima muestra valida
    // 1: entrada
    RegB  v1_r {}, last1_r {};
    RegW  l1_r {}, d1_r {}, a1_re_r {}, a1_im_r {};
    RegWA u1_re_r {}, u1_im_r {};
    // 2: productos
    RegB  v2_r {}, first2_r {}, close2_r {}, err2_r {};
    RegW  d2_r {};
    RegWA rr2_r {}, ii2_r {}, ri2_r {}, ir2_r {};       // ur*ar, ui*ai, ur*ai, ui*ar
    // 3: acumulador
    RegB  v3_r {}, err3_r {};
    RegW  d3_r {};
    RegWA p_re_r {}, p_im_r {};
    // 4: salida
    RegB  v4_r {}, err4_r {};
    RegWA m_re_r {}, m_im_r {};
    RegWA pd_re_r {}, pd_im_r {};                        // depuracion (no esta en el RTL)
    std::uint64_t cycle_ {0};

    word_t check(word_t v, word_t lo, word_t hi, const char* what, int a) const
    {
        if (v < lo || v > hi) {
            throw std::range_error(
                std::string("unfold: ") + what + (a >= 0 ? "[" + std::to_string(a) + "]" : std::string()) + " = "
                + std::to_string(v) + " fuera de [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"
                + " (ciclo " + std::to_string(cycle_) + ")");
        }
        return v;
    }
};

} // namespace unfold

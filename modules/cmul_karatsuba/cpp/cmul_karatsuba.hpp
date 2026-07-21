#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cstdint>

namespace cmul_karatsuba {

/*
 * --------------------------------------------------------------------------
 * Parameters
 * --------------------------------------------------------------------------
 *
 * These values are equivalent to Verilog parameters.
 * They may be overridden from the build system using compiler defines.
 */

#ifndef CMUL_KARATSUBA_NB_IN
#define CMUL_KARATSUBA_NB_IN 16
#endif

#ifndef CMUL_KARATSUBA_NBF_IN
#define CMUL_KARATSUBA_NBF_IN 14
#endif

#ifndef CMUL_KARATSUBA_NB_OUT
#define CMUL_KARATSUBA_NB_OUT 16
#endif

#ifndef CMUL_KARATSUBA_NBF_OUT
#define CMUL_KARATSUBA_NBF_OUT 14
#endif

enum parameters : int {
    NB_IN   = CMUL_KARATSUBA_NB_IN,
    NBF_IN  = CMUL_KARATSUBA_NBF_IN,
    NB_OUT  = CMUL_KARATSUBA_NB_OUT,
    NBF_OUT = CMUL_KARATSUBA_NBF_OUT
};

/*
 * --------------------------------------------------------------------------
 * Localparameters
 * --------------------------------------------------------------------------
 *
 * Crecimiento de anchos (todo EXACTO, sin cuantizacion intermedia):
 *
 *   sum  = a+b / c+d / d-c      -> NB_IN + 1 bits   (1 bit de guarda)
 *   prod = in_t * sum_t          -> NB_IN + NB_SUM = 2*NB_IN + 1 bits
 *   full = k1-k3 / k1+k2         -> NB_PROD + 1     = 2*NB_IN + 2 bits
 *
 * Notar que full es 1 bit MAS ancho que en la version de 4 multiplicadores
 * (ahi era 2*NB_IN+1). Es solo headroom: el valor matematico es el mismo,
 * porque |ac-bd| y |ad+bc| entran en 2*NB_IN+1 bits. El bit extra viene de
 * que los productos de Gauss (con un operando de NB_IN+1 bits) pueden ser
 * individualmente mas grandes que ac, bd, ad, bc.
 */

enum localparameters : int {
    NBI_IN   = NB_IN  - NBF_IN              ,
    NBI_OUT  = NB_OUT - NBF_OUT             ,

    NB_SUM   = NB_IN + 1                    ,   // a+b, c+d, d-c
    NBF_SUM  = NBF_IN                       ,
    NBI_SUM  = NB_SUM - NBF_SUM             ,   // = NBI_IN + 1

    NB_PROD  = NB_IN + NB_SUM               ,   // = 2*NB_IN + 1
    NBF_PROD = NBF_IN + NBF_SUM             ,   // = 2*NBF_IN
    NBI_PROD = NB_PROD - NBF_PROD           ,   // = 2*NBI_IN + 1

    NB_FULL  = NB_PROD + 1                  ,   // = 2*NB_IN + 2
    NBF_FULL = NBF_PROD                     ,
    NBI_FULL = NB_FULL - NBF_FULL
};

/*
 * --------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------
 *
 * ap_fixed<W, I, Q, O>:
 *   W = ancho total, I = bits enteros (signo incluido)
 *   Q = AP_RND -> round-half-up (empates hacia +inf)
 *   O = AP_SAT -> saturacion en overflow
 *
 * Los modos van EXPLICITOS a proposito. El default de ap_fixed<W,I> es
 * AP_TRN + AP_WRAP (trunca hacia -inf, envuelve en overflow), que NO es lo
 * que hace el RTL (cast con ROUND_MODE=1 -> redondeo + saturacion).
 *
 * sum_t / prod_t / full_t son exactos: nunca se cuantiza en ellos, asi que
 * sus modos no llegan a actuar. Se dejan iguales por consistencia.
 */
#ifndef DOUBLE
    using in_t   = ap_fixed<NB_IN,   NBI_IN,   AP_RND, AP_SAT>;
    using out_t  = ap_fixed<NB_OUT,  NBI_OUT,  AP_RND, AP_SAT>;
    using sum_t  = ap_fixed<NB_SUM,  NBI_SUM,  AP_RND, AP_SAT>;
    using prod_t = ap_fixed<NB_PROD, NBI_PROD, AP_RND, AP_SAT>;
    using full_t = ap_fixed<NB_FULL, NBI_FULL, AP_RND, AP_SAT>;
#else
    using in_t   = double;
    using out_t  = double;
    using sum_t  = double;
    using prod_t = double;
    using full_t = double;
#endif
/*
 * --------------------------------------------------------------------------
 * cmul_karatsuba_model
 * --------------------------------------------------------------------------
 *
 * Multiplicador complejo por el algoritmo de Gauss (3 multiplicaciones en
 * vez de 4, a cambio de 3 sumas previas):
 *
 *   (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
 *
 *   k1 = c * (a + b)
 *   k2 = a * (d - c)
 *   k3 = b * (c + d)
 *
 *   Re = k1 - k3 = (ac + bc) - (bc + bd) = ac - bd
 *   Im = k1 + k2 = (ac + bc) + (ad - ac) = ad + bc
 *
 * Como todos los intermedios son EXACTOS y se cuantiza una sola vez al
 * final, el resultado es bit-identico al del multiplicador de 4
 * multiplicadores (cmul). Verificado sobre vectores aleatorios y de borde.
 *
 * Interfaz identica a cmul_model, asi que es reemplazo directo.
 */

class cmul_karatsuba_model final : public rtl::Module {
public:
    cmul_karatsuba_model() = default;

    /*
     * ----------------------------------------------------------------------
     * Ports
     * ----------------------------------------------------------------------
     */

    rtl::InPort<in_t> i_1_re;   // a
    rtl::InPort<in_t> i_1_im;   // b
    rtl::InPort<in_t> i_2_re;   // c
    rtl::InPort<in_t> i_2_im;   // d

    rtl::OutPort<out_t> o_re;
    rtl::OutPort<out_t> o_im;

    /*
     * ----------------------------------------------------------------------
     * Internal wires
     * ----------------------------------------------------------------------
     */

    rtl::Wire<in_t> w_a;
    rtl::Wire<in_t> w_b;
    rtl::Wire<in_t> w_c;
    rtl::Wire<in_t> w_d;

    // sumas previas (pre-adders)
    rtl::Wire<sum_t> w_a_plus_b;    // a + b
    rtl::Wire<sum_t> w_d_minus_c;   // d - c
    rtl::Wire<sum_t> w_c_plus_d;    // c + d

    // los tres productos
    rtl::Wire<prod_t> w_k1;         // c * (a + b)
    rtl::Wire<prod_t> w_k2;         // a * (d - c)
    rtl::Wire<prod_t> w_k3;         // b * (c + d)

    // combinaciones finales, antes de cuantizar
    rtl::Wire<full_t> w_k1_minus_k3;
    rtl::Wire<full_t> w_k1_plus_k2;

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;

private:
};

} // namespace cmul_karatsuba
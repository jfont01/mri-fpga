#include "cmul_karatsuba.hpp"

namespace cmul_karatsuba {

void cmul_karatsuba_model::connect_clocks(rtl::ClockDomain& clk)
{
    (void)clk;
}

void cmul_karatsuba_model::init()
{
    i_1_re = in_t {0};
    i_1_im = in_t {0};
    i_2_re = in_t {0};
    i_2_im = in_t {0};

    o_re = out_t {0};
    o_im = out_t {0};
}

void cmul_karatsuba_model::combinational()
{
    w_a = i_1_re.value();
    w_b = i_1_im.value();
    w_c = i_2_re.value();
    w_d = i_2_im.value();

    /*
     * Sumas previas, exactas (1 bit de guarda en sum_t).
     * En hardware son los pre-adders del DSP48E2.
     */
    w_a_plus_b  = sum_t(w_a.value()) + sum_t(w_b.value());
    w_d_minus_c = sum_t(w_d.value()) - sum_t(w_c.value());
    w_c_plus_d  = sum_t(w_c.value()) + sum_t(w_d.value());

    /*
     * Los TRES productos (el ahorro frente a los 4 del metodo directo).
     * in_t * sum_t -> ap_fixed<NB_IN + NB_SUM, NBI_IN + NBI_SUM> = prod_t.
     * Exactos: no hay cuantizacion.
     */
    w_k1 = w_c.value() * w_a_plus_b.value();    // c * (a + b) = ac + bc
    w_k2 = w_a.value() * w_d_minus_c.value();   // a * (d - c) = ad - ac
    w_k3 = w_b.value() * w_c_plus_d.value();    // b * (c + d) = bc + bd

    /*
     * Recombinacion, exacta en full_t:
     *   k1 - k3 = ac - bd
     *   k1 + k2 = ad + bc
     */
    w_k1_minus_k3 = full_t(w_k1.value()) - full_t(w_k3.value());
    w_k1_plus_k2  = full_t(w_k1.value()) + full_t(w_k2.value());

    /*
     * UNICA cuantizacion de todo el datapath: full_t -> out_t.
     * out_t es ap_fixed<..., AP_RND, AP_SAT>, o sea round-half-up +
     * saturacion: la misma convencion que el modulo cast (ROUND_MODE=1).
     *
     * Como todos los intermedios fueron exactos, el resultado es
     * bit-identico al del multiplicador de 4 multiplicadores.
     */
    o_re = out_t(w_k1_minus_k3.value());
    o_im = out_t(w_k1_plus_k2.value());
}

void cmul_karatsuba_model::sequential()
{
}

} // namespace cmul_karatsuba
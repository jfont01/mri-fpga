#include "cmul.hpp"

namespace cmul {

void cmul_model::connect_clocks(rtl::ClockDomain& clk)
{
    (void)clk;
}

void cmul_model::init()
{
    i_1_re = in_t {0};
    i_1_im = in_t {0};
    i_2_re = in_t {0};
    i_2_im = in_t {0};

    o_re = out_t {0};
    o_im = out_t {0};
}

void cmul_model::combinational()
{
    w_a = i_1_re.value();
    w_b = i_1_im.value();
    w_c = i_2_re.value();
    w_d = i_2_im.value();


    w_ac = w_a.value() * w_c.value();
    w_bd = w_b.value() * w_d.value();
    w_ad = w_a.value() * w_d.value();
    w_bc = w_b.value() * w_c.value();


    w_ac_minus_bd = full_t(w_ac.value()) - full_t(w_bd.value());
    w_ad_plus_bc  = full_t(w_ad.value()) + full_t(w_bc.value());


    o_re = out_t(w_ac_minus_bd.value());
    o_im = out_t(w_ad_plus_bc.value());
}

void cmul_model::sequential()
{
}

} // namespace cmul
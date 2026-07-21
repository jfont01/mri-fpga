#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cstdint>

namespace cmul {

/*
 * --------------------------------------------------------------------------
 * Parameters
 * --------------------------------------------------------------------------
 *
 * These values are equivalent to Verilog parameters.
 * They may be overridden from the build system using compiler defines.
*/

#ifndef CMUL_NB_IN
#define CMUL_NB_IN 16
#endif

#ifndef CMUL_NBF_IN
#define CMUL_NBF_IN 14
#endif

#ifndef CMUL_NB_OUT
#define CMUL_NB_OUT 16
#endif

#ifndef CMUL_NBF_OUT
#define CMUL_NBF_OUT 14
#endif

 

enum parameters : int {

    NB_IN   = CMUL_NB_IN,
    NBF_IN  = CMUL_NBF_IN,
    NB_OUT  = CMUL_NB_OUT,
    NBF_OUT = CMUL_NBF_OUT

};

/*
 * --------------------------------------------------------------------------
 * Localparameters
 * --------------------------------------------------------------------------
 *
 * These values are equivalent to Verilog localparams.
 * They are derived from parameters and must not be overridden externally.
 */

enum localparameters : int {

    NBI_IN   = NB_IN  - NBF_IN          ,
    NBI_OUT  = NB_OUT - NBF_OUT         ,
    NB_PROD  = NB_IN * 2                ,
    NBF_PROD = NBF_IN * 2               ,
    NBI_PROD = NB_PROD - NBF_PROD       ,
    NB_FULL  = NB_PROD + 1              ,
    NBF_FULL = NBF_PROD                 ,
    NBI_FULL = NB_FULL - NBF_FULL

};

/*
 * --------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------
 *
 * ap_fixed standard convention:
 *   ap_fixed<W, I>
 *   W = total width
 *   I = integer bits, sign included
 *
 * using bit_t = bool;
 */
#ifndef DOUBLE
    using in_t   = ap_fixed<NB_IN,   NBI_IN,   AP_RND, AP_SAT>;
    using out_t  = ap_fixed<NB_OUT,  NBI_OUT,  AP_RND, AP_SAT>;
    using prod_t = ap_fixed<NB_PROD, NBI_PROD, AP_RND, AP_SAT>;
    using full_t = ap_fixed<NB_FULL, NBI_FULL, AP_RND, AP_SAT>;
#else
    using in_t   = double;
    using out_t  = double;
    using prod_t = double;
    using full_t = double;
#endif

class cmul_model final : public rtl::Module {
public:
    cmul_model() = default;

    /*
     * ----------------------------------------------------------------------
     * Ports
     * ----------------------------------------------------------------------
     */

    rtl::InPort<in_t>  i_1_re;
    rtl::InPort<in_t>  i_1_im;
    rtl::InPort<in_t>  i_2_re;
    rtl::InPort<in_t>  i_2_im;

    rtl::OutPort<out_t>  o_re;
    rtl::OutPort<out_t>  o_im;

    /*
     * ----------------------------------------------------------------------
     * Internal wires/registers
     * ----------------------------------------------------------------------
     */

    rtl::Wire<in_t> w_a;
    rtl::Wire<in_t> w_b;
    rtl::Wire<in_t> w_c;
    rtl::Wire<in_t> w_d;

    rtl::Wire<prod_t> w_ac;
    rtl::Wire<prod_t> w_bd;
    rtl::Wire<prod_t> w_ad;
    rtl::Wire<prod_t> w_bc;

    rtl::Wire<full_t> w_ac_minus_bd;
    rtl::Wire<full_t> w_ad_plus_bc;

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;

private:

};

} // namespace cmul

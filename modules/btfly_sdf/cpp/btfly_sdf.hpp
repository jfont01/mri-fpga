#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cstdint>

namespace btfly_sdf {

/*
 * --------------------------------------------------------------------------
 * Parameters
 * --------------------------------------------------------------------------
 *
 * These values are equivalent to Verilog parameters.
 * They may be overridden from the build system using compiler defines.
 *
 * Example:
 *
 * #ifndef BTFLY_SDF_NB_IN
 * #define BTFLY_SDF_NB_IN 16
 * #endif
 *
 * #ifndef BTFLY_SDF_NBF_IN
 * #define BTFLY_SDF_NBF_IN 14
 * #endif
 *
 * #ifndef BTFLY_SDF_NB_OUT
 * #define BTFLY_SDF_NB_OUT 16
 * #endif
 *
 * #ifndef BTFLY_SDF_NBF_OUT
 * #define BTFLY_SDF_NBF_OUT 14
 * #endif
 */

enum parameters : int {
    /*
     * Example:
     *
     * NB_IN   = BTFLY_SDF_NB_IN,
     * NBF_IN  = BTFLY_SDF_NBF_IN,
     * NB_OUT  = BTFLY_SDF_NB_OUT,
     * NBF_OUT = BTFLY_SDF_NBF_OUT
     */
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
    /*
     * Example:
     *
     * NBI_IN  = NB_IN  - NBF_IN,
     * NBI_OUT = NB_OUT - NBF_OUT
     */
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
 * Example:
 *
 * using bit_t = bool;
 * using in_t  = ap_fixed<NB_IN,  NBI_IN>;
 * using out_t = ap_fixed<NB_OUT, NBI_OUT>;
 */

class btfly_sdf_model final : public rtl::Module {
public:
    btfly_sdf_model() = default;

    /*
     * ----------------------------------------------------------------------
     * Ports
     * ----------------------------------------------------------------------
     *
     * Example:
     *
     * rtl::InPort<bit_t> i_en;
     * rtl::InPort<in_t>  i_word;
     *
     * rtl::OutPort<bit_t> o_en;
     * rtl::OutPort<out_t> o_word;
     */

    /*
     * ----------------------------------------------------------------------
     * Internal wires/registers
     * ----------------------------------------------------------------------
     *
     * Example:
     *
     * rtl::Wire<out_t> y_w;
     *
     * rtl::Reg<bit_t> valid_r;
     * rtl::Reg<out_t> y_r;
     */

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;

private:

};

} // namespace btfly_sdf

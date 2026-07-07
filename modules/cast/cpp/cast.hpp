#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>

#include <type_traits>

namespace cast {

#ifndef CAST_NB_IN
#define CAST_NB_IN 8
#endif

#ifndef CAST_NBF_IN
#define CAST_NBF_IN 5
#endif

#ifndef CAST_NB_OUT
#define CAST_NB_OUT 6
#endif

#ifndef CAST_NBF_OUT
#define CAST_NBF_OUT 4
#endif

#ifndef CAST_ROUND_MODE
#define CAST_ROUND_MODE 1
#endif

enum parameters : int {
    NB_IN      = CAST_NB_IN,
    NBF_IN     = CAST_NBF_IN,
    NB_OUT     = CAST_NB_OUT,
    NBF_OUT    = CAST_NBF_OUT,
    ROUND_MODE = CAST_ROUND_MODE
};

enum localparameters : int {
    NBI_IN  = NB_IN  - NBF_IN,
    NBI_OUT = NB_OUT - NBF_OUT,

    LSB_TO_DROP = NBF_IN - NBF_OUT
};

static_assert(NB_IN > 0, "cast: NB_IN must be positive");
static_assert(NB_OUT > 0, "cast: NB_OUT must be positive");

static_assert(NBF_IN >= 0, "cast: NBF_IN must be non-negative");
static_assert(NBF_OUT >= 0, "cast: NBF_OUT must be non-negative");

static_assert(NBF_IN <= NB_IN, "cast: NBF_IN must be <= NB_IN");
static_assert(NBF_OUT <= NB_OUT, "cast: NBF_OUT must be <= NB_OUT");

/*
 * Same restriction as the current RTL:
 *
 *   if (LSB_TO_DROP < 0)
 *       unsupported parameter set: NBF_IN < NBF_OUT
 *
 * If later the RTL supports fractional expansion, this static_assert can be removed.
 */
static_assert(
    LSB_TO_DROP >= 0,
    "cast: unsupported parameter set: NBF_IN < NBF_OUT"
);

/*
 * Input fixed-point type.
 *
 * Default ap_fixed quantization/overflow is enough for the input type because
 * this type represents the already-quantized input word.
 */
using in_t = ap_fixed<NB_IN, NBI_IN>;

/*
 * Output fixed-point type.
 *
 * ROUND_MODE = 0:
 *   truncation + saturation
 *
 * ROUND_MODE = 1:
 *   rounding + saturation
 *
 * The output type itself encodes the quantization and overflow policy.
 */
using out_trunc_t = ap_fixed<NB_OUT, NBI_OUT, AP_TRN, AP_SAT>;
using out_round_t = ap_fixed<NB_OUT, NBI_OUT, AP_RND, AP_SAT>;

using out_t = typename std::conditional<
    (ROUND_MODE != 0),
    out_round_t,
    out_trunc_t
>::type;

class cast_model final : public rtl::Module {
public:
    rtl::InPort<in_t>   i_word;
    rtl::OutPort<out_t> o_word;

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;
};

} // namespace cast
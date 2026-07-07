#include "cast.hpp"

namespace cast {

void cast_model::connect_clocks(rtl::ClockDomain& clk)
{
    /*
     * Combinational module.
     * No registers to connect.
     */
    (void)clk;
}

void cast_model::init()
{
    i_word = in_t{};
    o_word = out_t{};
}

void cast_model::combinational()
{
    /*
     * The ap_fixed output type defines:
     *
     *   - total width
     *   - integer width
     *   - fractional width
     *   - rounding/truncation policy
     *   - saturation policy
     *
     * Therefore the cast is expressed directly as a type conversion.
     */
    o_word = out_t(i_word.value());
}

void cast_model::sequential()
{
    /*
     * Combinational module.
     * No sequential behavior.
     */
}

} // namespace cast
#include "btfly_sdf.hpp"

namespace btfly_sdf {

void btfly_sdf_model::connect_clocks(rtl::ClockDomain& clk)
{
    (void)clk;

    /*
     * Add sequential state/registers to the clock domain.
     *
     * Example:
     *
     * clk.add(valid_r);
     * clk.add(y_r);
     */
}

void btfly_sdf_model::init()
{
    /*
     * Initialize ports, wires and registers.
     *
     * Example:
     *
     * i_en   = false;
     * i_word = in_t {};
     *
     * o_en   = false;
     * o_word = out_t {};
     *
     * y_w = out_t {};
     *
     * valid_r.set_initial_value(false);
     * y_r.set_initial_value(out_t {});
     */
}

void btfly_sdf_model::combinational()
{
    /*
     * Describe combinational behavior.
     *
     * Example:
     *
     * y_w = out_t(i_word.value());
     *
     * o_en   = valid_r.o;
     * o_word = y_r.o;
     */
}

void btfly_sdf_model::sequential()
{
    /*
     * Describe clocked behavior.
     *
     * Important:
     * - Assign next register values using reg.i
     * - Read current register values using reg.o
     *
     * Example:
     *
     * valid_r.i = i_en.value();
     *
     * if (i_en.value()) {
     *     y_r.i = y_w.value();
     * }
     * else {
     *     y_r.i = y_r.o;
     * }
     */
}

} // namespace btfly_sdf

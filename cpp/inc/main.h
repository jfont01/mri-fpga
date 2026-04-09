#ifndef _MAIN_H
#define _MAIN_H

#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include "ap_fixed.h"
#include "config.h"

template<int W, int I>
using fxp_t = ap_fixed<W, I, AP_RND_CONV, AP_SAT>;

template<int W, int I>
struct cfix_t {
    fxp_t<W, I> re;
    fxp_t<W, I> im;
};

using S_t      = fxp_t<NB_S, NBI_S>;
using y_t      = fxp_t<NB_Y, NBI_Y>;
using A_t      = fxp_t<NB_A, NBI_A>;

using cfix_S_t = cfix_t<NB_S, NBI_S>;
using cfix_y_t = cfix_t<NB_Y, NBI_Y>;
using cfix_A_t = cfix_t<NB_A, NBI_A>;

struct ComplexTensor {
    std::vector<size_t> shape;
    std::vector<std::complex<double>> data;
};

#endif
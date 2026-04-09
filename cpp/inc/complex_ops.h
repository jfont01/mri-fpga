#ifndef _COPS_H
#define _COPS_H

#include <array>
#include <stdexcept>
#include "main.h"

template<int W, int I>
using cmat2x2_t = std::array<std::array<cfix_t<W,I>, 2>, 2>;

inline size_t idx3(
    size_t l,
    size_t nx,
    size_t ny,
    size_t Nx,
    size_t Ny
) {
    return (l * Nx + nx) * Ny + ny;
}

template<int W, int I>
inline cfix_t<W,I> czero() {
    cfix_t<W,I> z;
    z.re = 0;
    z.im = 0;
    return z;
}

template<int W, int I>
inline cfix_t<W,I> cconj(const cfix_t<W,I>& a) {
    cfix_t<W,I> z;
    z.re = a.re;
    z.im = -a.im;
    return z;
}

template<int W, int I>
inline cfix_t<W,I> cadd(const cfix_t<W,I>& a, const cfix_t<W,I>& b) {
    cfix_t<W,I> z;
    z.re = a.re + b.re;
    z.im = a.im + b.im;
    return z;
}

template<int W, int I>
inline cfix_t<W,I> cmul(const cfix_t<W,I>& a, const cfix_t<W,I>& b) {
    cfix_t<W,I> z;
    z.re = a.re * b.re - a.im * b.im;
    z.im = a.re * b.im + a.im * b.re;
    return z;
}

template<int W, int I>
inline cfix_t<W,I> real_to_cfix(ap_fixed<W,I> x) {
    cfix_t<W,I> z;
    z.re = x;
    z.im = 0;
    return z;
}

#endif
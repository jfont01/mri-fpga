#ifndef _AIJ_H
#define _AIJ_H

#include <array>
#include <stdexcept>
#include "main.h"
#include "complex_ops.h"

template<int WS, int IS, int WA, int IA>
cmat2x2_t<WA,IA> compute_Aij(
    const std::vector<cfix_t<WS,IS>>& S_q,
    const std::vector<size_t>& shape,
    size_t nx,
    size_t ny_alias
) {
    if (shape.size() != 3) {
        throw std::runtime_error("compute_Aij: se esperaba shape (L,Nx,Ny)");
    }

    const size_t L  = shape[0];
    const size_t Nx = shape[1];
    const size_t Ny = shape[2];

    if (nx >= Nx) {
        throw std::runtime_error("compute_Aij: nx fuera de rango");
    }

    const size_t Af = 2;
    const size_t offset = Ny / Af;

    if (ny_alias >= offset) {
        throw std::runtime_error("compute_Aij: ny_alias fuera de rango");
    }

    const size_t ny0 = ny_alias;
    const size_t ny1 = ny_alias + offset;

    cfix_A_t A00 = czero<WA,IA>();
    cfix_A_t A11 = czero<WA,IA>();
    cfix_A_t A01 = czero<WA,IA>();

    for (size_t l = 0; l < L; l++) {
        const auto& s0 = S_q[idx3(l, nx, ny0, Nx, Ny)];
        const auto& s1 = S_q[idx3(l, nx, ny1, Nx, Ny)];

        // p00 = |s0|^2
        A_t p00_re = s0.re * s0.re + s0.im * s0.im;
        // p11 = |s1|^2
        A_t p11_re = s1.re * s1.re + s1.im * s1.im;

        cfix_A_t p00 = real_to_cfix<WA,IA>(p00_re);
        cfix_A_t p11 = real_to_cfix<WA,IA>(p11_re);

        // p01 = conj(s0) * s1
        cfix_A_t s0w, s1w;
        s0w.re = s0.re;
        s0w.im = s0.im;
        s1w.re = s1.re;
        s1w.im = s1.im;

        cfix_A_t p01 = cmul<WA,IA>(cconj<WA,IA>(s0w), s1w);

        A00 = cadd<WA,IA>(A00, p00);
        A11 = cadd<WA,IA>(A11, p11);
        A01 = cadd<WA,IA>(A01, p01);
    }

    cmat2x2_t<WA,IA> Aij;
    Aij[0][0] = A00;
    Aij[0][1] = A01;
    Aij[1][0] = cconj<WA,IA>(A01);
    Aij[1][1] = A11;

    return Aij;
}

template<int W, int I>
void print_Aij(const cmat2x2_t<W,I>& Aij) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << "Aij[" << i << "][" << j << "] = "
                      << Aij[i][j].re
                      << " + j"
                      << Aij[i][j].im
                      << "\n";
        }
    }
}



#endif
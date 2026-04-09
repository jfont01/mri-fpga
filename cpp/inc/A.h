#ifndef _A_H
#define _A_H

#include <vector>
#include <stdexcept>
#include "main.h"
#include "Aij.h"
#include "complex_ops.h"

inline size_t idx4(
    size_t a,
    size_t b,
    size_t nx,
    size_t ny_alias,
    size_t B,
    size_t Nx,
    size_t Offset
) {
    return ((a * B + b) * Nx + nx) * Offset + ny_alias;
}

template<int WS, int IS, int WA, int IA>
std::vector<cfix_t<WA,IA>> compute_A(
    const std::vector<cfix_t<WS,IS>>& S_q,
    const std::vector<size_t>& S_shape,
    std::vector<size_t>& A_shape
) {
    if (S_shape.size() != 3) {
        throw std::runtime_error("compute_A: se esperaba shape (L,Nx,Ny)");
    }

    const size_t L  = S_shape[0];
    const size_t Nx = S_shape[1];
    const size_t Ny = S_shape[2];

    if (S_q.size() != L * Nx * Ny) {
        throw std::runtime_error("compute_A: size de S_q inconsistente con S_shape");
    }

    const size_t Af = 2;
    if ((Ny % Af) != 0) {
        throw std::runtime_error("compute_A: Ny debe ser divisible por Af=2");
    }

    const size_t offset = Ny / Af;

    A_shape = {2, 2, Nx, offset};

    std::vector<cfix_t<WA,IA>> A_q(2 * 2 * Nx * offset, czero<WA,IA>());

    for (size_t nx = 0; nx < Nx; nx++) {
        for (size_t ny_alias = 0; ny_alias < offset; ny_alias++) {
            cmat2x2_t<WA,IA> Aij =
                compute_Aij<WS,IS,WA,IA>(S_q, S_shape, nx, ny_alias);

            A_q[idx4(0, 0, nx, ny_alias, 2, Nx, offset)] = Aij[0][0];
            A_q[idx4(0, 1, nx, ny_alias, 2, Nx, offset)] = Aij[0][1];
            A_q[idx4(1, 0, nx, ny_alias, 2, Nx, offset)] = Aij[1][0];
            A_q[idx4(1, 1, nx, ny_alias, 2, Nx, offset)] = Aij[1][1];
        }
    }

    return A_q;
}

template<int W, int I>
void print_A_tensor_info(
    const std::vector<cfix_t<W,I>>& A_q,
    const std::vector<size_t>& A_shape
) {
    if (A_shape.size() != 4) {
        throw std::runtime_error("print_A_tensor_info: se esperaba shape (2,2,Nx,offset)");
    }

    std::cout << "A shape = ("
              << A_shape[0] << ", "
              << A_shape[1] << ", "
              << A_shape[2] << ", "
              << A_shape[3] << ")\n";

    std::cout << "A total elems = " << A_q.size() << "\n";
}

#endif
#ifndef _HELPERS_H
#define _HELPERS_H

#include <fstream>
#include <string>
#include <stdexcept>
#include "Aij.h"
#include "ap_int.h"

template<int W, int I>
void save_Aij_hex_dat(
    const cmat2x2_t<W,I>& Aij,
    const std::string& out_path
) {
    std::ofstream fout(out_path);
    if (!fout) {
        throw std::runtime_error("No se pudo abrir " + out_path);
    }

    fout << "# shape=(2, 2)\n";
    fout << "# NB=" << W << "\n";
    fout << "# NBF=" << (W - I) << "\n";
    fout << "# format: a b re_hex im_hex\n";

    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            ap_uint<W> re_raw = Aij[a][b].re.range(W-1, 0);
            ap_uint<W> im_raw = Aij[a][b].im.range(W-1, 0);

            fout << a << " " << b << " "
                 << re_raw.to_string(16) << " "
                 << im_raw.to_string(16) << "\n";
        }
    }
}

template<int W, int I>
inline size_t idx4_A(
    size_t a,
    size_t b,
    size_t i,
    size_t j,
    size_t B,
    size_t Nx,
    size_t Offset
) {
    return ((a * B + b) * Nx + i) * Offset + j;
}

template<int W, int I>
void save_A_hex_dat(
    const std::vector<cfix_t<W,I>>& A_q,
    const std::vector<size_t>& A_shape,
    const std::string& out_path
) {
    if (A_shape.size() != 4) {
        throw std::runtime_error("save_A_hex_dat: se esperaba shape (2,2,Nx,offset)");
    }

    const size_t A0 = A_shape[0];
    const size_t A1 = A_shape[1];
    const size_t Nx = A_shape[2];
    const size_t Offset = A_shape[3];

    if (A0 != 2 || A1 != 2) {
        throw std::runtime_error("save_A_hex_dat: se esperaba shape (2,2,Nx,offset)");
    }

    const size_t expected = A0 * A1 * Nx * Offset;
    if (A_q.size() != expected) {
        throw std::runtime_error("save_A_hex_dat: size de A_q inconsistente con A_shape");
    }

    std::ofstream fout(out_path);
    if (!fout) {
        throw std::runtime_error("No se pudo abrir " + out_path);
    }

    for (size_t a = 0; a < A0; a++) {
        for (size_t b = 0; b < A1; b++) {
            for (size_t i = 0; i < Nx; i++) {
                for (size_t j = 0; j < Offset; j++) {
                    const auto& x = A_q[idx4_A<W,I>(a, b, i, j, A1, Nx, Offset)];

                    ap_uint<W> re_raw = x.re.range(W-1, 0);
                    ap_uint<W> im_raw = x.im.range(W-1, 0);

                    fout << a << " " << b << " " << i << " " << j << " "
                         << re_raw.to_string(16) << " "
                         << im_raw.to_string(16) << "\n";
                }
            }
        }
    }
}

#endif
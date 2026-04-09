#ifndef _LOAD_STIM_H
#define _LOAD_STIM_H

#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include "ap_fixed.h"
#include "main.h"

//#define PRINT_ALL

using namespace std;


inline void unravel_idx_3d(
    size_t idx,
    size_t Nx,
    size_t Ny,
    size_t& l,
    size_t& nx,
    size_t& ny
) {
    ny = idx % Ny;
    idx /= Ny;
    nx = idx % Nx;
    idx /= Nx;
    l = idx;
}

template<typename T>
inline void print_shape(const vector<T>& shape) {
    cout << "shape = (";
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i + 1 < shape.size()) cout << ", ";
    }
    cout << ")\n";
}

size_t numel_from_shape(const vector<size_t>& shape);

ComplexTensor load_complex_npy(const string& path);

template<int W, int I>
vector<cfix_t<W, I>> quantize_complex_tensor(
    const ComplexTensor& X
) {
    vector<cfix_t<W, I>> out(X.data.size());

    for (size_t k = 0; k < X.data.size(); k++) {
        out[k].re = fxp_t<W, I>(X.data[k].real());
        out[k].im = fxp_t<W, I>(X.data[k].imag());
    }

    return out;
}

template<int W, int I>
void print_values(
    const vector<cfix_t<W, I>>& X,
    const vector<size_t>& shape
) {
    if (shape.size() != 3) {
        cerr << "print_values: se esperaba shape 3D (L, Nx, Ny)\n";
        return;
    }

    size_t L  = shape[0];
    size_t Nx = shape[1];
    size_t Ny = shape[2];

    size_t expected = L * Nx * Ny;
    if (X.size() != expected) {
        cerr << "print_values: size inconsistente con shape\n";
        return;
    }

    for (size_t i = 0; i < X.size(); i++) {
        size_t l, nx, ny;
        unravel_idx_3d(i, Nx, Ny, l, nx, ny);

        cout << "X[" << l << "," << nx << "," << ny << "] = " 
             << X[i].re.to_double()
             << " + j"
             << X[i].im.to_double()
             << "\n";
    }
}

#endif
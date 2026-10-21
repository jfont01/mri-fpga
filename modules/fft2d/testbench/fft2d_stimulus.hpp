#pragma once

// -----------------------------------------------------------------------------
// fft2d_stimulus.hpp -- k-space de entrada de la fft2d, comun al testbench de
// sim (fft2d_tb.cpp) y al gtest (fft2d_gtest.cpp): con el mismo caso del JSON
// los dos ven exactamente el mismo estimulo.
//
// Resultado: ks[l][2 * (k' * N1 + r) + {0, 1}] = (re, im) de la bobina l, ya
// en el formato de entrada (in_t), en el orden en que entra al hardware (fila
// k', readout r contiguo).
//
//   FFT2D_N_COILS        bobinas por corrida                         (3)
//   FFT2D_STIM_TYPE      0 ruido uniforme +-A                        (default)
//                        1 impulso en k-space: A en (k0 + l, r0)
//                        2 tono: A e^{-j2pi(r x0 / N1 + k' p0 / N2)} -> la
//                          IDFT es un delta en (p0 + l, x0) de valor A*T*2^-S
//                        3 archivo (FFT2D_STIM_FILE)
//   FFT2D_STIM_AMP_MILLI A en milesimas del fondo de escala         (900)
//   FFT2D_STIM_SEED      semilla del ruido                           (1)
//   FFT2D_STIM_K0, FFT2D_STIM_R0   posicion del impulso              (5, 3)
//   FFT2D_STIM_P0, FFT2D_STIM_X0   posicion del delta del tono        (7, 20)
//   FFT2D_STIM_FILE      ruta relativa a la carpeta del modulo, int32
//                        little-endian [L][N2][N1][re, im], enteros crudos
//                        del formato de entrada (los escribe
//                        model/gen_fft2d_stim.py)
//
// En el impulso y el tono la posicion se corre una fila por bobina, para que
// una bobina cambiada de lugar se vea.
// -----------------------------------------------------------------------------

#include "fft2d.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FFT2D_N_COILS
#define FFT2D_N_COILS 3              // bobinas por corrida (tb y gtest)
#endif
#ifndef FFT2D_STIM_TYPE
#define FFT2D_STIM_TYPE 0
#endif
#ifndef FFT2D_STIM_AMP_MILLI
#define FFT2D_STIM_AMP_MILLI 900
#endif
#ifndef FFT2D_STIM_SEED
#define FFT2D_STIM_SEED 1
#endif
#ifndef FFT2D_STIM_K0
#define FFT2D_STIM_K0 5
#endif
#ifndef FFT2D_STIM_R0
#define FFT2D_STIM_R0 3
#endif
#ifndef FFT2D_STIM_P0
#define FFT2D_STIM_P0 7
#endif
#ifndef FFT2D_STIM_X0
#define FFT2D_STIM_X0 20
#endif
#ifndef FFT2D_STIM_FILE
#define FFT2D_STIM_FILE testbench/stim/kspace.bin
#endif

namespace fft2d_stimulus {

using fft2d::in_t;
using KSpace = std::vector<std::vector<in_t>>;

inline const char* type_name()
{
    switch (FFT2D_STIM_TYPE) {
    case 0:  return "ruido";
    case 1:  return "impulso";
    case 2:  return "tono";
    case 3:  return "archivo";
    default: return "?";
    }
}

inline std::string file_path()
{
    return FFT1D_R22SDF_STRINGIFY(FFT2D_STIM_FILE);
}

inline KSpace generate(const fft2d::Params& p, int n_coils)
{
    const int    n1 = p.n1(), n2 = p.n2();
    const long   t  = p.frame();
    const double a  = FFT2D_STIM_AMP_MILLI / 1000.0;
    constexpr double PI = 3.14159265358979323846;

    KSpace ks(n_coils, std::vector<in_t>(2 * t, in_t(0)));

    switch (FFT2D_STIM_TYPE) {
    case 0: {
        std::mt19937_64 rng(FFT2D_STIM_SEED);
        std::uniform_real_distribution<double> u(-a, a);
        for (auto& coil : ks) {
            for (auto& v : coil) v = in_t(u(rng));      // re, im intercalados
        }
        break;
    }
    case 1:
        for (int l = 0; l < n_coils; ++l) {
            const long i = static_cast<long>((FFT2D_STIM_K0 + l) % n2) * n1 + (FFT2D_STIM_R0 % n1);
            ks[l][2 * i] = in_t(a);
        }
        break;
    case 2:
        for (int l = 0; l < n_coils; ++l) {
            const long p0 = (FFT2D_STIM_P0 + l) % n2;
            const long x0 = FFT2D_STIM_X0 % n1;
            for (int k = 0; k < n2; ++k) {
                for (int r = 0; r < n1; ++r) {
                    // fase en vueltas, con los productos modulo N (exacto)
                    const double turns = static_cast<double>((r * x0) % n1) / n1
                                       + static_cast<double>((k * p0) % n2) / n2;
                    const long i = static_cast<long>(k) * n1 + r;
                    ks[l][2 * i]     = in_t(a * std::cos(-2.0 * PI * turns));
                    ks[l][2 * i + 1] = in_t(a * std::sin(-2.0 * PI * turns));
                }
            }
        }
        break;
    case 3: {
#ifdef DOUBLE
        // los enteros crudos se interpretan en el formato de entrada de fixed
        const double lsb = std::ldexp(1.0, -fft1d_r22sdf::NBF_IN);
#endif
        const std::string path = file_path();
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("fft2d_stimulus: no se puede abrir " + path +
                                     " (generarlo con model/gen_fft2d_stim.py)");
        }
        std::vector<std::int32_t> raw(static_cast<size_t>(2 * t * n_coils));
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 4));
        if (f.gcount() != static_cast<std::streamsize>(raw.size() * 4)) {
            throw std::runtime_error("fft2d_stimulus: " + path + " no tiene N_COILS * N1 * N2 complejos");
        }
        for (int l = 0; l < n_coils; ++l) {
            for (long i = 0; i < 2 * t; ++i) {
                const std::int32_t r = raw[static_cast<size_t>(l) * 2 * t + i];
#ifdef DOUBLE
                ks[l][i] = in_t(r * lsb);
#else
                in_t v;
                v.range(fft1d_r22sdf::NB_IN - 1, 0) =
                    ap_uint<fft1d_r22sdf::NB_IN>(static_cast<std::uint32_t>(r));
                ks[l][i] = v;
#endif
            }
        }
        break;
    }
    default:
        throw std::runtime_error("fft2d_stimulus: FFT2D_STIM_TYPE desconocido");
    }
    return ks;
}

} // namespace fft2d_stimulus

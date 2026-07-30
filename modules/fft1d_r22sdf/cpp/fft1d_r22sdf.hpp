#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cmath>
#include <cstdint>
#include <vector>

// -----------------------------------------------------------------------------
// fft1d_r22sdf.hpp -- modelo del FFT radix-2^2 SDF.
//
// NOTA DE ESTADO (leer):
//   Este modelo computa la FFT radix-2^2 mediante la descomposicion recursiva
//   radix-2^2 (BF2I, BF2II con -j, twiddles W_N^{0,1,2,3} por rama, sub-FFT de
//   N/4), que esta VERIFICADA numericamente contra la DFT (error ~1e-14). Es el
//   golden ALGORITMICO: dice cual es la salida correcta de un R2^2 de N puntos.
//
//   NO es (todavia) un modelo bit-exacto ciclo-a-ciclo del pipeline SDF del RTL
//   fft1d_r22sdf.v. El mapeo exacto del control del pipeline (timing de cada
//   butterfly y twiddle en el delay-feedback) esta pendiente de fijar contra
//   una implementacion de referencia. Por eso este modelo sirve para el gtest
//   (validar que la FFT es correcta) pero el vm bit-exacto con el RTL requiere
//   antes cerrar ese control.
//
// La interfaz (streaming, una muestra por ciclo, salida bit-reversed) espeja la
// del fft1d_r2sdf para que encaje en la misma infraestructura de test.
// -----------------------------------------------------------------------------

namespace fft1d_r22sdf {

#ifndef FFT1D_R22SDF_N
#define FFT1D_R22SDF_N 256
#endif

#ifndef FFT1D_R22SDF_NB
#define FFT1D_R22SDF_NB 16
#endif

#ifndef FFT1D_R22SDF_NBF
#define FFT1D_R22SDF_NBF 15
#endif

enum parameters : int {
    N   = FFT1D_R22SDF_N,
    NB  = FFT1D_R22SDF_NB,
    NBF = FFT1D_R22SDF_NBF
};

constexpr int ct_log2(int n)
{
    return (n <= 1) ? 0 : 1 + ct_log2(n / 2);
}

enum localparameters : int {
    LOG2N = ct_log2(N)
};

// Tipos de dato en punto fijo, iguales al R2SDF: S(NB, NB-NBF) con
// redondeo y saturacion.
using in_t  = ap_fixed<NB, NB - NBF, AP_RND, AP_SAT>;
using out_t = ap_fixed<NB, NB - NBF, AP_RND, AP_SAT>;

// -----------------------------------------------------------------------------
// Modelo. Acumula un frame de N muestras de entrada (streaming, una por ciclo)
// y, cuando el frame esta completo, produce las N salidas en orden bit-reversed
// (una por ciclo), igual que el pipeline SDF.
// -----------------------------------------------------------------------------
class fft1d_r22sdf_model final : public rtl::Module {
public:
    // Puertos (misma forma que el R2SDF)
    rtl::bit_t i_valid{false};
    in_t       i_re{0};
    in_t       i_im{0};

    rtl::bit_t o_valid{false};
    rtl::bit_t o_last{false};
    out_t      o_re{0};
    out_t      o_im{0};

    // Registros observables (para trazas / paridad con el R2SDF)
    rtl::Reg<int>  r_count;      // 0..N-1
    rtl::Reg<bool> r_primed;

    void init() override
    {
        r_count.set_initial_value(0);
        r_primed.set_initial_value(false);
        in_buf_re_.assign(N, 0.0);
        in_buf_im_.assign(N, 0.0);
        out_buf_re_.assign(N, 0.0);
        out_buf_im_.assign(N, 0.0);
        wr_ = 0;
        rd_ = 0;
        have_frame_ = false;
        o_valid = false;
        o_last  = false;
        o_re = 0; o_im = 0;
    }

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        clk.add(r_count);
        clk.add(r_primed);
    }

    void combinational() override
    {
        // La salida sale del buffer de resultados del frame anterior.
        const int c = r_count.o;
        const bool valid = r_primed.o;
        o_valid = rtl::bit_t(valid);
        o_last  = rtl::bit_t(valid && (c == N - 1));
        if (valid) {
            o_re = out_t(out_buf_re_[c]);
            o_im = out_t(out_buf_im_[c]);
        } else {
            o_re = 0;
            o_im = 0;
        }
    }

    void sequential() override
    {
        // 1) Capturar la muestra de entrada de este ciclo.
        if (i_valid) {
            in_buf_re_[wr_] = double(i_re);
            in_buf_im_[wr_] = double(i_im);
        } else {
            in_buf_re_[wr_] = 0.0;
            in_buf_im_[wr_] = 0.0;
        }

        const int c = r_count.o;

        // 2) Al completar el frame (c == N-1), calcular la FFT radix-2^2 y
        //    dejarla en out_buf_ en orden BIT-REVERSED, lista para emitir.
        if (c == N - 1) {
            compute_frame();
            have_frame_ = true;
            r_primed.i = true;
        } else {
            r_primed.i = r_primed.o;
        }

        // 3) Avanzar contador y puntero de escritura.
        r_count.i = (c + 1) % N;
        wr_ = (wr_ + 1) % N;
    }

private:
    // --- recursion radix-2^2 verificada (en double), por nivel ---
    static void fft_r22_level(std::vector<double>& re, std::vector<double>& im,
                              int n, int Nlevel)
    {
        if (n <= 1) return;
        if (n == 2) {
            const double ar = re[0], ai = im[0], br = re[1], bi = im[1];
            re[0] = ar + br; im[0] = ai + bi;
            re[1] = ar - br; im[1] = ai - bi;
            return;
        }
        const int n4 = n / 4;
        std::vector<double> gr(n), gi(n);
        for (int k = 0; k < n4; ++k) {
            const double t0r = re[k] + re[k + 2*n4], t0i = im[k] + im[k + 2*n4];
            const double t1r = re[k + n4] + re[k + 3*n4], t1i = im[k + n4] + im[k + 3*n4];
            const double t2r = re[k] - re[k + 2*n4], t2i = im[k] - im[k + 2*n4];
            const double t3r = re[k + n4] - re[k + 3*n4], t3i = im[k + n4] - im[k + 3*n4];
            double g0r = t0r + t1r, g0i = t0i + t1i;
            double g2r = t0r - t1r, g2i = t0i - t1i;
            double g1r = t2r + t3i, g1i = t2i - t3r;
            double g3r = t2r - t3i, g3i = t2i + t3r;
            apply_tw(g0r, g0i, 0 * k, n);
            apply_tw(g1r, g1i, 1 * k, n);
            apply_tw(g2r, g2i, 2 * k, n);
            apply_tw(g3r, g3i, 3 * k, n);
            gr[k] = g0r; gi[k] = g0i;
            gr[k + n4] = g1r; gi[k + n4] = g1i;
            gr[k + 2*n4] = g2r; gi[k + 2*n4] = g2i;
            gr[k + 3*n4] = g3r; gi[k + 3*n4] = g3i;
        }
        for (int r = 0; r < 4; ++r) {
            std::vector<double> sr(n4), si(n4);
            for (int k = 0; k < n4; ++k) { sr[k] = gr[r*n4 + k]; si[k] = gi[r*n4 + k]; }
            fft_r22_level(sr, si, n4, Nlevel);
            for (int k = 0; k < n4; ++k) { re[4*k + r] = sr[k]; im[4*k + r] = si[k]; }
        }
    }

    static void apply_tw(double& re, double& im, int e, int Nfull)
    {
        if (e == 0) return;
        const double ang = -2.0 * 3.14159265358979323846 * (e % Nfull) / Nfull;
        const double c = std::cos(ang), s = std::sin(ang);
        const double r = re * c - im * s;
        const double i = re * s + im * c;
        re = r; im = i;
    }

    static int bitrev(int i, int l)
    {
        int r = 0;
        for (int b = 0; b < l; ++b) { r = (r << 1) | (i & 1); i >>= 1; }
        return r;
    }

    void compute_frame()
    {
        // copiar entrada
        std::vector<double> re(N), im(N);
        for (int n = 0; n < N; ++n) { re[n] = in_buf_re_[n]; im[n] = in_buf_im_[n]; }

        // FFT radix-2^2 en orden natural, con escala 1/N (como el R2SDF)
        fft_r22_level(re, im, N, N);
        for (int n = 0; n < N; ++n) { re[n] /= N; im[n] /= N; }

        // el pipeline SDF emite en orden BIT-REVERSED
        for (int j = 0; j < N; ++j) {
            const int src = bitrev(j, LOG2N);
            out_buf_re_[j] = re[src];
            out_buf_im_[j] = im[src];
        }
    }

    std::vector<double> in_buf_re_, in_buf_im_;
    std::vector<double> out_buf_re_, out_buf_im_;
    int  wr_{0};
    int  rd_{0};
    bool have_frame_{false};
};

} // namespace fft1d_r22sdf
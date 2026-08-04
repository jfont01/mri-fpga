#pragma once

// -----------------------------------------------------------------------------
// fft1d_stimulus.hpp
//
// Generador de estimulos configurable por defines, compartido por los
// testbenches de FFT (r2sdf, y a futuro r22sdf). Produce un frame completo de
// N muestras complejas que se precalcula una vez y luego se entrega muestra a
// muestra al DUT.
//
// TIPOS DE SENAL (define FFT1D_STIM_TYPE):
//   0 = MULTITONE  suma de hasta 8 senoidales complejas
//   1 = CHIRP      barrido lineal de frecuencia k0 -> k1
//   2 = IMPULSE    delta en una posicion
//   3 = NOISE      ruido pseudo-aleatorio con semilla fija (reproducible)
//
// CONVENCION DE PARAMETROS ENTEROS:
//   Los defines son enteros (asi viajan por -d sin comillas ni floats). Las
//   AMPLITUDES se pasan x1000: FFT1D_STIM_A0=300 -> 0.300. Las FASES en grados
//   enteros: FFT1D_STIM_PH0=90 -> 90 grados.
//
// NORMALIZACION:
//   Tras generar la senal se escala para que max(|re|,|im|) <= TARGET_PEAK
//   (0.9375, potencia de 2 exacta, con margen de redondeo bajo el maximo de
//   Q1.15). Evita saturacion en multitono/chirp, que corromperia la comparacion
//   con la FFT ideal. Solo escala amplitud: NO mueve frecuencias, el espectro
//   conserva sus picos. Para desactivarla definir FFT1D_STIM_NORMALIZE=0.
//
// El .sv NO usa este header: reinyecta los .dat que produce el .cpp. La
// normalizacion ocurre una sola vez, del lado C++, asi que ambos lados ven
// exactamente la misma senal.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fft1d_stimulus {

#ifndef FFT1D_STIM_TYPE
#define FFT1D_STIM_TYPE 0
#endif

enum stim_type : int {
    STIM_MULTITONE = 0,
    STIM_CHIRP     = 1,
    STIM_IMPULSE   = 2,
    STIM_NOISE     = 3
};

#ifndef FFT1D_STIM_NTONES
#define FFT1D_STIM_NTONES 1
#endif

#ifndef FFT1D_STIM_K0
#define FFT1D_STIM_K0 3
#endif
#ifndef FFT1D_STIM_A0
#define FFT1D_STIM_A0 500
#endif
#ifndef FFT1D_STIM_PH0
#define FFT1D_STIM_PH0 0
#endif
#ifndef FFT1D_STIM_K1
#define FFT1D_STIM_K1 0
#endif
#ifndef FFT1D_STIM_A1
#define FFT1D_STIM_A1 0
#endif
#ifndef FFT1D_STIM_PH1
#define FFT1D_STIM_PH1 0
#endif
#ifndef FFT1D_STIM_K2
#define FFT1D_STIM_K2 0
#endif
#ifndef FFT1D_STIM_A2
#define FFT1D_STIM_A2 0
#endif
#ifndef FFT1D_STIM_PH2
#define FFT1D_STIM_PH2 0
#endif
#ifndef FFT1D_STIM_K3
#define FFT1D_STIM_K3 0
#endif
#ifndef FFT1D_STIM_A3
#define FFT1D_STIM_A3 0
#endif
#ifndef FFT1D_STIM_PH3
#define FFT1D_STIM_PH3 0
#endif
#ifndef FFT1D_STIM_K4
#define FFT1D_STIM_K4 0
#endif
#ifndef FFT1D_STIM_A4
#define FFT1D_STIM_A4 0
#endif
#ifndef FFT1D_STIM_PH4
#define FFT1D_STIM_PH4 0
#endif
#ifndef FFT1D_STIM_K5
#define FFT1D_STIM_K5 0
#endif
#ifndef FFT1D_STIM_A5
#define FFT1D_STIM_A5 0
#endif
#ifndef FFT1D_STIM_PH5
#define FFT1D_STIM_PH5 0
#endif
#ifndef FFT1D_STIM_K6
#define FFT1D_STIM_K6 0
#endif
#ifndef FFT1D_STIM_A6
#define FFT1D_STIM_A6 0
#endif
#ifndef FFT1D_STIM_PH6
#define FFT1D_STIM_PH6 0
#endif
#ifndef FFT1D_STIM_K7
#define FFT1D_STIM_K7 0
#endif
#ifndef FFT1D_STIM_A7
#define FFT1D_STIM_A7 0
#endif
#ifndef FFT1D_STIM_PH7
#define FFT1D_STIM_PH7 0
#endif

#ifndef FFT1D_STIM_CHIRP_K0
#define FFT1D_STIM_CHIRP_K0 2
#endif
#ifndef FFT1D_STIM_CHIRP_K1
#define FFT1D_STIM_CHIRP_K1 20
#endif
#ifndef FFT1D_STIM_CHIRP_AMP
#define FFT1D_STIM_CHIRP_AMP 400
#endif

#ifndef FFT1D_STIM_IMP_POS
#define FFT1D_STIM_IMP_POS 0
#endif
#ifndef FFT1D_STIM_IMP_AMP
#define FFT1D_STIM_IMP_AMP 900
#endif

#ifndef FFT1D_STIM_NOISE_SEED
#define FFT1D_STIM_NOISE_SEED 1
#endif
#ifndef FFT1D_STIM_NOISE_AMP
#define FFT1D_STIM_NOISE_AMP 300
#endif

#ifndef FFT1D_STIM_NORMALIZE
#define FFT1D_STIM_NORMALIZE 1
#endif

namespace detail {

constexpr double PI = 3.14159265358979323846;
constexpr double TARGET_PEAK = 0.9375;   // 30720/32768, potencia de 2

inline double amp_from_milli(int milli) { return static_cast<double>(milli) / 1000.0; }
inline double deg_to_rad(int deg)       { return static_cast<double>(deg) * PI / 180.0; }

// PRNG reproducible e independiente de la STL: xorshift32. Devuelve [-1, 1).
struct xorshift32 {
    std::uint32_t s;
    explicit xorshift32(std::uint32_t seed) : s(seed ? seed : 0xDEADBEEFu) {}
    std::uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    double uniform() {
        const double u = (next() >> 8) / static_cast<double>(1u << 24);
        return 2.0 * u - 1.0;
    }
};

} // namespace detail

inline void generate_frame(int N, std::vector<double>& re, std::vector<double>& im)
{
    using namespace detail;
    re.assign(N, 0.0);
    im.assign(N, 0.0);

    const int type = FFT1D_STIM_TYPE;

    if (type == STIM_MULTITONE) {
        const int ntones = FFT1D_STIM_NTONES;
        const int ks[8]  = {FFT1D_STIM_K0, FFT1D_STIM_K1, FFT1D_STIM_K2, FFT1D_STIM_K3,
                            FFT1D_STIM_K4, FFT1D_STIM_K5, FFT1D_STIM_K6, FFT1D_STIM_K7};
        const int as[8]  = {FFT1D_STIM_A0, FFT1D_STIM_A1, FFT1D_STIM_A2, FFT1D_STIM_A3,
                            FFT1D_STIM_A4, FFT1D_STIM_A5, FFT1D_STIM_A6, FFT1D_STIM_A7};
        const int phs[8] = {FFT1D_STIM_PH0, FFT1D_STIM_PH1, FFT1D_STIM_PH2, FFT1D_STIM_PH3,
                            FFT1D_STIM_PH4, FFT1D_STIM_PH5, FFT1D_STIM_PH6, FFT1D_STIM_PH7};
        for (int t = 0; t < ntones && t < 8; ++t) {
            const double a  = amp_from_milli(as[t]);
            const double ph = deg_to_rad(phs[t]);
            for (int n = 0; n < N; ++n) {
                const double angle = 2.0 * PI * static_cast<double>(ks[t])
                                     * static_cast<double>(n) / static_cast<double>(N) + ph;
                re[n] += a * std::cos(angle);
                im[n] += a * std::sin(angle);
            }
        }
    }
    else if (type == STIM_CHIRP) {
        const int    k0  = FFT1D_STIM_CHIRP_K0;
        const int    k1  = FFT1D_STIM_CHIRP_K1;
        const double amp = amp_from_milli(FFT1D_STIM_CHIRP_AMP);
        double phase = 0.0;
        for (int n = 0; n < N; ++n) {
            const double kinst = k0 + (k1 - k0) * static_cast<double>(n) / static_cast<double>(N);
            phase += 2.0 * PI * kinst / static_cast<double>(N);
            re[n] = amp * std::cos(phase);
            im[n] = amp * std::sin(phase);
        }
    }
    else if (type == STIM_IMPULSE) {
        const int    pos = FFT1D_STIM_IMP_POS % N;
        const double amp = amp_from_milli(FFT1D_STIM_IMP_AMP);
        re[pos] = amp;
        im[pos] = 0.0;
    }
    else if (type == STIM_NOISE) {
        xorshift32 rng(static_cast<std::uint32_t>(FFT1D_STIM_NOISE_SEED));
        const double amp = amp_from_milli(FFT1D_STIM_NOISE_AMP);
        for (int n = 0; n < N; ++n) {
            re[n] = amp * rng.uniform();
            im[n] = amp * rng.uniform();
        }
    }

    if (FFT1D_STIM_NORMALIZE) {
        double peak = 0.0;
        for (int n = 0; n < N; ++n) {
            peak = std::max(peak, std::fabs(re[n]));
            peak = std::max(peak, std::fabs(im[n]));
        }
        if (peak > TARGET_PEAK) {
            const double scale = TARGET_PEAK / peak;
            for (int n = 0; n < N; ++n) { re[n] *= scale; im[n] *= scale; }
        }
    }
}

inline const char* type_name()
{
    switch (FFT1D_STIM_TYPE) {
        case STIM_MULTITONE: return "multitone";
        case STIM_CHIRP:     return "chirp";
        case STIM_IMPULSE:   return "impulse";
        case STIM_NOISE:     return "noise";
        default:             return "unknown";
    }
}

} // namespace fft1d_stimulus
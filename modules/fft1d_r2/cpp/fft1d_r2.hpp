#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cstdint>

constexpr double PI = 3.14159265358979323846;

namespace fft1d_r2 {

/*
 * --------------------------------------------------------------------------
 * Parameters
 * --------------------------------------------------------------------------
 *
 * These values are equivalent to Verilog parameters.
 * They may be overridden from the build system using compiler defines.
 *
 *   #ifndef FFT1D_R2_N
 *   #define FFT1D_R2_N 64
 *   #endif
 *
 *   #ifndef FFT1D_R2_NB
 *   #define FFT1D_R2_NB 16
 *   #endif
 *
 *   #ifndef FFT1D_R2_NBF
 *   #define FFT1D_R2_NBF 15
 *   #endif
 *
 * N debe ser potencia de 2. El formato de datos por default es Q1.15
 * (NB=16, NBF=15, 1 bit de signo, sin bits enteros extra) -- valores en
 * [-1, 1).
 */

#ifndef FFT1D_R2_N
#define FFT1D_R2_N 512
#endif

#ifndef FFT1D_R2_NB
#define FFT1D_R2_NB 16
#endif

#ifndef FFT1D_R2_NBF
#define FFT1D_R2_NBF 15
#endif

enum parameters : int {
    N   = FFT1D_R2_N,
    NB  = FFT1D_R2_NB,
    NBF = FFT1D_R2_NBF,
};

/*
 * --------------------------------------------------------------------------
 * Localparameters
 * --------------------------------------------------------------------------
 */

constexpr int ct_clog2(int n) { int r = 0; while ((1 << r) < n) ++r; return r; }

enum localparameters : int {
    NBI   = NB - NBF,   // bits enteros (incluye el signo). Default: 1 -> Q1.15
    LOG2N = ct_clog2(N), // cantidad de etapas (6 para N=64)
    NH    = N / 2,      // mariposas por etapa / entradas de la ROM de twiddles
    LOADING_state = 0,
    COMPUTE_state = 1,
    OUTPUT_state = 2 ,
    NB_COUNT = LOG2N,
    NB_STAGE = ct_clog2(LOG2N),
    NB_BTFLY = LOG2N - 1
};

/*
 * --------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------
 *
 * ap_fixed<W, I, Q, O>:
 *   W = ancho total, I = bits enteros (con signo)
 *   Q = AP_RND   -> redondeo (round-half-up)
 *   O = AP_SAT   -> saturacion en overflow
 *
 * Elegido explicito (no el default AP_TRN/AP_WRAP de ap_fixed) porque
 * redondeo + saturacion es la convencion que venimos usando en todo el
 * datapath de punto fijo del proyecto.
 */

using bit_t     = rtl::bit_t;
using state_t   = ap_uint<2>;
using count_t   = ap_uint<NB_COUNT>;
using stage_t   = ap_uint<NB_STAGE>;
using btfly_t   = ap_uint<NB_BTFLY>;

#ifdef DOUBLE
    using in_t  = double;
    using out_t = double;
#else
    using in_t  = ap_fixed<NB, NBI, AP_RND, AP_SAT>;
    using out_t = ap_fixed<NB, NBI, AP_RND, AP_SAT>;
#endif

/*
 * --------------------------------------------------------------------------
 * fft1d_r2_model
 * --------------------------------------------------------------------------
 *
 * FFT radix-2 Cooley-Tukey de N puntos, arquitectura ITERATIVA basada en
 * memoria: una unica mariposa compleja, reutilizada N/2 * log2(N) veces
 * (32*6 = 192 ciclos para N=64), en vez de un pipeline con N-1 lineas de
 * retardo (R2SDF). Es la version mas simple posible: un multiplicador
 * complejo, dos memorias chicas (parte real/imaginaria), una ROM de
 * twiddles, y un FSM con 3 contadores.
 *
 * Protocolo:
 *   - LOADING : mientras i_valid=1, durante N ciclos, entra una muestra
 *               por ciclo en orden NATURAL (i_re/i_im). Internamente se
 *               guarda en la posicion bit-reversed (direccionamiento
 *               "gratis" durante la carga -> no hace falta una etapa de
 *               permutacion separada).
 *   - COMPUTE : LOG2N etapas x N/2 mariposas, una por ciclo. Cada mariposa
 *               escala su salida a la mitad (>>1) para no desbordar
 *               [-1,1) -- factor total acumulado: 1/N.
 *   - OUTPUT  : durante N ciclos, sale un resultado por ciclo (o_valid=1),
 *               ya en orden NATURAL. o_last=1 en la ultima muestra.
 *
 * Convencion de twiddle: W_N^{-nk} (FFT directa, sentido estandar).
 * Resultado: o == FFT(entrada) / N  (verificado bit a bit contra
 * np.fft.fft(x)/N en Python antes de escribir este archivo).
 *
 * Para invertir el sentido (IFFT), el unico cambio es el signo del angulo
 * en init() al precalcular la ROM de twiddles.
 */

class fft1d_r2_model final : public rtl::Module {
public:
    fft1d_r2_model() = default;

    /*
     * ----------------------------------------------------------------------
     * Ports
     * ----------------------------------------------------------------------
     */

    rtl::InPort<bit_t> i_valid;
    rtl::InPort<in_t>  i_re;
    rtl::InPort<in_t>  i_im;

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<bit_t> o_last;
    rtl::OutPort<out_t>  o_re;
    rtl::OutPort<out_t>  o_im;

    rtl::Reg<state_t> r_state;
    rtl::Reg<count_t> r_count; // usado en LOADING y en OUTPUT (0..N-1)
    rtl::Reg<stage_t> r_stage; // usado en COMPUTE (0..LOG2N-1)
    rtl::Reg<btfly_t> r_btfly;    // usado en COMPUTE (0..N/2-1)

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;

private:
    enum state_t : int {
        LOADING = 0,
        COMPUTE = 1,
        OUTPUT  = 2,
    };

    static int bit_reverse(int x, int bits);


    /*
     * ------------------------------------------------------------------
     * Memoria interna (no es un rtl::Reg: se modela como una RAM simple,
     * escrita directamente dentro de sequential())
     * ------------------------------------------------------------------
     */

    in_t mem_re_[N];
    in_t mem_im_[N];

    in_t twiddle_re_[NH];
    in_t twiddle_im_[NH];
};

} // namespace fft1d_r2
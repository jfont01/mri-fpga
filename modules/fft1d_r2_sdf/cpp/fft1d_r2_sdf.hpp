#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <cstdint>

namespace fft1d_r2sdf {

/*
 * --------------------------------------------------------------------------
 * Parameters
 * --------------------------------------------------------------------------
 *
 * Equivalentes a los parameters de Verilog; se pueden pisar con defines desde
 * el build system. N debe ser potencia de 2.
 */

#ifndef FFT1D_R2SDF_N
#define FFT1D_R2SDF_N 512
#endif

#ifndef FFT1D_R2SDF_NB
#define FFT1D_R2SDF_NB 16
#endif

#ifndef FFT1D_R2SDF_NBF
#define FFT1D_R2SDF_NBF 15
#endif

enum parameters : int {
    N   = FFT1D_R2SDF_N,
    NB  = FFT1D_R2SDF_NB,
    NBF = FFT1D_R2SDF_NBF
};

/*
 * --------------------------------------------------------------------------
 * Localparameters
 * --------------------------------------------------------------------------
 *
 * OJO con log2: ct_log2 hace division entera, o sea PISO. Es correcto para N
 * (potencia de dos) pero NO seria un $clog2 general. Si en algun momento hace
 * falta el techo de un valor que no es potencia de dos, usar ct_clog2.
 */

constexpr int ct_log2(int n)
{
    return (n <= 1) ? 0 : 1 + ct_log2(n / 2);
}

constexpr int ct_clog2(int n)
{
    int r = 0;
    while ((1 << r) < n) {
        ++r;
    }
    return r;
}

enum localparameters : int {
    NBI   = NB - NBF,      // bits enteros, signo incluido. Default: 1 -> Q1.15
    LOG2N = ct_log2(N),    // cantidad de etapas
    NH    = N / 2,         // entradas de la ROM de twiddles
    LATENCY = N - 1,       // ciclos desde la primera muestra de entrada hasta
                           // la primera salida valida, muestreando despues de
                           // sim.cycle() (o del posedge, del lado SV)
    DELAY_TOTAL = N - 1    // suma de todas las lineas de retardo
};

/*
 * --------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------
 *
 * Modos explicitos (AP_RND + AP_SAT). El default de ap_fixed es AP_TRN/AP_WRAP
 * y no coincide con lo que hace el modulo cast del RTL.
 */

using bit_t = rtl::bit_t;

#ifdef DOUBLE
using in_t  = double;
using out_t = double;
#else
using in_t  = ap_fixed<NB, NBI, AP_RND, AP_SAT>;
using out_t = ap_fixed<NB, NBI, AP_RND, AP_SAT>;
#endif

/*
 * --------------------------------------------------------------------------
 * fft1d_r2sdf_model
 * --------------------------------------------------------------------------
 *
 * FFT radix-2 Single-path Delay Feedback (R2SDF), arquitectura STREAMING:
 * una muestra por ciclo, entrada y salida continuas.
 *
 * Frente a la version iterativa basada en memoria:
 *   iterativa : 2N + (N/2)log2(N) ciclos por frame, 1 mariposa reusada
 *   R2SDF     : N ciclos por frame (throughput continuo), log2(N) mariposas
 *
 * Para N=512 son 3328 contra 512 ciclos: 6,5x menos.
 *
 * ESTRUCTURA
 * ----------
 * log2(N) etapas encadenadas. La etapa k tiene una linea de retardo de
 *
 *     L_k = 2^(log2(N)-1-k)      ->  N/2, N/4, ... , 1
 *
 * (total N-1 muestras complejas). En cada ciclo, con p = n mod L_k y
 * ctrl = bit log2(L_k) del contador:
 *
 *   ctrl = 0 : la linea pasa a la salida; la entrada entra a la linea
 *   ctrl = 1 : mariposa
 *                salida        = (d + x) / 2
 *                a la linea    = ((d - x) / 2) * W^e,   e = p * 2^k
 *
 * El x1/2 en las dos ramas acumula 2^-log2(N) = 1/N, misma convencion que la
 * FFT iterativa, asi que el analisis de punto fijo se traslada igual.
 *
 * DECISIONES DE DISENO
 * --------------------
 *   escalado        : x1/2 por etapa (1/N total)
 *   entrada         : 1 muestra/ciclo garantizada (sin huecos; usar un FIFO
 *                     externo si la fuente no lo sostiene)
 *   orden de salida : BIT-REVERSED, sin reordenar (el corner-turn de fft2d
 *                     lo absorbe gratis)
 *   latencia        : N-1 ciclos
 *
 * ROM de twiddles: una sola de N/2 entradas alcanza para TODAS las etapas.
 * El exponente maximo que se usa es N/2 - 1, y el indice sale de rebanar los
 * bits bajos del contador y correrlos k lugares (es cableado, no aritmetica).
 *
 * NO es bit-exacto contra fft1d_r2_model: R2SDF multiplica DESPUES de la
 * mariposa y la iterativa antes, asi que los redondeos caen en lugares
 * distintos. La equivalencia se verifica por gtest contra la DFT en double
 * con cota de error, no por comparacion bit a bit.
 */

class fft1d_r2sdf_model final : public rtl::Module {
public:
    fft1d_r2sdf_model() = default;

    /*
     * ----------------------------------------------------------------------
     * Ports
     * ----------------------------------------------------------------------
     */

    rtl::InPort<bit_t> i_valid;
    rtl::InPort<in_t>  i_re;
    rtl::InPort<in_t>  i_im;

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<out_t> o_re;
    rtl::OutPort<out_t> o_im;

    void connect_clocks(rtl::ClockDomain& clk) override;
    void init() override;
    void combinational() override;
    void sequential() override;

    /*
     * ----------------------------------------------------------------------
     * Registers
     * ----------------------------------------------------------------------
     *
     * Publicos para poder trazarlos desde el testbench con add_out_reg_o().
     *
     * El contador es libre y de LOG2N bits: tanto ctrl como el indice de
     * twiddle son periodicos en N, asi que envuelve solo.
     */

    rtl::Reg<int>  r_count;    // 0..N-1, libre
    rtl::Reg<bool> r_primed;   // pipeline lleno -> o_valid

    /*
     * Registro de salida.
     *
     * La cadena de etapas es combinacional de punta a punta, asi que sin este
     * registro la salida dependeria de la entrada del mismo ciclo. Eso hace
     * que el valor observado dependa de EN QUE MOMENTO del ciclo se muestrea,
     * y obliga a que el testbench C++ y el SystemVerilog usen exactamente la
     * misma convencion (fragil).
     *
     * Registrando la salida, o_re/o_im pasan a ser funcion solo del estado:
     * la observacion deja de ser ambigua. Ademas corta el camino combinacional
     * en la salida, que es lo que se quiere en el RTL.
     *
     * Costo: un ciclo mas de latencia -> LATENCY = N.
     */
    rtl::Reg<out_t> r_out_re;
    rtl::Reg<out_t> r_out_im;
    rtl::Reg<bool>  r_out_valid;

    /*
     * Longitud y offset de la linea de retardo de cada etapa dentro del
     * arreglo plano. La suma de longitudes es N-1.
     */
    static constexpr int stage_len(int k)
    {
        return 1 << (LOG2N - 1 - k);
    }

    static constexpr int stage_base(int k)
    {
        int base = 0;
        for (int j = 0; j < k; ++j) {
            base += 1 << (LOG2N - 1 - j);
        }
        return base;
    }

private:
    /*
     * Escalado x1/2 con redondeo.
     *
     * TEMPLATE a proposito: el argumento es la suma a±t, que vive en
     * ap_fixed<NB+1, NBI+1> (gana un bit entero). Si el parametro fuera
     * "const in_t&", la suma se cuantizaria Y SATURARIA al entrar, antes de
     * dividirse por dos: cada vez que |a±t| >= 1 el resultado quedaria pegado
     * al maximo. En modo DOUBLE el error no aparece (double no satura), asi
     * que solo se ve en punto fijo.
     */
    template <typename T>
    static in_t scale_half(const T& x)
    {
        return in_t(x * in_t(0.5));
    }

    /*
     * ------------------------------------------------------------------
     * Memoria interna
     * ------------------------------------------------------------------
     */

    in_t delay_re_[DELAY_TOTAL];
    in_t delay_im_[DELAY_TOTAL];

    in_t twiddle_re_[NH];
    in_t twiddle_im_[NH];

    // valor a escribir en la linea de cada etapa, calculado en combinational()
    in_t next_re_[LOG2N];
    in_t next_im_[LOG2N];

    // salida de la cadena combinacional, hacia el registro de salida
    in_t chain_re_;
    in_t chain_im_;
};

} // namespace fft1d_r2sdf
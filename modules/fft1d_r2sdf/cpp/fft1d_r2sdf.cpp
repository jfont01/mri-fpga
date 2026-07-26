#include "fft1d_r2sdf.hpp"

#include <cmath>

namespace fft1d_r2sdf {

namespace {
constexpr double PI = 3.14159265358979323846;
}

void fft1d_r2sdf_model::connect_clocks(rtl::ClockDomain& clk)
{
    clk.add(r_count);
    clk.add(r_primed);
    clk.add(r_out_re);
    clk.add(r_out_im);
    clk.add(r_out_valid);
}

void fft1d_r2sdf_model::init()
{
    i_valid = false;
    i_re    = in_t(0);
    i_im    = in_t(0);

    o_valid = false;
    o_re    = out_t(0);
    o_im    = out_t(0);

    r_count.set_initial_value(0);
    r_primed.set_initial_value(false);
    r_out_re.set_initial_value(out_t(0));
    r_out_im.set_initial_value(out_t(0));
    r_out_valid.set_initial_value(false);

    for (int i = 0; i < DELAY_TOTAL; ++i) {
        delay_re_[i] = in_t(0);
        delay_im_[i] = in_t(0);
    }

    for (int k = 0; k < LOG2N; ++k) {
        next_re_[k] = in_t(0);
        next_im_[k] = in_t(0);
    }

    chain_re_ = in_t(0);
    chain_im_ = in_t(0);

    /*
     * ROM de twiddles: W_N^k = exp(-j*2*pi*k/N), k = 0..N/2-1.
     * Una sola ROM sirve para todas las etapas (el exponente que usa la etapa
     * k es p*2^k, siempre menor que N/2).
     *
     * OJO: W^0 = 1.0 no es representable en Q1.15 y satura a 0x7FFF, o sea
     * 1 - 2^-15. Introduce una perdida de ganancia sistematica, chica pero
     * determinista, cada vez que el exponente es 0.
     */
    for (int k = 0; k < NH; ++k) {
        const double angle = -2.0 * PI * static_cast<double>(k) / static_cast<double>(N);
        twiddle_re_[k] = in_t(std::cos(angle));
        twiddle_im_[k] = in_t(std::sin(angle));
    }
}

void fft1d_r2sdf_model::combinational()
{
    const int count = r_count.o;

    /*
     * La cadena es combinacional de punta a punta: la muestra que entra en el
     * ciclo n atraviesa las log2(N) etapas en el mismo ciclo. Los unicos
     * registros del datapath son las lineas de retardo.
     *
     * Eso hace que el camino critico sea la cadena entera (log2(N) mariposas
     * y multiplicadores en serie). En el RTL habra que decidir si se insertan
     * registros entre etapas: mejora Fmax y suma log2(N) ciclos de latencia,
     * pero deja de espejar este modelo. Si se hace, hay que replicarlo aca.
     */
    in_t v_re = i_valid.value() ? i_re.value() : in_t(0);
    in_t v_im = i_valid.value() ? i_im.value() : in_t(0);

    for (int k = 0; k < LOG2N; ++k) {
        const int len  = stage_len(k);
        const int base = stage_base(k);

        const int p    = count % len;              // puntero de la linea
        const int ctrl = (count / len) % 2;        // = bit log2(len) del contador

        const in_t d_re = delay_re_[base + p];
        const in_t d_im = delay_im_[base + p];

        in_t y_re, y_im;   // hacia la etapa siguiente
        in_t n_re, n_im;   // hacia la linea de retardo

        if (ctrl == 0) {
            // carga: la linea entrega su valor mas viejo, la entrada entra
            y_re = d_re;
            y_im = d_im;
            n_re = v_re;
            n_im = v_im;
        }
        else {
            // mariposa, con x1/2 en las dos ramas
            y_re = scale_half(d_re + v_re);
            y_im = scale_half(d_im + v_im);

            const in_t s_re = scale_half(d_re - v_re);
            const in_t s_im = scale_half(d_im - v_im);

            /*
             * Twiddle sobre la rama que vuelve a la linea.
             *
             * Las etapas con linea de retardo de 2 o menos (las dos ultimas)
             * solo usan twiddles triviales: W^0 = 1 (addr par) y W^(N/4) = -j
             * (addr impar). El RTL (fft1d_r2sdf_stage.v, gen_trivial) los aplica
             * con valores EXACTOS por cableado, NO desde la ROM: en Q1.15 el
             * W^0 de la ROM esta saturado a 0x7FFF (1 - 2^-15), asi que leerlo
             * introduciria una perdida de ganancia de 1 LSB que el RTL no tiene.
             * Para mantener la bit-exactitud del vm, aca hacemos lo mismo.
             *
             *   -j * (re + j*im) = im - j*re   ->   (re, im) -> (im, -re)
             */
            if (len <= 2) {
                const int ctrl_mj = (len > 1) ? (p & 1) : 0;   // addr[0]
                if (ctrl_mj == 0) {
                    // x1: identidad
                    n_re = s_re;
                    n_im = s_im;
                }
                else {
                    // x(-j): (re, im) -> (im, -re), con -re saturado igual que el RTL
                    n_re = s_im;
                    n_im = in_t(-s_re);
                }
            }
            else {
                // twiddle no trivial desde la ROM: e = p * 2^k
                const int  e     = p << k;
                const in_t tw_re = twiddle_re_[e];
                const in_t tw_im = twiddle_im_[e];

                // producto complejo exacto, UNA cuantizacion por componente
                n_re = in_t(tw_re * s_re - tw_im * s_im);
                n_im = in_t(tw_re * s_im + tw_im * s_re);
            }
        }

        next_re_[k] = n_re;
        next_im_[k] = n_im;

        v_re = y_re;
        v_im = y_im;
    }

    // las salidas salen del registro: funcion del estado, no de la entrada
    o_re    = r_out_re.o;
    o_im    = r_out_im.o;
    o_valid = bit_t(r_out_valid.o);

    // se guardan para que sequential() los cargue en el registro de salida
    chain_re_ = v_re;
    chain_im_ = v_im;
}

void fft1d_r2sdf_model::sequential()
{
    const int count = r_count.o;

    /*
     * Escritura de las lineas de retardo. Se modelan como buffers circulares
     * con puntero p = n mod L, que es equivalente a un registro de
     * desplazamiento de largo L (el valor leido es el escrito L ciclos atras)
     * y se corresponde con la implementacion real: BRAM mas un contador de
     * direccion para las etapas largas, registros para las cortas.
     */
    for (int k = 0; k < LOG2N; ++k) {
        const int len  = stage_len(k);
        const int base = stage_base(k);
        const int p    = count % len;

        delay_re_[base + p] = next_re_[k];
        delay_im_[base + p] = next_im_[k];
    }

    // contador libre de LOG2N bits: ctrl y el indice de twiddle son
    // periodicos en N, asi que envolver es correcto
    r_count.i = (count + 1) % N;

    r_primed.i = r_primed.o || (count == N - 1);

    r_out_re.i    = out_t(chain_re_);
    r_out_im.i    = out_t(chain_im_);
    r_out_valid.i = r_primed.o || (count == N - 1);
}

} // namespace fft1d_r2sdf
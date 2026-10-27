#pragma once

// -----------------------------------------------------------------------------
// fft2d.hpp -- modelo ciclo a ciclo de la IFFT 2D por bobina (paso 5 del plan de
// la FFT 2D), con MEMORIA IDEAL entre las dos pasadas.
//
//   i_* -> nucleo 1 (fft1d_r22sdf, N1 = largo de fila) -> buffer[wr_sel]
//                                                            (ping-pong)
//          buffer[rd_sel] -> registro de lectura -> nucleo 2 (N2 = filas) -> o_*
//
// Convenciones (bitacora, seccion 3.8):
//   * Entrada por bobina: N2 filas k' de N1 muestras r cada una, readout
//     contiguo (K_hw[l, k', r]). Cada fila entra sin huecos (el SDF lo exige);
//     entre filas puede haber huecos.
//   * Pasada 1 sobre r, pasada 2 sobre k', las dos con INVERSE = 1 (e^{+j}).
//   * La pasada 1 escribe cada salida en el orden en que sale (slot t, que es
//     bit-reversed): direccion fila * N1 + t. No se reordena nada.
//   * La pasada 2 lee por columnas: slot t1 = 0..N1-1 y, dentro de cada una,
//     k' = 0..N2-1 en orden natural. El corner-turn es una transposicion pura.
//   * Salida: muestra o = t1 * N2 + t2 de la bobina = A[p = br(t2), x = br(t1)],
//     con A = 2^-(S1+S2) * sum_{k',r} K[k', r] e^{+j2pi(r x / N1 + k' p / N2)}.
//
// Memoria ideal: dos buffers de N2 x N1 complejos (el formato de salida de la
// pasada 1), un puerto de escritura y uno de lectura con latencia 1 (salida
// registrada, como una BRAM). Si en un ciclo se lee y se escribe la misma
// posicion, se lee el dato viejo (read-first).
//
// Protocolo:
//   * Escritor: la bobina l escribe en el buffer l mod 2. Al escribir la ultima
//     muestra marca el buffer lleno (full) y pasa al otro.
//   * Lector: si esta libre y el buffer rd_sel esta lleno, lo toma (baja full)
//     y lee N1 * N2 posiciones seguidas, una por ciclo. Si al terminar el otro
//     ya esta lleno, sigue sin burbuja.
//   * Regla de seguridad: el escritor puede empezar una bobina en un buffer
//     vacio o en uno que se esta leyendo, nunca en uno lleno sin leer
//     (overrun). Ademas, cada escritura sobre el buffer que se esta leyendo
//     verifica que esa posicion ya se leyo. Las dos condiciones tiran
//     std::logic_error: con entrada a 1 muestra/ciclo no pueden darse (margen
//     minimo N1 + N2 - 1 - d ciclos), asi que si saltan es un bug.
//
// Latencia hasta el primer o_valid, con la misma convencion que el modelo 1D
// (la muestra 0 entra en el ciclo 0):
//
//     L2D = L1 + T + 2 + L2,   T = N1 * N2
//
// El 2 es la escritura (el dato sale registrado del nucleo 1 y se escribe en el
// flanco siguiente, junto con full) y la lectura registrada. El lector arranca
// en el ciclo en que ve full. 1040 + 524288 + 2 + 527 = 525857 para 1024 x 512.
//
// Formatos: los del modelo 1D (de compilacion, comunes a los dos nucleos). Se
// pueden fijar con FFT2D_NB, FFT2D_NBF, FFT2D_NB_TW, ... (los pasa el JSON con
// auto_defines) o directamente con FFT1D_R22SDF_*. Por defecto, los elegidos:
// 20/18/16. El resto (N1, N2, mascaras, twiddles, CMUL_PIPE, INVERSE) es de
// elaboracion: fft2d::Params, con defaults en FFT2D_*.
// -----------------------------------------------------------------------------

#ifndef FFT2D_NB
#define FFT2D_NB 20
#endif
#ifndef FFT2D_NBF
#define FFT2D_NBF (FFT2D_NB - 1)
#endif
#ifndef FFT2D_NB_TW
#define FFT2D_NB_TW 18
#endif
#ifndef FFT2D_NB_IN
#define FFT2D_NB_IN 16
#endif
#ifndef FFT2D_NB_OUT
#define FFT2D_NB_OUT 16
#endif

#ifndef FFT1D_R22SDF_NB
#define FFT1D_R22SDF_NB FFT2D_NB
#endif
#ifndef FFT1D_R22SDF_NBF
#define FFT1D_R22SDF_NBF FFT2D_NBF
#endif
#ifndef FFT1D_R22SDF_NB_TW
#define FFT1D_R22SDF_NB_TW FFT2D_NB_TW
#endif
#if defined(FFT2D_NBF_TW) && !defined(FFT1D_R22SDF_NBF_TW)
#define FFT1D_R22SDF_NBF_TW FFT2D_NBF_TW
#endif
#ifndef FFT1D_R22SDF_NB_IN
#define FFT1D_R22SDF_NB_IN FFT2D_NB_IN
#endif
#if defined(FFT2D_NBF_IN) && !defined(FFT1D_R22SDF_NBF_IN)
#define FFT1D_R22SDF_NBF_IN FFT2D_NBF_IN
#endif
#ifndef FFT1D_R22SDF_NB_OUT
#define FFT1D_R22SDF_NB_OUT FFT2D_NB_OUT
#endif
#if defined(FFT2D_NBF_OUT) && !defined(FFT1D_R22SDF_NBF_OUT)
#define FFT1D_R22SDF_NBF_OUT FFT2D_NBF_OUT
#endif

#include "../../fft1d_r22sdf/cpp/fft1d_r22sdf.hpp"

#include <stdexcept>
#include <string>
#include <vector>

/*
 * Parametros de elaboracion por defecto: la configuracion de produccion. Las
 * rutas de twiddles son relativas a la carpeta del modulo fft2d (desde donde
 * corren los runners).
 */
#ifndef FFT2D_N1
#define FFT2D_N1 1024                // pasada 1: largo de fila (readout)
#endif
#ifndef FFT2D_N2
#define FFT2D_N2 512                 // pasada 2: filas por bobina (phase-encode)
#endif
#ifndef FFT2D_SCALE_MASK1
#define FFT2D_SCALE_MASK1 801        // provisoria (bitacora, seccion 4.8)
#endif
#ifndef FFT2D_SCALE_MASK2
#define FFT2D_SCALE_MASK2 384
#endif
#ifndef FFT2D_TW_DIR1
#define FFT2D_TW_DIR1 ../fft1d_r22sdf/rtl/twiddles/n1024_tw18
#endif
#ifndef FFT2D_TW_DIR2
#define FFT2D_TW_DIR2 ../fft1d_r22sdf/rtl/twiddles/n512_tw18
#endif
#ifndef FFT2D_CMUL_PIPE
#define FFT2D_CMUL_PIPE 1
#endif
#ifndef FFT2D_INVERSE
#define FFT2D_INVERSE 1
#endif
// Redondeo convergente (empates al par) en las mariposas que escalan y en la
// salida de los dos nucleos (fft1d_r22sdf, ROUND_CONV). Default 0: half-up.
#ifndef FFT2D_ROUND_CONV
#define FFT2D_ROUND_CONV 0
#endif

namespace fft2d {

using fft1d_r22sdf::bit_t;
using fft1d_r22sdf::in_t;
using fft1d_r22sdf::out_t;

// El buffer guarda la salida de la pasada 1, que entra a la pasada 2: tiene
// que entrar sin perdida en el formato de entrada.
static_assert(fft1d_r22sdf::NBF_IN >= fft1d_r22sdf::NBF_OUT &&
              (fft1d_r22sdf::NB_IN - fft1d_r22sdf::NBF_IN) >=
              (fft1d_r22sdf::NB_OUT - fft1d_r22sdf::NBF_OUT),
              "fft2d: la salida de la pasada 1 no entra sin perdida en la entrada de la pasada 2");

struct Params {
    fft1d_r22sdf::Params pass1;      // N1 = pass1.n (largo de fila)
    fft1d_r22sdf::Params pass2;      // N2 = pass2.n (filas por bobina)

    Params()
    {
        pass1.n          = FFT2D_N1;
        pass1.scale_mask = FFT2D_SCALE_MASK1;
        pass1.inverse    = (FFT2D_INVERSE != 0);
        pass1.cmul_pipe  = (FFT2D_CMUL_PIPE != 0);
        pass1.tw_dir     = FFT1D_R22SDF_STRINGIFY(FFT2D_TW_DIR1);
        pass1.round_conv = (FFT2D_ROUND_CONV != 0);

        pass2.n          = FFT2D_N2;
        pass2.scale_mask = FFT2D_SCALE_MASK2;
        pass2.inverse    = (FFT2D_INVERSE != 0);
        pass2.cmul_pipe  = (FFT2D_CMUL_PIPE != 0);
        pass2.tw_dir     = FFT1D_R22SDF_STRINGIFY(FFT2D_TW_DIR2);
        pass2.round_conv = (FFT2D_ROUND_CONV != 0);
    }

    int  n1()    const { return pass1.n; }
    int  n2()    const { return pass2.n; }
    long frame() const { return static_cast<long>(pass1.n) * pass2.n; }   // T

    // Ver cabecera. long: con 1024 x 512 entra en int, pero no sobra.
    long latency() const { return pass1.latency() + frame() + 2 + pass2.latency(); }

    // S1 + S2: la salida es la suma doble por 2^-scale_shift()
    int scale_shift() const { return pass1.scale_shift() + pass2.scale_shift(); }

    void validate() const
    {
        pass1.validate();
        pass2.validate();
        if (frame() > (1L << 30)) {
            throw std::invalid_argument("fft2d: N1 * N2 > 2^30");
        }
    }
};

class fft2d_model final : public rtl::Module {
public:
    /*
     * Ports
     */
    rtl::InPort<bit_t> i_valid;
    rtl::InPort<in_t>  i_re;
    rtl::InPort<in_t>  i_im;

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<bit_t> o_last;      // ultima muestra de la bobina
    rtl::OutPort<out_t> o_re;
    rtl::OutPort<out_t> o_im;

    explicit fft2d_model(const Params& p = Params{})
        : p_(checked(p)),
          core1_(p_.pass1),
          core2_(p_.pass2),
          n1_(p_.n1()),
          n2_(p_.n2()),
          t_(static_cast<int>(p_.frame()))
    {
        for (Buffer& b : buf_) {
            b.re.assign(t_, out_t(0));
            b.im.assign(t_, out_t(0));
        }
    }

    const Params& params()  const { return p_; }
    long          latency() const { return p_.latency(); }

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        core1_.connect_clocks(clk);
        core2_.connect_clocks(clk);

        clk.add(wr_sel);
        clk.add(wr_count);
        clk.add(full[0]);
        clk.add(full[1]);
        clk.add(rd_active);
        clk.add(rd_sel);
        clk.add(rd_count);
        clk.add(rd_valid);
        clk.add(rd_re);
        clk.add(rd_im);
        clk.add(out_count);
    }

    void init() override
    {
        core1_.init();
        core2_.init();

        wr_sel.set_initial_value(0);
        wr_count.set_initial_value(0);
        full[0].set_initial_value(false);
        full[1].set_initial_value(false);
        rd_active.set_initial_value(false);
        rd_sel.set_initial_value(0);
        rd_count.set_initial_value(0);
        rd_valid.set_initial_value(false);
        rd_re.set_initial_value(in_t(0));
        rd_im.set_initial_value(in_t(0));
        out_count.set_initial_value(0);

        o_valid = bit_t(false);
        o_last  = bit_t(false);
        o_re    = out_t(0);
        o_im    = out_t(0);
    }

    void combinational() override
    {
        // nucleo 1: los puertos de entrada del top
        core1_.i_valid = i_valid.value();
        core1_.i_re    = i_re.value();
        core1_.i_im    = i_im.value();
        core1_.combinational();

        // nucleo 2: la salida registrada de la memoria
        core2_.i_valid = bit_t(rd_valid.o);
        core2_.i_re    = rd_re.o;
        core2_.i_im    = rd_im.o;
        core2_.combinational();

        const bool v = core2_.o_valid.value().to_bool();
        o_valid = bit_t(v);
        o_last  = bit_t(v && (out_count.o == t_ - 1));
        o_re    = core2_.o_re.value();     // el nucleo ya fuerza 0 con valid = 0
        o_im    = core2_.o_im.value();
    }

    void sequential() override
    {
        core1_.sequential();
        core2_.sequential();

        // ------------------------------------------------ lector (read-first)
        const int  rs       = rd_sel.o;
        const bool rd_start = !rd_active.o && full[rs].o;
        const bool reading  = rd_active.o || rd_start;

        if (reading) {
            const int pos  = rd_count.o;               // orden de lectura
            const int addr = (pos % n2_) * n1_ + pos / n2_;   // fila k', columna t1
            const bool last = (pos == t_ - 1);

            rd_re.i     = in_t(buf_[rs].re[addr]);
            rd_im.i     = in_t(buf_[rs].im[addr]);
            rd_valid.i  = true;
            rd_count.i  = last ? 0 : pos + 1;
            rd_active.i = !last;
            rd_sel.i    = last ? 1 - rs : rs;
        }
        else {
            rd_re.i     = in_t(0);
            rd_im.i     = in_t(0);
            rd_valid.i  = false;
            rd_active.i = false;
        }

        // ------------------------------------------------------ escritor
        int set_full = -1;
        if (core1_.o_valid.value().to_bool()) {
            const int  ws   = wr_sel.o;
            const int  idx  = wr_count.o;              // = fila * N1 + slot
            const bool last = (idx == t_ - 1);

            if (idx == 0 && full[ws].o) {
                throw std::logic_error(
                    "fft2d: overrun, la pasada 1 empieza una bobina en un buffer lleno sin leer");
            }
            if (reading && rs == ws) {
                const int k   = idx / n1_;
                const int t   = idx % n1_;
                const int pos = t * n2_ + k;           // cuando lo lee el lector
                if (pos > rd_count.o) {
                    throw std::logic_error(
                        "fft2d: se pisa una posicion que la pasada 2 todavia no leyo");
                }
            }

            buf_[ws].re[idx] = core1_.o_re.value();
            buf_[ws].im[idx] = core1_.o_im.value();

            wr_count.i = last ? 0 : idx + 1;
            if (last) {
                set_full = ws;
                wr_sel.i = 1 - ws;
            }
        }

        // ------------------------------------------------- banderas full
        const int clr_full = rd_start ? rs : -1;
        if (set_full >= 0 && set_full == clr_full) {
            throw std::logic_error("fft2d: full se pone y se baja en el mismo ciclo");
        }
        for (int b = 0; b < 2; ++b) {
            full[b].i = (full[b].o && b != clr_full) || b == set_full;
        }

        // ------------------------------------------- contador de salida
        if (core2_.o_valid.value().to_bool()) {
            out_count.i = (out_count.o == t_ - 1) ? 0 : out_count.o + 1;
        }
    }

private:
    static const Params& checked(const Params& p)
    {
        p.validate();
        return p;
    }

    struct Buffer {
        std::vector<out_t> re, im;
    };

    Params p_;

public:
    // Estado observable (para trazas y tests). Los nucleos son modelos 1D
    // completos: core1_.unit_[0].di_count, etc.
    fft1d_r22sdf::fft1d_r22sdf_model core1_;
    fft1d_r22sdf::fft1d_r22sdf_model core2_;

    rtl::Reg<int>    wr_sel;      // buffer que escribe la pasada 1
    rtl::Reg<int>    wr_count;    // posicion dentro de la bobina (0 .. T-1)
    rtl::Reg<bool>   full[2];     // escrito completo y todavia no tomado
    rtl::Reg<bool>   rd_active;
    rtl::Reg<int>    rd_sel;      // buffer que lee la pasada 2
    rtl::Reg<int>    rd_count;    // posicion de lectura (0 .. T-1)
    rtl::Reg<bool>   rd_valid;    // salida registrada de la memoria
    rtl::Reg<in_t>   rd_re, rd_im;
    rtl::Reg<int>    out_count;   // muestra de salida dentro de la bobina

private:
    const int n1_;
    const int n2_;
    const int t_;
    Buffer    buf_[2];
};

} // namespace fft2d

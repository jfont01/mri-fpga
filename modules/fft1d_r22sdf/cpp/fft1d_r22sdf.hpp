#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// -----------------------------------------------------------------------------
// fft1d_r22sdf.hpp -- modelo bit-exacto del RTL fft1d_r22sdf.v (radix-2^2 SDF).
//
// Espeja la estructura del RTL: una cadena de log4(N) unidades, cada una con
//
//   i_din -> BF2II (linea M/2, conmutador -j) -> reg
//         -> BF2I  (linea M/4)                -> reg
//         -> multiplicador de twiddle          -> reg -> o_dout
//
// y el control por etapa (cada etapa lleva su propio contador, que arranca
// cuando los datos efectivamente le llegan). La ultima unidad no multiplica.
//
// Puntos sensibles replicados del RTL (los que costaron encontrar):
//
//   * tw_num se TRUNCA a LOG_N-2 bits. Los dos bits altos del contador ya se
//     consumieron en tw_sel; dejarlos corre la direccion en N/2, y como
//     W^(k+N/2) = -W^k eso equivale a invertir el signo del twiddle.
//   * El conmutador -j actua solo en la fase de carga (bf = 0).
//   * bf2_bf y bf2_start estan REGISTRADOS, para alinearse con el registro de
//     pipeline que separa las dos mariposas.
//
// Aritmetica: el escalado x1/2 de la mariposa se hace sumando en precision
// exacta y cuantizando a in_t (AP_RND, AP_SAT), que es exactamente lo que hace
// el cast del RTL al bajar un bit fraccional.
// -----------------------------------------------------------------------------

namespace fft1d_r22sdf {

#ifndef FFT1D_R22SDF_N
#define FFT1D_R22SDF_N 64
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

constexpr int ct_log2(int n) { return (n <= 1) ? 0 : 1 + ct_log2(n / 2); }

enum localparameters : int {
    LOG_N   = ct_log2(N),
    LOG2N   = LOG_N,             // alias (nomenclatura de los otros modulos)
    N_UNITS = LOG_N / 2,         // log4(N)

    /*
     * Latencia hasta el primer o_valid, contando desde la primera muestra de
     * entrada. El pipeline esta segmentado (registros entre las dos mariposas
     * de cada unidad y a la salida del multiplicador), asi que es bastante
     * mayor que la del R2SDF: 70 para N=64, 265 para N=256, 1036 para N=1024.
     * Verificada contra el RTL.
     */
    LATENCY = N + 3 * (N_UNITS - 1)
};

#ifdef DOUBLE
using in_t  = double;
using out_t = double;
#else
using in_t  = ap_fixed<NB, NB - NBF, AP_RND, AP_SAT>;
using out_t = ap_fixed<NB, NB - NBF, AP_RND, AP_SAT>;
#endif

using bit_t = rtl::bit_t;

class fft1d_r22sdf_model final : public rtl::Module {
public:
    /*
     * Ports
     */
    rtl::InPort<bit_t> i_valid;
    rtl::InPort<in_t>  i_re;
    rtl::InPort<in_t>  i_im;

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<bit_t> o_last;
    rtl::OutPort<out_t> o_re;
    rtl::OutPort<out_t> o_im;

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        for (int u = 0; u < N_UNITS; ++u) {
            Unit& q = unit_[u];
            clk.add(q.di_count);
            clk.add(q.addr1);
            clk.add(q.addr2);
            clk.add(q.bf1_sp_en);
            clk.add(q.bf1_count);
            clk.add(q.bf1_do_re);
            clk.add(q.bf1_do_im);
            clk.add(q.bf2_bf);
            clk.add(q.bf2_start);
            clk.add(q.bf2_sp_en);
            clk.add(q.bf2_count);
            clk.add(q.bf2_do_re);
            clk.add(q.bf2_do_im);
            clk.add(q.bf2_do_en);
            clk.add(q.tw_re);
            clk.add(q.tw_im);
            clk.add(q.mu_en);
            clk.add(q.mu_do_re);
            clk.add(q.mu_do_im);
            clk.add(q.mu_do_en);
        }
        clk.add(r_out_count);
    }

    void init() override
    {
        build_twiddle_rom();

        for (int u = 0; u < N_UNITS; ++u) {
            Unit& q = unit_[u];

            q.log_m    = LOG_N - 2 * u;
            q.depth1   = 1 << (q.log_m - 1);
            q.depth2   = 1 << (q.log_m - 2);
            q.has_mult = (q.log_m != 2);

            q.dl1_re.assign(q.depth1, in_t(0));
            q.dl1_im.assign(q.depth1, in_t(0));
            q.dl2_re.assign(q.depth2, in_t(0));
            q.dl2_im.assign(q.depth2, in_t(0));

            q.di_count.set_initial_value(0);
            q.addr1.set_initial_value(0);
            q.addr2.set_initial_value(0);
            q.bf1_sp_en.set_initial_value(false);
            q.bf1_count.set_initial_value(0);
            q.bf1_do_re.set_initial_value(in_t(0));
            q.bf1_do_im.set_initial_value(in_t(0));
            q.bf2_bf.set_initial_value(false);
            q.bf2_start.set_initial_value(false);
            q.bf2_sp_en.set_initial_value(false);
            q.bf2_count.set_initial_value(0);
            q.bf2_do_re.set_initial_value(in_t(0));
            q.bf2_do_im.set_initial_value(in_t(0));
            q.bf2_do_en.set_initial_value(false);
            q.tw_re.set_initial_value(in_t(0));
            q.tw_im.set_initial_value(in_t(0));
            q.mu_en.set_initial_value(false);
            q.mu_do_re.set_initial_value(in_t(0));
            q.mu_do_im.set_initial_value(in_t(0));
            q.mu_do_en.set_initial_value(false);
        }

        r_out_count.set_initial_value(0);

        o_valid = bit_t(false);
        o_last  = bit_t(false);
        o_re    = out_t(0);
        o_im    = out_t(0);
    }

    void combinational() override
    {
        evaluate();

        const Unit& last = unit_[N_UNITS - 1];
        const bool  v    = last.has_mult ? last.mu_do_en.o : last.bf2_do_en.o;

        o_valid = bit_t(v);
        o_last  = bit_t(v && (r_out_count.o == N - 1));

        /*
         * Gating de salida: con o_valid=0 el RTL entrega basura del pipeline sin
         * cebar (X en simulacion). Ambos lados fuerzan 0 para que el vm no
         * compare muestras invalidas.
         */
        if (!v) {
            o_re = out_t(0);
            o_im = out_t(0);
        }
        else if (last.has_mult) {
            o_re = out_t(last.mu_do_re.o);
            o_im = out_t(last.mu_do_im.o);
        }
        else {
            o_re = out_t(last.bf2_do_re.o);
            o_im = out_t(last.bf2_do_im.o);
        }
    }

    void sequential() override
    {
        evaluate();

        bool  chain_valid = i_valid.value().to_bool();

        for (int u = 0; u < N_UNITS; ++u) {
            Unit& q = unit_[u];
            const Wires& w = wire_[u];

            // --- linea de la 1ra etapa (BF2II) ---
            q.dl1_re[q.addr1.o % q.depth1] = w.db1_di_re;
            q.dl1_im[q.addr1.o % q.depth1] = w.db1_di_im;
            q.addr1.i = (q.addr1.o + 1) % q.depth1;

            // --- linea de la 2da etapa (BF2I) ---
            q.dl2_re[q.addr2.o % q.depth2] = w.db2_di_re;
            q.dl2_im[q.addr2.o % q.depth2] = w.db2_di_im;
            q.addr2.i = (q.addr2.o + 1) % q.depth2;

            // --- contador de entrada ---
            q.di_count.i = chain_valid ? ((q.di_count.o + 1) % N) : 0;

            // --- control de la 1ra etapa ---
            const bool bf1_start = (q.di_count.o == q.depth1 - 1);
            const bool bf1_end   = (q.bf1_count.o == N - 1);
            q.bf1_sp_en.i = bf1_start ? true : (bf1_end ? false : q.bf1_sp_en.o);
            q.bf1_count.i = q.bf1_sp_en.o ? ((q.bf1_count.o + 1) % N) : 0;

            q.bf1_do_re.i = w.bf1_sp_re;
            q.bf1_do_im.i = w.bf1_sp_im;

            // --- control de la 2da etapa (registrado) ---
            q.bf2_bf.i    = bit_of(q.bf1_count.o, q.log_m - 2);
            q.bf2_start.i = (q.bf1_count.o == q.depth2 - 1) && q.bf1_sp_en.o;

            const bool bf2_end = (q.bf2_count.o == N - 1);
            q.bf2_sp_en.i = q.bf2_start.o ? true : (bf2_end ? false : q.bf2_sp_en.o);
            q.bf2_count.i = q.bf2_sp_en.o ? ((q.bf2_count.o + 1) % N) : 0;

            q.bf2_do_re.i = w.bf2_sp_re;
            q.bf2_do_im.i = w.bf2_sp_im;
            q.bf2_do_en.i = q.bf2_sp_en.o;

            // --- twiddle y multiplicador ---
            if (q.has_mult) {
                q.tw_re.i = tw_rom_re_[w.tw_addr];
                q.tw_im.i = tw_rom_im_[w.tw_addr];
                q.mu_en.i = (w.tw_addr != 0);

                q.mu_do_re.i = q.mu_en.o ? w.mu_re : q.bf2_do_re.o;
                q.mu_do_im.i = q.mu_en.o ? w.mu_im : q.bf2_do_im.o;
                q.mu_do_en.i = q.bf2_do_en.o;
            }

            // la entrada de la unidad siguiente es la salida (registrada) de esta
            chain_valid = q.has_mult ? q.mu_do_en.o : q.bf2_do_en.o;
        }

        // --- contador de salida (para o_last) ---
        const Unit& last = unit_[N_UNITS - 1];
        const bool  v    = last.has_mult ? last.mu_do_en.o : last.bf2_do_en.o;
        r_out_count.i = v ? ((r_out_count.o + 1) % N) : r_out_count.o;
    }

    // --- estado expuesto para trazado del vm ---
    struct Unit {
        int  log_m{0};
        int  depth1{1};
        int  depth2{1};
        bool has_mult{true};

        std::vector<in_t> dl1_re, dl1_im;
        std::vector<in_t> dl2_re, dl2_im;

        rtl::Reg<int>  di_count;
        rtl::Reg<int>  addr1;
        rtl::Reg<int>  addr2;

        rtl::Reg<bool> bf1_sp_en;
        rtl::Reg<int>  bf1_count;
        rtl::Reg<in_t> bf1_do_re, bf1_do_im;

        rtl::Reg<bool> bf2_bf;
        rtl::Reg<bool> bf2_start;
        rtl::Reg<bool> bf2_sp_en;
        rtl::Reg<int>  bf2_count;
        rtl::Reg<in_t> bf2_do_re, bf2_do_im;
        rtl::Reg<bool> bf2_do_en;

        rtl::Reg<in_t> tw_re, tw_im;
        rtl::Reg<bool> mu_en;
        rtl::Reg<in_t> mu_do_re, mu_do_im;
        rtl::Reg<bool> mu_do_en;
    };

    // valores combinacionales de cada unidad, recalculados por evaluate()
    struct Wires {
        in_t bf1_sp_re{0}, bf1_sp_im{0};
        in_t db1_di_re{0}, db1_di_im{0};
        in_t bf2_sp_re{0}, bf2_sp_im{0};
        in_t db2_di_re{0}, db2_di_im{0};
        in_t mu_re{0}, mu_im{0};
        int  tw_addr{0};
    };

private:
    static bool bit_of(int value, int pos)
    {
        return ((value >> pos) & 1) != 0;
    }

    /*
     * Escalado x1/2 de la mariposa. La suma/resta es exacta y la division por 2
     * tambien, asi que cuantizar el resultado a in_t (AP_RND, AP_SAT) reproduce
     * exactamente el cast del RTL, que baja un bit fraccional.
     */
    static in_t half(const in_t& a, const in_t& b, bool add)
    {
        const double s = add ? (double(a) + double(b)) : (double(a) - double(b));
        return in_t(s * 0.5);
    }

    void build_twiddle_rom()
    {
        constexpr double PI = 3.14159265358979323846;
        tw_rom_re_.resize(N);
        tw_rom_im_.resize(N);
        for (int k = 0; k < N; ++k) {
            const double ang = -2.0 * PI * static_cast<double>(k) / static_cast<double>(N);
            tw_rom_re_[k] = in_t(std::cos(ang));
            tw_rom_im_[k] = in_t(std::sin(ang));
        }
    }

    /*
     * Recalcula toda la logica combinacional. Se llama tanto desde
     * combinational() como al principio de sequential(), para que los valores
     * de wire_ correspondan al estado actual de los registros.
     */
    void evaluate()
    {
        in_t din_re = i_re.value();
        in_t din_im = i_im.value();

        for (int u = 0; u < N_UNITS; ++u) {
            Unit&  q = unit_[u];
            Wires& w = wire_[u];

            // ---------------------------------------- 1ra etapa: BF2II
            const in_t d1_re = q.dl1_re[q.addr1.o % q.depth1];
            const in_t d1_im = q.dl1_im[q.addr1.o % q.depth1];

            const bool bf1_bf = bit_of(q.di_count.o, q.log_m - 1);
            const bool bf1_mj = (((q.bf1_count.o >> (q.log_m - 2)) & 0x3) == 0x3);

            const in_t y0_re = half(d1_re, din_re, true);
            const in_t y0_im = half(d1_im, din_im, true);
            const in_t y1_re = half(d1_re, din_re, false);
            const in_t y1_im = half(d1_im, din_im, false);

            if (bf1_bf) {
                w.db1_di_re = y1_re;
                w.db1_di_im = y1_im;
                w.bf1_sp_re = y0_re;
                w.bf1_sp_im = y0_im;
            }
            else {
                w.db1_di_re = din_re;
                w.db1_di_im = din_im;
                if (bf1_mj) {
                    // -j * (re + j im) = im - j re
                    w.bf1_sp_re = d1_im;
                    w.bf1_sp_im = in_t(-double(d1_re));
                }
                else {
                    w.bf1_sp_re = d1_re;
                    w.bf1_sp_im = d1_im;
                }
            }

            // ----------------------------------------- 2da etapa: BF2I
            const in_t d2_re = q.dl2_re[q.addr2.o % q.depth2];
            const in_t d2_im = q.dl2_im[q.addr2.o % q.depth2];

            const in_t x2_re = q.bf1_do_re.o;
            const in_t x2_im = q.bf1_do_im.o;

            const in_t z0_re = half(d2_re, x2_re, true);
            const in_t z0_im = half(d2_im, x2_im, true);
            const in_t z1_re = half(d2_re, x2_re, false);
            const in_t z1_im = half(d2_im, x2_im, false);

            if (q.bf2_bf.o) {
                w.db2_di_re = z1_re;
                w.db2_di_im = z1_im;
                w.bf2_sp_re = z0_re;
                w.bf2_sp_im = z0_im;
            }
            else {
                w.db2_di_re = x2_re;
                w.db2_di_im = x2_im;
                w.bf2_sp_re = d2_re;
                w.bf2_sp_im = d2_im;
            }

            // --------------------------------------- twiddle y producto
            if (q.has_mult) {
                const int sel = (bit_of(q.bf2_count.o, q.log_m - 2) ? 2 : 0)
                              | (bit_of(q.bf2_count.o, q.log_m - 1) ? 1 : 0);

                // tw_num truncado a LOG_N-2 bits (ver nota de cabecera)
                const int num_mask = (1 << (LOG_N - 2)) - 1;
                const int tw_num   = (q.bf2_count.o << (LOG_N - q.log_m)) & num_mask;

                w.tw_addr = (tw_num * sel) & ((1 << LOG_N) - 1);

                const double ar = double(q.bf2_do_re.o);
                const double ai = double(q.bf2_do_im.o);
                const double br = double(q.tw_re.o);
                const double bi = double(q.tw_im.o);

                w.mu_re = in_t(ar * br - ai * bi);
                w.mu_im = in_t(ar * bi + ai * br);
            }

            // salida de la unidad -> entrada de la siguiente (siempre registrada)
            if (q.has_mult) {
                din_re = q.mu_do_re.o;
                din_im = q.mu_do_im.o;
            }
            else {
                din_re = q.bf2_do_re.o;
                din_im = q.bf2_do_im.o;
            }
        }
    }

    std::vector<in_t>          tw_rom_re_;
    std::vector<in_t>          tw_rom_im_;

public:
    // Estado observable por el testbench del vm.
    std::array<Unit, N_UNITS>  unit_;
    std::array<Wires, N_UNITS> wire_;
    rtl::Reg<int>              r_out_count;
};

} // namespace fft1d_r22sdf
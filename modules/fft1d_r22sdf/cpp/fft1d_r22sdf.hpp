#pragma once

#include "rtlsim.hpp"

#include <ap_fixed.h>
#include <ap_int.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// fft1d_r22sdf.hpp -- modelo bit-exacto del RTL fft1d_r22sdf.v (radix-2^2 SDF).
//
// Espeja la estructura del RTL: una cadena de floor(log2(N)/2) unidades, cada una con
//
//   i_din -> BF2II (linea M/2, conmutador -j) -> reg
//         -> BF2I  (linea M/4)                -> reg
//         -> multiplicador de twiddle          -> reg -> o_dout
//
// y el control por etapa (cada etapa lleva su propio contador, que arranca
// cuando los datos efectivamente le llegan). Con log2(N) par la ultima unidad
// (M = 4) no multiplica. Con log2(N) impar (p. ej. 512) todas multiplican y se
// agrega una etapa radix-2 final de span 1 sin twiddle (Tail), como
// fft1d_r22sdf_tail.v.
//
// Puntos sensibles replicados del RTL (los que costaron encontrar):
//
//   * tw_num se TRUNCA a LOG_N-2 bits. Los dos bits altos del contador ya se
//     consumieron en tw_sel; dejarlos corre la direccion en N/2, y como
//     W^(k+N/2) = -W^k eso equivale a invertir el signo del twiddle.
//   * El conmutador -j actua solo en la fase de carga (bf = 0).
//   * bf2_bf y bf2_start estan REGISTRADOS, para alinearse con el registro de
//     pipeline que separa las dos mariposas.
//   * INVERSE = 1: swap(a + jb) = b + ja a la entrada y a la salida, igual que
//     el RTL. Asi el mismo nucleo calcula sum X[k] e^{+j2pi kn/N} (por 2^-S).
//   * CMUL_PIPE = 1: el cmul registra la suma completa ANTES del cast (mp_re,
//     mp_im) y el bypass de W^0, su seleccion y el valid se atrasan un ciclo
//     para seguir alineados. Los valores no cambian; la latencia sube un ciclo
//     por unidad con multiplicador.
//
// Aritmetica: el escalado x1/2 de la mariposa se hace sumando en precision
// exacta y cuantizando a data_t (AP_RND, AP_SAT), que es exactamente lo que hace
// el cast del RTL al bajar un bit fraccional. Con ROUND_CONV = 1 las mariposas
// que escalan y la salida cuantizan con AP_RND_CONV (empates al par), como el
// cast.v con ROUND_CONV = 1; el cmul sigue con AP_RND.
//
// Modo DOUBLE (-DDOUBLE): todos los tipos son double y los twiddles son exactos
// (se calculan, no se leen de TW_DIR). No hay ninguna cuantizacion, asi que
// separa los errores estructurales de los numericos. No es bit-exacto con el
// RTL: para eso esta el modo fixed.
//
// Parametros de compilacion y de elaboracion (desde el 2026-10-20):
//
//   * Los FORMATOS (NB, NBF, NB_TW, NBF_TW, NB_IN, NBF_IN, NB_OUT, NBF_OUT) son
//     de compilacion: definen los tipos ap_fixed y son comunes a todas las
//     instancias de un binario.
//   * N, SCALE_MASK, INVERSE, CMUL_PIPE, ROUND_CONV y TW_DIR son de ELABORACION: cada
//     instancia los recibe en un Params, como los parameter de un modulo
//     Verilog. Asi un mismo binario puede tener una FFT de 1024 y otra de 512
//     (la FFT2D). Params{} toma los valores de los macros FFT1D_R22SDF_*, y las
//     constantes del namespace (N, LATENCY, SCALE_MASK, ...) siguen
//     describiendo esa configuracion por defecto, para el tb y el gtest.
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

/*
 * Formato de los twiddles, independiente del datapath. Por defecto, el de los
 * datos; si se fija solo NB_TW, hereda los bits enteros del datapath (con un
 * datapath Q1.x, los twiddles quedan Q1.(NB_TW-1)).
 */
#ifndef FFT1D_R22SDF_NB_TW
#  define FFT1D_R22SDF_NB_TW FFT1D_R22SDF_NB
#endif
#ifndef FFT1D_R22SDF_NBF_TW
#  define FFT1D_R22SDF_NBF_TW (FFT1D_R22SDF_NB_TW - (FFT1D_R22SDF_NB - FFT1D_R22SDF_NBF))
#endif

/*
 * Formatos de los puertos. Mismo criterio que el RTL: si no se fijan los bits
 * fraccionales, el formato hereda la cantidad de bits enteros del datapath. Por
 * defecto, entrada y salida tienen el formato del datapath.
 */
#ifndef FFT1D_R22SDF_NB_IN
#  define FFT1D_R22SDF_NB_IN FFT1D_R22SDF_NB
#endif
#ifndef FFT1D_R22SDF_NBF_IN
#  define FFT1D_R22SDF_NBF_IN (FFT1D_R22SDF_NB_IN - (FFT1D_R22SDF_NB - FFT1D_R22SDF_NBF))
#endif
#ifndef FFT1D_R22SDF_NB_OUT
#  define FFT1D_R22SDF_NB_OUT FFT1D_R22SDF_NB
#endif
#ifndef FFT1D_R22SDF_NBF_OUT
#  define FFT1D_R22SDF_NBF_OUT (FFT1D_R22SDF_NB_OUT - (FFT1D_R22SDF_NB - FFT1D_R22SDF_NBF))
#endif

#ifndef FFT1D_R22SDF_SCALE_MASK
#define FFT1D_R22SDF_SCALE_MASK 0xFFFFFFFFu
#endif

#ifndef FFT1D_R22SDF_TW_DIR
#define FFT1D_R22SDF_TW_DIR twiddles
#endif

/*
 * Registro antes del cast de cada multiplicador (PREG del DSP48E2). Mismo
 * default que el RTL: 1 (desde el 2026-10-18). Con 0, combinacional.
 */
#ifndef FFT1D_R22SDF_CMUL_PIPE
#define FFT1D_R22SDF_CMUL_PIPE 1
#endif

/*
 * Transformada inversa con el mismo nucleo: swap real/imaginaria a la entrada y
 * a la salida. Mismo default que el RTL: 0 (FFT directa).
 */
#ifndef FFT1D_R22SDF_INVERSE
#define FFT1D_R22SDF_INVERSE 0
#endif

/*
 * Redondeo convergente (empates al par) en las mariposas que escalan y en el
 * cast de salida. Mismo default que el RTL: 0 (half-up, con sesgo).
 */
#ifndef FFT1D_R22SDF_ROUND_CONV
#define FFT1D_R22SDF_ROUND_CONV 0
#endif

#define FFT1D_R22SDF_STRINGIFY_IMPL(x) #x
#define FFT1D_R22SDF_STRINGIFY(x) FFT1D_R22SDF_STRINGIFY_IMPL(x)

constexpr std::uint32_t SCALE_MASK = FFT1D_R22SDF_SCALE_MASK;
constexpr int           CMUL_PIPE  = FFT1D_R22SDF_CMUL_PIPE;
constexpr int           INVERSE    = FFT1D_R22SDF_INVERSE;
constexpr int           ROUND_CONV = FFT1D_R22SDF_ROUND_CONV;
constexpr bool stage_scales(int s) { return ((SCALE_MASK >> s) & 1u) != 0u; }

enum parameters : int {
    N   = FFT1D_R22SDF_N,
    NB  = FFT1D_R22SDF_NB,
    NBF = FFT1D_R22SDF_NBF,

    NB_TW  = FFT1D_R22SDF_NB_TW,
    NBF_TW = FFT1D_R22SDF_NBF_TW,

    NB_IN   = FFT1D_R22SDF_NB_IN,
    NBF_IN  = FFT1D_R22SDF_NBF_IN,
    NB_OUT  = FFT1D_R22SDF_NB_OUT,
    NBF_OUT = FFT1D_R22SDF_NBF_OUT
};

constexpr int ct_log2(int n) { return (n <= 1) ? 0 : 1 + ct_log2(n / 2); }

/*
 * Latencia hasta el primer o_valid en funcion de log2(N) y CMUL_PIPE (ver el
 * comentario de LATENCY). La usan la configuracion por defecto y Params.
 */
constexpr int latency_of(int log_n, int cmul_pipe)
{
    return (log_n % 2 != 0)
        ? (1 << log_n) + (3 + cmul_pipe) * (log_n / 2) - 1
        : (1 << log_n) + (3 + cmul_pipe) * (log_n / 2 - 1);
}

enum localparameters : int {
    LOG_N   = ct_log2(N),
    LOG2N   = LOG_N,             // alias (nomenclatura de los otros modulos)
    N_UNITS  = LOG_N / 2,        // floor(log2(N)/2) unidades radix-2^2
    HAS_TAIL = LOG_N % 2,        // log2(N) impar: etapa radix-2 final

    /*
     * Latencia hasta el primer o_valid, contando desde la primera muestra de
     * entrada. El pipeline esta segmentado (registros entre las dos mariposas
     * de cada unidad y a la salida del multiplicador), asi que es bastante
     * mayor que la del R2SDF: 70 para N=64, 265 para N=256, 1036 para N=1024.
     * Con CMUL_PIPE = 1 cada unidad con multiplicador suma un ciclo: 72, 268 y
     * 1040. Con log2(N) impar todas las unidades multiplican y la etapa final
     * agrega su linea de 1 y su registro: N + (3 + CMUL_PIPE) * N_UNITS - 1,
     * o sea 527 para N = 512 con CMUL_PIPE = 1. Verificada contra el RTL.
     */
    LATENCY = HAS_TAIL ? N + (3 + CMUL_PIPE) * N_UNITS - 1
                       : N + (3 + CMUL_PIPE) * (N_UNITS - 1)
};

/*
 *   in_t   : puerto de entrada       Q(NB_IN,  NBF_IN)
 *   data_t : datapath interno        Q(NB,     NBF)
 *   out_t  : puerto de salida        Q(NB_OUT, NBF_OUT)
 *   tw_t   : twiddles                Q(NB_TW,  NBF_TW)
 *
 * in_t -> data_t es exacto (solo se permite ensanchar). data_t -> out_t
 * redondea y satura (AP_RND, AP_SAT), igual que el cast de salida del RTL.
 * data_conv_t y out_conv_t son los mismos formatos con AP_RND_CONV, para
 * ROUND_CONV = 1; convertirlos a data_t y out_t es exacto.
 */
#ifdef DOUBLE
using in_t        = double;
using data_t      = double;
using out_t       = double;
using tw_t        = double;
using data_conv_t = double;
using out_conv_t  = double;
#else
using in_t   = ap_fixed<NB_IN,  NB_IN  - NBF_IN,  AP_RND, AP_SAT>;
using data_t = ap_fixed<NB,     NB     - NBF,     AP_RND, AP_SAT>;
using out_t  = ap_fixed<NB_OUT, NB_OUT - NBF_OUT, AP_RND, AP_SAT>;
using tw_t   = ap_fixed<NB_TW,  NB_TW  - NBF_TW,  AP_RND, AP_SAT>;

using data_conv_t = ap_fixed<NB,     NB     - NBF,     AP_RND_CONV, AP_SAT>;
using out_conv_t  = ap_fixed<NB_OUT, NB_OUT - NBF_OUT, AP_RND_CONV, AP_SAT>;

/*
 * El producto complejo se evalua en double: es exacto solo si el producto de
 * un dato (NB bits) por un twiddle (NB_TW bits), mas el bit de la suma, entra
 * en los 53 bits de mantisa. Fuera de ese rango el modelo deja de ser
 * bit-exacto con el RTL. El DSP48E2 (27 x 18 = 45) queda adentro.
 */
static_assert(NB + NB_TW <= 52,
              "fft1d_r22sdf: NB + NB_TW > 52, el producto en double no es exacto");
#endif

// Las mismas restricciones que chequea el RTL.
static_assert(N == (1 << LOG_N) && LOG_N >= 2,
              "fft1d_r22sdf: N tiene que ser potencia de 2 y >= 4");
static_assert(NBF >= NBF_IN && (NB - NBF) >= (NB_IN - NBF_IN),
              "fft1d_r22sdf: la entrada no entra sin perdida en el datapath");
static_assert(NBF >= NBF_OUT,
              "fft1d_r22sdf: la salida no puede tener mas bits fraccionales que el datapath");
static_assert(CMUL_PIPE == 0 || CMUL_PIPE == 1,
              "fft1d_r22sdf: CMUL_PIPE tiene que ser 0 o 1");
static_assert(INVERSE == 0 || INVERSE == 1,
              "fft1d_r22sdf: INVERSE tiene que ser 0 o 1");
static_assert(ROUND_CONV == 0 || ROUND_CONV == 1,
              "fft1d_r22sdf: ROUND_CONV tiene que ser 0 o 1");

static_assert(LATENCY == latency_of(LOG_N, CMUL_PIPE),
              "fft1d_r22sdf: latency_of no coincide con LATENCY");

/*
 * Parametros de elaboracion de una instancia (ver cabecera). Los defaults son
 * los de los macros, asi que fft1d_r22sdf_model dut; es exactamente el modelo
 * de siempre.
 *
 * Las mismas restricciones que el RTL se chequean en validate(), que tira
 * std::invalid_argument (el RTL corta la elaboracion con $fatal). Los bits de
 * scale_mask por encima de log2(N)-1 se ignoran, como en el RTL.
 */
struct Params {
    int           n          = N;
    std::uint32_t scale_mask = SCALE_MASK;
    bool          inverse    = (INVERSE != 0);
    bool          cmul_pipe  = (CMUL_PIPE != 0);
    std::string   tw_dir     = FFT1D_R22SDF_STRINGIFY(FFT1D_R22SDF_TW_DIR);
    bool          round_conv = (ROUND_CONV != 0);

    int  log_n()    const { return ct_log2(n); }
    int  n_units()  const { return log_n() / 2; }
    bool has_tail() const { return (log_n() % 2) != 0; }
    int  latency()  const { return latency_of(log_n(), cmul_pipe ? 1 : 0); }

    bool stage_scales(int s) const { return ((scale_mask >> s) & 1u) != 0u; }

    // Cantidad de etapas que escalan: la salida es X[k] * 2^-scale_shift().
    int scale_shift() const
    {
        int s = 0;
        for (int k = 0; k < log_n(); ++k) {
            s += stage_scales(k) ? 1 : 0;
        }
        return s;
    }

    void validate() const
    {
        // 2^30 como techo: contadores, direcciones y tw_num * sel son int
        if (n < 4 || n > (1 << 30) || n != (1 << log_n())) {
            throw std::invalid_argument(
                "fft1d_r22sdf: N tiene que ser potencia de 2 y >= 4 (N = "
                + std::to_string(n) + ")");
        }
    }
};

using bit_t = rtl::bit_t;

class fft1d_r22sdf_model final : public rtl::Module {
public:
    /*
     * Elaboracion. Sin argumentos usa Params{} (los macros). La configuracion
     * queda fija: Simulator::add() llama a connect_clocks() con los vectores de
     * unidades ya dimensionados, y no se redimensionan nunca (ClockDomain
     * guarda punteros a los registros).
     */
    explicit fft1d_r22sdf_model(const Params& p = Params{})
        : p_(p)
    {
        p_.validate();

        n_         = p_.n;
        log_n_     = p_.log_n();
        n_units_   = p_.n_units();
        has_tail_  = p_.has_tail();
        inverse_   = p_.inverse;
        cmul_pipe_ = p_.cmul_pipe;
        round_conv_ = p_.round_conv;

        unit_.resize(n_units_);
        wire_.resize(n_units_);
    }

    const Params& params()  const { return p_; }
    int           latency() const { return p_.latency(); }

    /*
     * Ports
     */
    rtl::InPort<bit_t> i_valid;
    rtl::InPort<in_t>    i_re;
    rtl::InPort<in_t>    i_im;

    rtl::OutPort<bit_t> o_valid;
    rtl::OutPort<bit_t> o_last;
    rtl::OutPort<out_t> o_re;
    rtl::OutPort<out_t> o_im;

    void connect_clocks(rtl::ClockDomain& clk) override
    {
        for (int u = 0; u < n_units_; ++u) {
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
            if (cmul_pipe_) {
                clk.add(q.mp_re);
                clk.add(q.mp_im);
                clk.add(q.bf2_do_d_re);
                clk.add(q.bf2_do_d_im);
                clk.add(q.mu_en_d);
                clk.add(q.bf2_do_en_d);
            }
            clk.add(q.mu_do_re);
            clk.add(q.mu_do_im);
            clk.add(q.mu_do_en);
        }
        if (has_tail_) {
            clk.add(tail_.di_count);
            clk.add(tail_.sp_en);
            clk.add(tail_.count);
            clk.add(tail_.dl_re);
            clk.add(tail_.dl_im);
            clk.add(tail_.do_re);
            clk.add(tail_.do_im);
            clk.add(tail_.do_en);
        }
        clk.add(r_out_count);
    }

    void init() override
    {
        build_twiddle_rom();

        for (int u = 0; u < n_units_; ++u) {
            Unit& q = unit_[u];

            q.log_m    = log_n_ - 2 * u;
            q.depth1   = 1 << (q.log_m - 1);
            q.depth2   = 1 << (q.log_m - 2);
            q.has_mult = (q.log_m != 2);

            q.scale_bf2ii = p_.stage_scales(2*u);
            q.scale_bf2i  = p_.stage_scales(2*u + 1);

            q.dl1_re.assign(q.depth1, data_t(0));
            q.dl1_im.assign(q.depth1, data_t(0));
            q.dl2_re.assign(q.depth2, data_t(0));
            q.dl2_im.assign(q.depth2, data_t(0));

            q.di_count.set_initial_value(0);
            q.addr1.set_initial_value(0);
            q.addr2.set_initial_value(0);
            q.bf1_sp_en.set_initial_value(false);
            q.bf1_count.set_initial_value(0);
            q.bf1_do_re.set_initial_value(data_t(0));
            q.bf1_do_im.set_initial_value(data_t(0));
            q.bf2_bf.set_initial_value(false);
            q.bf2_start.set_initial_value(false);
            q.bf2_sp_en.set_initial_value(false);
            q.bf2_count.set_initial_value(0);
            q.bf2_do_re.set_initial_value(data_t(0));
            q.bf2_do_im.set_initial_value(data_t(0));
            q.bf2_do_en.set_initial_value(false);
            q.tw_re.set_initial_value(tw_t(0));
            q.tw_im.set_initial_value(tw_t(0));
            q.mu_en.set_initial_value(false);
            q.mp_re.set_initial_value(0.0);
            q.mp_im.set_initial_value(0.0);
            q.bf2_do_d_re.set_initial_value(data_t(0));
            q.bf2_do_d_im.set_initial_value(data_t(0));
            q.mu_en_d.set_initial_value(false);
            q.bf2_do_en_d.set_initial_value(false);
            q.mu_do_re.set_initial_value(data_t(0));
            q.mu_do_im.set_initial_value(data_t(0));
            q.mu_do_en.set_initial_value(false);
        }

        tail_.scale = p_.stage_scales(log_n_ - 1);   // ultima etapa
        tail_.di_count.set_initial_value(0);
        tail_.sp_en.set_initial_value(false);
        tail_.count.set_initial_value(0);
        tail_.dl_re.set_initial_value(data_t(0));
        tail_.dl_im.set_initial_value(data_t(0));
        tail_.do_re.set_initial_value(data_t(0));
        tail_.do_im.set_initial_value(data_t(0));
        tail_.do_en.set_initial_value(false);

        r_out_count.set_initial_value(0);

        o_valid = bit_t(false);
        o_last  = bit_t(false);
        o_re    = out_t(0);
        o_im    = out_t(0);
    }

    void combinational() override
    {
        evaluate();

        const bool v = fin_valid();

        o_valid = bit_t(v);
        o_last  = bit_t(v && (r_out_count.o == n_ - 1));

        /*
         * Gating de salida: con o_valid=0 el RTL entrega basura del pipeline sin
         * cebar (X en simulacion). Ambos lados fuerzan 0 para que el vm no
         * compare muestras invalidas.
         */
        if (!v) {
            o_re = out_t(0);
            o_im = out_t(0);
        }
        else {
            const data_t y_re = fin_re();
            const data_t y_im = fin_im();

            // INVERSE: swap a la salida (como en el RTL, despues del cast)
            o_re = to_out(inverse_ ? y_im : y_re);
            o_im = to_out(inverse_ ? y_re : y_im);
        }
    }

    void sequential() override
    {
        evaluate();

        bool  chain_valid = i_valid.value().to_bool();

        for (int u = 0; u < n_units_; ++u) {
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
            q.di_count.i = chain_valid ? ((q.di_count.o + 1) % n_) : 0;

            // --- control de la 1ra etapa ---
            const bool bf1_start = (q.di_count.o == q.depth1 - 1);
            const bool bf1_end   = (q.bf1_count.o == n_ - 1);
            q.bf1_sp_en.i = bf1_start ? true : (bf1_end ? false : q.bf1_sp_en.o);
            q.bf1_count.i = q.bf1_sp_en.o ? ((q.bf1_count.o + 1) % n_) : 0;

            q.bf1_do_re.i = w.bf1_sp_re;
            q.bf1_do_im.i = w.bf1_sp_im;

            // --- control de la 2da etapa (registrado) ---
            q.bf2_bf.i    = bit_of(q.bf1_count.o, q.log_m - 2);
            q.bf2_start.i = (q.bf1_count.o == q.depth2 - 1) && q.bf1_sp_en.o;

            const bool bf2_end = (q.bf2_count.o == n_ - 1);
            q.bf2_sp_en.i = q.bf2_start.o ? true : (bf2_end ? false : q.bf2_sp_en.o);
            q.bf2_count.i = q.bf2_sp_en.o ? ((q.bf2_count.o + 1) % n_) : 0;

            q.bf2_do_re.i = w.bf2_sp_re;
            q.bf2_do_im.i = w.bf2_sp_im;
            q.bf2_do_en.i = q.bf2_sp_en.o;

            // --- twiddle y multiplicador ---
            if (q.has_mult) {
                q.tw_re.i = tw_rom_re_[w.tw_addr];
                q.tw_im.i = tw_rom_im_[w.tw_addr];
                q.mu_en.i = (w.tw_addr != 0);

                if (cmul_pipe_) {
                    // registro interno del cmul: suma completa, antes del cast
                    q.mp_re.i = w.mp_re;
                    q.mp_im.i = w.mp_im;

                    // bypass, seleccion y valid atrasados el mismo ciclo
                    q.bf2_do_d_re.i = q.bf2_do_re.o;
                    q.bf2_do_d_im.i = q.bf2_do_im.o;
                    q.mu_en_d.i     = q.mu_en.o;
                    q.bf2_do_en_d.i = q.bf2_do_en.o;

                    q.mu_do_re.i = q.mu_en_d.o ? data_t(q.mp_re.o) : q.bf2_do_d_re.o;
                    q.mu_do_im.i = q.mu_en_d.o ? data_t(q.mp_im.o) : q.bf2_do_d_im.o;
                    q.mu_do_en.i = q.bf2_do_en_d.o;
                }
                else {
                    q.mu_do_re.i = q.mu_en.o ? data_t(w.mp_re) : q.bf2_do_re.o;
                    q.mu_do_im.i = q.mu_en.o ? data_t(w.mp_im) : q.bf2_do_im.o;
                    q.mu_do_en.i = q.bf2_do_en.o;
                }
            }

            // la entrada de la unidad siguiente es la salida (registrada) de esta
            chain_valid = q.has_mult ? q.mu_do_en.o : q.bf2_do_en.o;
        }

        // --- etapa radix-2 final (log2 N impar), mismo control que fft1d_r22sdf_tail.v ---
        if (has_tail_) {
            Tail& t = tail_;
            const bool start = chain_valid && (t.di_count.o == 0);   // con DEPTH = 1 exige valid
            const bool end   = (t.count.o == n_ - 1);

            t.di_count.i = chain_valid ? ((t.di_count.o + 1) % n_) : 0;
            t.sp_en.i    = start ? true : (end ? false : t.sp_en.o);
            t.count.i    = t.sp_en.o ? ((t.count.o + 1) % n_) : 0;

            t.dl_re.i = tail_wire_.db_di_re;   // linea de 1 = registro
            t.dl_im.i = tail_wire_.db_di_im;
            t.do_re.i = tail_wire_.sp_re;
            t.do_im.i = tail_wire_.sp_im;
            t.do_en.i = t.sp_en.o;
        }

        // --- contador de salida (para o_last) ---
        const bool v = fin_valid();
        r_out_count.i = v ? ((r_out_count.o + 1) % n_) : r_out_count.o;
    }

    // --- estado expuesto para trazado del vm ---
    struct Unit {
        int  log_m{0};
        int  depth1{1};
        int  depth2{1};
        bool has_mult{true};

        bool scale_bf2ii{true};
        bool scale_bf2i{true};

        std::vector<data_t> dl1_re, dl1_im;
        std::vector<data_t> dl2_re, dl2_im;

        rtl::Reg<int>  di_count;
        rtl::Reg<int>  addr1;
        rtl::Reg<int>  addr2;

        rtl::Reg<bool> bf1_sp_en;
        rtl::Reg<int>  bf1_count;
        rtl::Reg<data_t> bf1_do_re, bf1_do_im;

        rtl::Reg<bool> bf2_bf;
        rtl::Reg<bool> bf2_start;
        rtl::Reg<bool> bf2_sp_en;
        rtl::Reg<int>  bf2_count;
        rtl::Reg<data_t> bf2_do_re, bf2_do_im;
        rtl::Reg<bool> bf2_do_en;

        rtl::Reg<tw_t> tw_re, tw_im;
        rtl::Reg<bool> mu_en;

        // solo con CMUL_PIPE = 1 (cmul.v: r_ac_minus_bd/r_ad_plus_bc; unidad:
        // r_bf2_do_d, r_mu_en_d, r_bf2_do_en_d). La suma completa es exacta en
        // double (ver static_assert de NB + NB_TW).
        rtl::Reg<double> mp_re, mp_im;
        rtl::Reg<data_t> bf2_do_d_re, bf2_do_d_im;
        rtl::Reg<bool>   mu_en_d;
        rtl::Reg<bool>   bf2_do_en_d;

        rtl::Reg<data_t> mu_do_re, mu_do_im;
        rtl::Reg<bool> mu_do_en;
    };

    // Etapa radix-2 final (solo con log2(N) impar): fft1d_r22sdf_tail.v
    struct Tail {
        bool scale{true};
        rtl::Reg<int>    di_count;
        rtl::Reg<bool>   sp_en;
        rtl::Reg<int>    count;
        rtl::Reg<data_t> dl_re, dl_im;     // delay_line con DEPTH = 1 (un registro)
        rtl::Reg<data_t> do_re, do_im;     // registro de salida
        rtl::Reg<bool>   do_en;
    };

    struct TailWires {
        data_t sp_re{0}, sp_im{0};
        data_t db_di_re{0}, db_di_im{0};
    };

    // valores combinacionales de cada unidad, recalculados por evaluate()
    struct Wires {
        data_t bf1_sp_re{0}, bf1_sp_im{0};
        data_t db1_di_re{0}, db1_di_im{0};
        data_t bf2_sp_re{0}, bf2_sp_im{0};
        data_t db2_di_re{0}, db2_di_im{0};
        double mp_re{0}, mp_im{0};    // suma completa del cmul, antes del cast
        int  tw_addr{0};
    };

private:
    // Salida de la cadena: la etapa final si existe, si no la ultima unidad.
    bool fin_valid() const
    {
        if (has_tail_) return tail_.do_en.o;
        const Unit& last = unit_[n_units_ - 1];
        return last.has_mult ? last.mu_do_en.o : last.bf2_do_en.o;
    }
    data_t fin_re() const
    {
        if (has_tail_) return tail_.do_re.o;
        const Unit& last = unit_[n_units_ - 1];
        return last.has_mult ? last.mu_do_re.o : last.bf2_do_re.o;
    }
    data_t fin_im() const
    {
        if (has_tail_) return tail_.do_im.o;
        const Unit& last = unit_[n_units_ - 1];
        return last.has_mult ? last.mu_do_im.o : last.bf2_do_im.o;
    }

    static bool bit_of(int value, int pos)
    {
        return ((value >> pos) & 1) != 0;
    }

    /*
     * Escalado x1/2 de la mariposa. La suma/resta es exacta y la division por 2
     * tambien, asi que cuantizar el resultado a data_t (AP_RND, AP_SAT) reproduce
     * exactamente el cast del RTL, que baja un bit fraccional. Con ROUND_CONV,
     * se cuantiza con AP_RND_CONV (empates al par). Sin escalar no se descarta
     * ningun bit: solo satura, igual en los dos modos.
     */
    data_t butterfly_op(
        const data_t& a,
        const data_t& b,
        bool add,
        bool scale) const
    {
        const double s = add
            ? (double(a) + double(b))
            : (double(a) - double(b));

        if (!scale) {
            return data_t(s);
        }
        return round_conv_ ? data_t(data_conv_t(s * 0.5)) : data_t(s * 0.5);
    }

    // Cast de salida Q(NB, NBF) -> Q(NB_OUT, NBF_OUT), con el redondeo elegido.
    out_t to_out(const data_t& v) const
    {
        return round_conv_ ? out_t(out_conv_t(v)) : out_t(v);
    }

#ifdef DOUBLE
    /*
     * Modo double: twiddles EXACTOS, W^k = exp(-j 2 pi k / N), calculados en
     * double. No se leen los .mem (TW_DIR y NB_TW no se usan en este modo).
     *
     * Asi el modo double no tiene ninguna cuantizacion y aisla la estructura:
     * si falla, el error es de indice, signo o control, nunca de formato. La
     * cuantizacion de los twiddles se valida en el modo fixed, que lee los
     * mismos .mem que el RTL.
     *
     * k < N, asi que el argumento queda en [0, 2 pi) y cos/sin dan error del
     * orden de 1e-16.
     */
    void build_twiddle_rom()
    {
        constexpr double PI = 3.14159265358979323846;   // M_PI no es estandar (MSYS)

        tw_rom_re_.resize(n_);
        tw_rom_im_.resize(n_);

        for (int k = 0; k < n_; ++k) {
            const double angle = -2.0 * PI * static_cast<double>(k) / static_cast<double>(n_);
            tw_rom_re_[k] = std::cos(angle);
            tw_rom_im_[k] = std::sin(angle);
        }
    }
#else
    // Interpreta W bits crudos (complemento a 2) como un valor con F bits fraccionales.
    template <typename T, int W, int F>
    static T raw_to_fixed(uint64_t raw)
    {
        ap_uint<W> bits = raw;

        T value;
        value.range(W - 1, 0) = bits;

        return value;
    }

    static void load_twiddle_file(
        const std::string& filename,
        int n,
        std::vector<tw_t>& rom)
    {
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open twiddle file: " + filename);
        }

        rom.clear();
        rom.reserve(n);

        std::string token;

        while (file >> token) {
            // Permitir comentarios simples //
            if (token.rfind("//", 0) == 0) {
                std::string dummy;
                std::getline(file, dummy);
                continue;
            }

            const uint64_t raw = std::stoull(token, nullptr, 16);

            rom.push_back(raw_to_fixed<tw_t, NB_TW, NBF_TW>(raw));

            if (static_cast<int>(rom.size()) == n)
                break;
        }

        if (static_cast<int>(rom.size()) != n) {
            throw std::runtime_error(
                "Twiddle file " + filename +
                " contains " + std::to_string(rom.size()) +
                " entries; expected " + std::to_string(n));
        }
    }

    void build_twiddle_rom()
    {
        load_twiddle_file(
            p_.tw_dir + "/tw_re.mem",
            n_,
            tw_rom_re_);

        load_twiddle_file(
            p_.tw_dir + "/tw_im.mem",
            n_,
            tw_rom_im_);
    }
#endif

    /*
     * Recalcula toda la logica combinacional. Se llama tanto desde
     * combinational() como al principio de sequential(), para que los valores
     * de wire_ correspondan al estado actual de los registros.
     */
    void evaluate()
    {
        // in_t -> data_t: exacto. INVERSE: swap a la entrada.
        data_t din_re = data_t(inverse_ ? i_im.value() : i_re.value());
        data_t din_im = data_t(inverse_ ? i_re.value() : i_im.value());

        for (int u = 0; u < n_units_; ++u) {
            Unit&  q = unit_[u];
            Wires& w = wire_[u];

            // ---------------------------------------- 1ra etapa: BF2II
            const data_t d1_re = q.dl1_re[q.addr1.o % q.depth1];
            const data_t d1_im = q.dl1_im[q.addr1.o % q.depth1];

            const bool bf1_bf = bit_of(q.di_count.o, q.log_m - 1);
            const bool bf1_mj = (((q.bf1_count.o >> (q.log_m - 2)) & 0x3) == 0x3);

            const data_t y0_re = butterfly_op(d1_re, din_re, true,  q.scale_bf2ii);
            const data_t y0_im = butterfly_op(d1_im, din_im, true,  q.scale_bf2ii);
            const data_t y1_re = butterfly_op(d1_re, din_re, false, q.scale_bf2ii);
            const data_t y1_im = butterfly_op(d1_im, din_im, false, q.scale_bf2ii);

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
                    w.bf1_sp_im = data_t(-double(d1_re));
                }
                else {
                    w.bf1_sp_re = d1_re;
                    w.bf1_sp_im = d1_im;
                }
            }

            // ----------------------------------------- 2da etapa: BF2I
            const data_t d2_re = q.dl2_re[q.addr2.o % q.depth2];
            const data_t d2_im = q.dl2_im[q.addr2.o % q.depth2];

            const data_t x2_re = q.bf1_do_re.o;
            const data_t x2_im = q.bf1_do_im.o;

            const data_t z0_re = butterfly_op(d2_re, x2_re, true,  q.scale_bf2i);
            const data_t z0_im = butterfly_op(d2_im, x2_im, true,  q.scale_bf2i);
            const data_t z1_re = butterfly_op(d2_re, x2_re, false, q.scale_bf2i);
            const data_t z1_im = butterfly_op(d2_im, x2_im, false, q.scale_bf2i);

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
                const int num_mask = (1 << (log_n_ - 2)) - 1;
                const int tw_num   = (q.bf2_count.o << (log_n_ - q.log_m)) & num_mask;

                w.tw_addr = (tw_num * sel) & ((1 << log_n_) - 1);

                const double ar = double(q.bf2_do_re.o);
                const double ai = double(q.bf2_do_im.o);
                const double br = double(q.tw_re.o);
                const double bi = double(q.tw_im.o);

                // suma completa: el cast a data_t se hace donde se registra
                w.mp_re = ar * br - ai * bi;
                w.mp_im = ar * bi + ai * br;
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

        // ------------------------------- etapa radix-2 final (log2 N impar)
        if (has_tail_) {
            const Tail& t  = tail_;
            TailWires&  tw = tail_wire_;
            const bool  bf = bit_of(t.di_count.o, 0);
            const data_t d_re = t.dl_re.o;
            const data_t d_im = t.dl_im.o;

            if (bf) {
                tw.sp_re    = butterfly_op(d_re, din_re, true,  t.scale);
                tw.sp_im    = butterfly_op(d_im, din_im, true,  t.scale);
                tw.db_di_re = butterfly_op(d_re, din_re, false, t.scale);
                tw.db_di_im = butterfly_op(d_im, din_im, false, t.scale);
            }
            else {
                tw.sp_re    = d_re;
                tw.sp_im    = d_im;
                tw.db_di_re = din_re;
                tw.db_di_im = din_im;
            }
        }
    }

    std::vector<tw_t>          tw_rom_re_;
    std::vector<tw_t>          tw_rom_im_;

    // Configuracion de elaboracion (fija desde el constructor)
    Params p_;
    int    n_{0};
    int    log_n_{0};
    int    n_units_{0};
    bool   has_tail_{false};
    bool   inverse_{false};
    bool   cmul_pipe_{false};
    bool   round_conv_{false};

public:
    // Estado observable por el testbench del vm (n_units_ elementos).
    std::vector<Unit>          unit_;
    std::vector<Wires>         wire_;
    Tail                       tail_;
    TailWires                  tail_wire_;
    rtl::Reg<int>              r_out_count;
};

} // namespace fft1d_r22sdf
#pragma once
/*
 * ps_sense.hpp -- calculo de U'' en el PS (una vez por sesion).
 *
 * Es la version en C++ de model/unfold_ref.py, operacion por operacion: el
 * resultado tiene que ser IDENTICO bit a bit (los double de U, el exponente E y
 * el buffer de palabras de 32 bits). Por eso:
 *   - re / im separados y el mismo orden de sumas que la referencia;
 *   - 1/d y 1/g una sola vez, potencias de 2 con ldexp, redondeo con rint (al par);
 *   - compilar con -ffp-contract=off y sin -ffast-math (si el compilador junta
 *     a*b + c en un FMA el resultado cambia en el ultimo bit). build_buffer()
 *     lo chequea en tiempo de ejecucion (fp_contract_off()).
 *
 * Por posicion plegada (x, p), con S[l][a] = s_l(x, p + a*n2)   (L x Af):
 *
 *     A = S^H S + lam*I                       Af x Af hermitiana
 *     A = Lt D Lt^H                           Lt triangular inferior unitaria,
 *                                             D real > 0: sin raices
 *     U = A^-1 S^H                            Af x L, sustitucion por bobina
 *
 * y para la PL:
 *
 *     U'[a][l]  = U[a][l] * c_l,   c_l = (1/g_l) * (1/K)
 *     E         = exponente de frexp(max |Re U'|, |Im U'|)    (comun a la sesion)
 *     U''_q     = sat(rint(U' * 2^(nb-1-E)))                  Q1.(nb-1)
 *     palabra   = (im << 16) | (re & 0xFFFF)
 *
 * en el orden del stream de la fft2d: o = t1*n2 + t2 <-> (x = br(t1), p = br(t2)).
 *
 * Portable (C++17, sin excepciones, sin memoria dinamica, sin iostream): compila
 * igual para la PC, para el A53 bare-metal (aarch64-none-elf) y para Linux.
 */

#include <cstddef>
#include <cstdint>

#ifndef PS_SENSE_MAX_L
#define PS_SENSE_MAX_L 16
#endif

#ifndef PS_SENSE_MAX_AF
#define PS_SENSE_MAX_AF 4
#endif

namespace ps_sense {

// Limites de los arreglos de trabajo por posicion (van en la pila: con 16 y 4
// son unos 3 KiB). Se pueden cambiar con -DPS_SENSE_MAX_L / -DPS_SENSE_MAX_AF.
constexpr int MAX_L  = PS_SENSE_MAX_L;
constexpr int MAX_AF = PS_SENSE_MAX_AF;

static_assert(MAX_L >= 1 && MAX_L <= 64, "PS_SENSE_MAX_L fuera de rango");
static_assert(MAX_AF >= 1 && MAX_AF <= 16, "PS_SENSE_MAX_AF fuera de rango");

// --------------------------------------------------------------- tipos

enum class status : int {
    ok = 0,
    null_ptr,        // algun puntero obligatorio es nulo
    bad_dims,        // L, Af, n_ro o n_pe fuera de rango (o no potencia de 2)
    bad_param,       // g_l <= 0 o no finito, lam < 0, nb fuera de [2, 16], ...
    fp_contract_on,  // el compilador junta a*b + c en FMA: no seria bit exacto
};

const char* status_str(status s);

enum class layout : int {
    pos_coil_a = 0,  // indice ((o*L + l)*Af + a): todo lo de una posicion junto
    coil_pos_a = 1,  // indice ((l*T + o)*Af + a): una bobina entera por vez
};

struct dims {
    int L    = 8;     // bobinas
    int af   = 2;     // factor de aceleracion
    int n_ro = 1024;  // x: readout = N1 de la fft2d (potencia de 2)
    int n_pe = 1024;  // y: phase-encode completo; n2 = n_pe / af = N2 de la fft2d

    int n2() const { return n_pe / af; }
    std::size_t positions() const { return static_cast<std::size_t>(n_ro) * static_cast<std::size_t>(n2()); }
    std::size_t buffer_words() const { return positions() * static_cast<std::size_t>(af) * static_cast<std::size_t>(L); }
};

struct config {
    double lam      = 0.0;              // Tikhonov (absoluto: lam*I)
    double k_cordic = 0.0;              // ganancia del CORDIC a plegar; 0 = no plegar
    int    nb       = 16;               // bits de U''_q (Q1.(nb-1)), 2..16
    layout lay      = layout::pos_coil_a;
};

struct report {
    int           E         = 0;    // exponente comun
    double        max_abs   = 0.0;  // max |Re|, |Im| de U' (solo posiciones validas)
    double        d_min     = 0.0;  // pivote mas chico y mas grande (validas)
    double        d_max     = 0.0;
    std::uint32_t n_invalid = 0;    // posiciones con L D L^H invalida: U'' = 0
    std::uint32_t n_sat     = 0;    // componentes saturadas al cuantizar
};

// --------------------------------------------- nucleo por posicion (x, p)
//
// Layouts (fila mayor, como los ultimos ejes de la referencia):
//   S : Sr[l*af + a]        L x Af
//   A : Ar[i*af + j]        Af x Af, solo i >= j (lo de arriba queda en 0)
//   Lt: Lr[i*af + j]        Af x Af, triangular inferior con unos en la diagonal
//   U : Ur[a*L + l]         Af x L

// A = S^H S + lam*I. Suma secuencial en l; la diagonal es exactamente real.
void gram(const double* Sr, const double* Si, int L, int af, double lam,
          double* Ar, double* Ai);

// A = Lt D Lt^H por columnas. dinv = 1/d (una division por pivote). Si A no es
// definida positiva quedan d <= 0 o inf/nan: lo detecta ldlh_ok().
void ldlh(const double* Ar, const double* Ai, int af,
          double* Lr, double* Li, double* d, double* dinv);

// Todos los pivotes finitos y > 0.
bool ldlh_ok(const double* d, int af);

// U por sustitucion: por bobina, Lt z = conj(S[l][:]), w = dinv*z, Lt^H u = w.
void solve_u(const double* Lr, const double* Li, const double* dinv,
             const double* Sr, const double* Si, int L, int af,
             double* Ur, double* Ui);

// Todo junto (gram + ldlh + solve_u). Devuelve ldlh_ok(); d puede ser nulo.
bool unfold_position(const double* Sr, const double* Si, int L, int af, double lam,
                     double* Ur, double* Ui, double* d = nullptr);

// ------------------------------------------------------ mapas de entrada
//
// smaps: complejos double intercalados (re, im), indice [l][x][y], y = 0..n_pe-1.
// Es el layout en memoria de un np.ndarray complex128 smaps[l, x, y] contiguo.
// gather_s arma S[l][a] = smaps[l][x][a*n2 + p] de una posicion.
void gather_s(const double* smaps, const dims& dm, int x, int p, double* Sr, double* Si);

// ------------------------------------------------- escala y cuantizacion

// K_n = prod_{i<n} sqrt(1 + 2^-2i): ganancia del CORDIC con n iteraciones.
double cordic_gain(int n_it);

// c_l = (1/g_l) * (1/K); con k_cordic == 0, c_l = 1/g_l.
void coil_scales(const double* g, int L, double k_cordic, double* c);

// sat(rint(v * 2^shift)) a nb bits con signo. *sat (si no es nulo) se suma 1 si satura.
std::int32_t quantize(double v, int shift, int nb, std::uint32_t* sat = nullptr);

// {im[15:0], re[15:0]}, como los puertos de la FFT.
inline std::uint32_t pack_word(std::int32_t re, std::int32_t im)
{
    return (static_cast<std::uint32_t>(im & 0xFFFF) << 16) | static_cast<std::uint32_t>(re & 0xFFFF);
}

// ---------------------------------------------------------- indices

// log2 de n si es potencia de 2 (n >= 1), -1 si no.
int log2_exact(long n);

// i con sus 'bits' bits menos significativos invertidos.
std::uint32_t bitrev(std::uint32_t i, int bits);

// Indice de la palabra (o, l, a) en el buffer.
std::size_t word_index(const dims& dm, layout lay, std::size_t o, int l, int a);

// --------------------------------------------------------------- sesion

// Chequeos de dimensiones y parametros (los usa build_buffer).
status check_dims(const dims& dm);
status check_config(const dims& dm, const config& cfg, const double* g);

// true si el compilador NO junto a*b + c en un FMA en este archivo.
bool fp_contract_off();

/*
 * Calcula el buffer U'' completo de la sesion.
 *   smaps : mapas [l][x][y] complejos intercalados (ver gather_s)
 *   g     : g_l, L ganancias del k-space
 *   out   : dm.buffer_words() palabras (Af*L*n_ro*n2; 32 MiB con 1024 x 1024, L=8, Af=2)
 *   rep   : resultados de la sesion (puede ser nulo)
 *
 * Dos pasadas sin temporales grandes: la 1 calcula U' en todas las posiciones
 * para encontrar el maximo (y E); la 2 lo vuelve a calcular (da los mismos bits)
 * y lo cuantiza y empaqueta directo en out. Las posiciones invalidas (L D L^H no
 * definida positiva, solo posible con lam = 0) quedan en 0 y no cuentan para E.
 */
status build_buffer(const dims& dm, const double* smaps, const double* g,
                    const config& cfg, std::uint32_t* out, report* rep = nullptr);

}  // namespace ps_sense

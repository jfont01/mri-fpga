// ps_sense.cpp -- ver ps_sense.hpp. Espejo de model/unfold_ref.py: cada bucle
// lleva al lado la linea de la referencia a la que corresponde, y el orden de
// las sumas y productos es el mismo (no reordenar ni "simplificar" sin cambiar
// tambien la referencia: se pierde el bit a bit).
#include "ps_sense.hpp"

#include <cmath>

namespace ps_sense {

const char* status_str(status s)
{
    switch (s) {
    case status::ok:             return "ok";
    case status::null_ptr:       return "puntero nulo";
    case status::bad_dims:       return "dimensiones invalidas";
    case status::bad_param:      return "parametro invalido";
    case status::fp_contract_on: return "compilado con contraccion a FMA (usar -ffp-contract=off)";
    }
    return "?";
}

// ===================================================== nucleo por posicion

// unfold_ref.gram
void gram(const double* Sr, const double* Si, int L, int af, double lam,
          double* Ar, double* Ai)
{
    for (int k = 0; k < af * af; ++k) {
        Ar[k] = 0.0;
        Ai[k] = 0.0;
    }
    for (int i = 0; i < af; ++i) {
        for (int j = 0; j <= i; ++j) {
            double accr = 0.0;
            double acci = 0.0;
            for (int l = 0; l < L; ++l) {
                const double sri = Sr[l * af + i], sii = Si[l * af + i];
                const double srj = Sr[l * af + j], sij = Si[l * af + j];
                // conj(s_i) * s_j = (sri - j sii)(srj + j sij)
                accr = accr + (sri * srj + sii * sij);
                acci = acci + (sri * sij - sii * srj);
            }
            if (i == j) {
                accr = accr + lam;
                acci = 0.0;  // exactamente real
            }
            Ar[i * af + j] = accr;
            Ai[i * af + j] = acci;
        }
    }
}

// unfold_ref.ldlh
void ldlh(const double* Ar, const double* Ai, int af,
          double* Lr, double* Li, double* d, double* dinv)
{
    for (int k = 0; k < af * af; ++k) {
        Lr[k] = 0.0;
        Li[k] = 0.0;
    }
    for (int j = 0; j < af; ++j) {
        // d_j = A_jj - sum_{k<j} |L_jk|^2 d_k
        double acc = Ar[j * af + j];
        for (int k = 0; k < j; ++k) {
            const double lr = Lr[j * af + k], li = Li[j * af + k];
            acc = acc - (lr * lr + li * li) * d[k];
        }
        d[j] = acc;
        dinv[j] = 1.0 / acc;
        Lr[j * af + j] = 1.0;
        // L_ij = (A_ij - sum_{k<j} L_ik d_k conj(L_jk)) / d_j,  i > j
        for (int i = j + 1; i < af; ++i) {
            double accr = Ar[i * af + j];
            double acci = Ai[i * af + j];
            for (int k = 0; k < j; ++k) {
                const double ar = Lr[i * af + k] * d[k], ai = Li[i * af + k] * d[k];
                const double br = Lr[j * af + k], bi = -Li[j * af + k];
                accr = accr - (ar * br - ai * bi);
                acci = acci - (ar * bi + ai * br);
            }
            Lr[i * af + j] = accr * dinv[j];
            Li[i * af + j] = acci * dinv[j];
        }
    }
}

// unfold_ref.ldlh_ok
bool ldlh_ok(const double* d, int af)
{
    for (int j = 0; j < af; ++j) {
        if (!(std::isfinite(d[j]) && d[j] > 0.0)) {
            return false;
        }
    }
    return true;
}

// unfold_ref.solve_U
void solve_u(const double* Lr, const double* Li, const double* dinv,
             const double* Sr, const double* Si, int L, int af,
             double* Ur, double* Ui)
{
    double zr[MAX_AF], zi[MAX_AF], ur[MAX_AF], ui[MAX_AF];
    for (int l = 0; l < L; ++l) {
        for (int i = 0; i < af; ++i) {  // Lt z = b,  b = conj(S[l][:])
            double accr = Sr[l * af + i];
            double acci = -Si[l * af + i];
            for (int k = 0; k < i; ++k) {
                const double lr = Lr[i * af + k], li = Li[i * af + k];
                accr = accr - (lr * zr[k] - li * zi[k]);
                acci = acci - (lr * zi[k] + li * zr[k]);
            }
            zr[i] = accr;
            zi[i] = acci;
        }
        for (int i = 0; i < af; ++i) {  // w = D^-1 z (en el mismo lugar)
            zr[i] = zr[i] * dinv[i];
            zi[i] = zi[i] * dinv[i];
        }
        for (int i = af - 1; i >= 0; --i) {  // Lt^H u = w,  (Lt^H)_ik = conj(Lt_ki)
            double accr = zr[i];
            double acci = zi[i];
            for (int k = i + 1; k < af; ++k) {
                const double lr = Lr[k * af + i], li = -Li[k * af + i];
                accr = accr - (lr * ur[k] - li * ui[k]);
                acci = acci - (lr * ui[k] + li * ur[k]);
            }
            ur[i] = accr;
            ui[i] = acci;
        }
        for (int i = 0; i < af; ++i) {
            Ur[i * L + l] = ur[i];
            Ui[i * L + l] = ui[i];
        }
    }
}

bool unfold_position(const double* Sr, const double* Si, int L, int af, double lam,
                     double* Ur, double* Ui, double* d)
{
    double Ar[MAX_AF * MAX_AF], Ai[MAX_AF * MAX_AF];
    double Lr[MAX_AF * MAX_AF], Li[MAX_AF * MAX_AF];
    double dd[MAX_AF], dinv[MAX_AF];
    gram(Sr, Si, L, af, lam, Ar, Ai);
    ldlh(Ar, Ai, af, Lr, Li, dd, dinv);
    solve_u(Lr, Li, dinv, Sr, Si, L, af, Ur, Ui);
    if (d != nullptr) {
        for (int j = 0; j < af; ++j) {
            d[j] = dd[j];
        }
    }
    return ldlh_ok(dd, af);
}

// ======================================================= mapas de entrada

// unfold_ref.smaps_to_S para una posicion: y = a*n2 + p
void gather_s(const double* smaps, const dims& dm, int x, int p, double* Sr, double* Si)
{
    const std::size_t n_ro = static_cast<std::size_t>(dm.n_ro);
    const std::size_t n_pe = static_cast<std::size_t>(dm.n_pe);
    const std::size_t n2 = static_cast<std::size_t>(dm.n2());
    for (int l = 0; l < dm.L; ++l) {
        const std::size_t row = (static_cast<std::size_t>(l) * n_ro + static_cast<std::size_t>(x)) * n_pe;
        for (int a = 0; a < dm.af; ++a) {
            const std::size_t idx = 2 * (row + static_cast<std::size_t>(a) * n2 + static_cast<std::size_t>(p));
            Sr[l * dm.af + a] = smaps[idx];
            Si[l * dm.af + a] = smaps[idx + 1];
        }
    }
}

// =================================================== escala y cuantizacion

// unfold_ref.cordic_gain
double cordic_gain(int n_it)
{
    double k = 1.0;
    for (int i = 0; i < n_it; ++i) {
        k *= std::sqrt(1.0 + std::ldexp(1.0, -2 * i));
    }
    return k;
}

// unfold_ref.scale_quantize: c_l
void coil_scales(const double* g, int L, double k_cordic, double* c)
{
    const double inv_k = (k_cordic != 0.0) ? 1.0 / k_cordic : 1.0;
    for (int l = 0; l < L; ++l) {
        c[l] = (1.0 / g[l]) * inv_k;
    }
}

// unfold_ref.scale_quantize: clip(rint(ldexp(v, f - E)), lo, hi)
std::int32_t quantize(double v, int shift, int nb, std::uint32_t* sat)
{
    const double hi = std::ldexp(1.0, nb - 1) - 1.0;
    const double lo = -std::ldexp(1.0, nb - 1);
    double q = std::rint(std::ldexp(v, shift));
    if (q > hi) {
        q = hi;
        if (sat != nullptr) ++*sat;
    }
    else if (q < lo) {
        q = lo;
        if (sat != nullptr) ++*sat;
    }
    return static_cast<std::int32_t>(q);
}

// ================================================================ indices

int log2_exact(long n)
{
    if (n < 1 || (n & (n - 1)) != 0) {
        return -1;
    }
    int b = 0;
    while ((1L << b) < n) {
        ++b;
    }
    return b;
}

std::uint32_t bitrev(std::uint32_t i, int bits)
{
    std::uint32_t r = 0;
    for (int b = 0; b < bits; ++b) {
        r = (r << 1) | ((i >> b) & 1u);
    }
    return r;
}

// unfold_ref.pack_hw
std::size_t word_index(const dims& dm, layout lay, std::size_t o, int l, int a)
{
    const std::size_t L = static_cast<std::size_t>(dm.L);
    const std::size_t af = static_cast<std::size_t>(dm.af);
    if (lay == layout::coil_pos_a) {
        return (static_cast<std::size_t>(l) * dm.positions() + o) * af + static_cast<std::size_t>(a);
    }
    return (o * L + static_cast<std::size_t>(l)) * af + static_cast<std::size_t>(a);
}

// ================================================================= sesion

status check_dims(const dims& dm)
{
    if (dm.L < 1 || dm.L > MAX_L || dm.af < 1 || dm.af > MAX_AF) {
        return status::bad_dims;
    }
    if (dm.n_ro < 1 || dm.n_pe < dm.af || dm.n_pe % dm.af != 0) {
        return status::bad_dims;
    }
    // el orden del stream es bit-reversed en los dos ejes: potencias de 2
    const int b1 = log2_exact(dm.n_ro);
    const int b2 = log2_exact(dm.n2());
    if (b1 < 0 || b2 < 0 || b1 + b2 > 30) {
        return status::bad_dims;
    }
    return status::ok;
}

status check_config(const dims& dm, const config& cfg, const double* g)
{
    if (g == nullptr) {
        return status::null_ptr;
    }
    for (int l = 0; l < dm.L; ++l) {
        if (!(std::isfinite(g[l]) && g[l] > 0.0)) {
            return status::bad_param;
        }
    }
    if (!(std::isfinite(cfg.lam) && cfg.lam >= 0.0)) {
        return status::bad_param;
    }
    if (cfg.k_cordic != 0.0 && !(std::isfinite(cfg.k_cordic) && cfg.k_cordic > 0.0)) {
        return status::bad_param;
    }
    if (cfg.nb < 2 || cfg.nb > 16) {
        return status::bad_param;
    }
    if (cfg.lay != layout::pos_coil_a && cfg.lay != layout::coil_pos_a) {
        return status::bad_param;
    }
    return status::ok;
}

// a = 1 + 2^-30:  a*a = 1 + 2^-29 + 2^-60.  Redondeado (sin FMA) el 2^-60 se
// pierde y a*a - (1 + 2^-29) da 0; con FMA da 2^-60. volatile: que no lo
// resuelva el compilador.
bool fp_contract_off()
{
    volatile double va = 1.0 + 0x1p-30;
    volatile double vc = -(1.0 + 0x1p-29);
    const double a = va;
    const double c = vc;
    const double r = a * a + c;
    return r == 0.0;
}

status build_buffer(const dims& dm, const double* smaps, const double* g,
                    const config& cfg, std::uint32_t* out, report* rep)
{
    if (smaps == nullptr || g == nullptr || out == nullptr) {
        return status::null_ptr;
    }
    status st = check_dims(dm);
    if (st != status::ok) {
        return st;
    }
    st = check_config(dm, cfg, g);
    if (st != status::ok) {
        return st;
    }
    if (!fp_contract_off()) {
        return status::fp_contract_on;
    }

    const int L = dm.L;
    const int af = dm.af;
    const int n2 = dm.n2();
    const int b1 = log2_exact(dm.n_ro);
    const int b2 = log2_exact(n2);

    double c[MAX_L];
    coil_scales(g, L, cfg.k_cordic, c);

    double Sr[MAX_L * MAX_AF], Si[MAX_L * MAX_AF];
    double Ur[MAX_AF * MAX_L], Ui[MAX_AF * MAX_L];
    double d[MAX_AF];
    report r;
    bool any_valid = false;

    // ---- pasada 1: maximo de |Re U'|, |Im U'| (y pivotes) sobre las validas
    for (int x = 0; x < dm.n_ro; ++x) {
        for (int p = 0; p < n2; ++p) {
            gather_s(smaps, dm, x, p, Sr, Si);
            if (!unfold_position(Sr, Si, L, af, cfg.lam, Ur, Ui, d)) {
                ++r.n_invalid;
                continue;
            }
            for (int j = 0; j < af; ++j) {
                if (!any_valid || d[j] < r.d_min) r.d_min = d[j];
                if (!any_valid || d[j] > r.d_max) r.d_max = d[j];
                any_valid = true;
            }
            for (int a = 0; a < af; ++a) {
                for (int l = 0; l < L; ++l) {
                    const double vr = std::fabs(Ur[a * L + l] * c[l]);
                    const double vi = std::fabs(Ui[a * L + l] * c[l]);
                    if (vr > r.max_abs) r.max_abs = vr;
                    if (vi > r.max_abs) r.max_abs = vi;
                }
            }
        }
    }

    int E = 0;
    (void)std::frexp(r.max_abs, &E);  // max = m * 2^E, m en [0.5, 1)
    r.E = E;
    const int shift = (cfg.nb - 1) - E;

    // ---- pasada 2: en el orden del stream, cuantizar y empaquetar
    for (int t1 = 0; t1 < dm.n_ro; ++t1) {
        const int x = static_cast<int>(bitrev(static_cast<std::uint32_t>(t1), b1));
        for (int t2 = 0; t2 < n2; ++t2) {
            const int p = static_cast<int>(bitrev(static_cast<std::uint32_t>(t2), b2));
            const std::size_t o = static_cast<std::size_t>(t1) * static_cast<std::size_t>(n2)
                                + static_cast<std::size_t>(t2);
            gather_s(smaps, dm, x, p, Sr, Si);
            const bool valid = unfold_position(Sr, Si, L, af, cfg.lam, Ur, Ui, nullptr);
            for (int l = 0; l < L; ++l) {
                for (int a = 0; a < af; ++a) {
                    std::int32_t qr = 0;
                    std::int32_t qi = 0;
                    if (valid) {
                        qr = quantize(Ur[a * L + l] * c[l], shift, cfg.nb, &r.n_sat);
                        qi = quantize(Ui[a * L + l] * c[l], shift, cfg.nb, &r.n_sat);
                    }
                    out[word_index(dm, cfg.lay, o, l, a)] = pack_word(qr, qi);
                }
            }
        }
    }

    if (rep != nullptr) {
        *rep = r;
    }
    return status::ok;
}

}  // namespace ps_sense

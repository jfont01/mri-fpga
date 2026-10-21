// -----------------------------------------------------------------------------
// fft2d_dump.cpp -- corre el modelo ciclo a ciclo de la fft2d con un k-space
// leido de archivo y escribe la salida, para comparar bit a bit contra la
// cadena de Python (model/fft2d_check_full.py, que lo compila con -O2).
//
//   fft2d_dump --in K.bin --out A.bin --coils L [--mask1 M1] [--mask2 M2]
//
// K.bin: int32 little-endian [L][N2][N1][re, im], enteros crudos del formato de
//        entrada (Q1.15 con 16 bits).
// A.bin: int32 [L][N1 * N2][re, im] en el orden de salida del hardware.
// stdout: una linea "clave = valor" por resultado (latencia, ciclos, o_last).
// -----------------------------------------------------------------------------

#include "fft2d.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fft2d::in_t;
using fft2d::out_t;

template <class T, int W>
T from_raw(std::int32_t r)
{
    T t;
    t.range(W - 1, 0) = ap_uint<W>(static_cast<std::uint32_t>(r));
    return t;
}

template <class T, int W>
std::int32_t to_raw(const T& t)
{
    ap_int<W> b;
    b.range(W - 1, 0) = t.range(W - 1, 0);
    return static_cast<std::int32_t>(b.to_int());
}

} // namespace

int main(int argc, char** argv)
{
#ifdef DOUBLE
    std::cerr << "fft2d_dump: compilar sin -DDOUBLE (compara enteros crudos)\n";
    return 2;
#else
    try {
        std::string in_path, out_path;
        int coils = 1;
        fft2d::Params p;

        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if      (k == "--in")    in_path  = v;
            else if (k == "--out")   out_path = v;
            else if (k == "--coils") coils    = std::stoi(v);
            else if (k == "--mask1") p.pass1.scale_mask = static_cast<std::uint32_t>(std::stoul(v));
            else if (k == "--mask2") p.pass2.scale_mask = static_cast<std::uint32_t>(std::stoul(v));
            else throw std::runtime_error("argumento desconocido: " + k);
        }

        const long t = p.frame();
        std::vector<std::int32_t> k(static_cast<size_t>(2 * t * coils));
        {
            std::ifstream f(in_path, std::ios::binary);
            if (!f) throw std::runtime_error("no se puede abrir " + in_path);
            f.read(reinterpret_cast<char*>(k.data()), static_cast<std::streamsize>(k.size() * 4));
            if (f.gcount() != static_cast<std::streamsize>(k.size() * 4)) {
                throw std::runtime_error(in_path + ": tamano distinto de L * N1 * N2 complejos");
            }
        }

        rtl::ClockDomain   clk;
        fft2d::fft2d_model dut(p);
        rtl::Simulator     sim(clk);
        sim.add(dut);
        sim.init();

        std::vector<std::int32_t> a;
        a.reserve(k.size());
        const long n_in = t * coils;
        long first = -1, last_valid = -1, n_last = 0, gaps = 0, c = 0;

        for (; static_cast<long>(a.size()) < 2 * n_in && c < n_in + p.latency() + 2 * t; ++c) {
            const bool v = c < n_in;
            dut.i_valid = fft2d::bit_t(v);
            dut.i_re    = v ? from_raw<in_t, fft1d_r22sdf::NB_IN>(k[2 * c])     : in_t(0);
            dut.i_im    = v ? from_raw<in_t, fft1d_r22sdf::NB_IN>(k[2 * c + 1]) : in_t(0);
            sim.cycle();

            if (dut.o_valid.value().to_bool()) {
                if (first < 0) first = c;
                if (last_valid >= 0 && c != last_valid + 1) ++gaps;
                last_valid = c;
                a.push_back(to_raw<out_t, fft1d_r22sdf::NB_OUT>(dut.o_re.value()));
                a.push_back(to_raw<out_t, fft1d_r22sdf::NB_OUT>(dut.o_im.value()));
            }
            if (dut.o_last.value().to_bool()) ++n_last;
        }

        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size() * 4));

        std::cout << "n1 = " << p.n1() << "\n"
                  << "n2 = " << p.n2() << "\n"
                  << "coils = " << coils << "\n"
                  << "mask1 = " << p.pass1.scale_mask << "\n"
                  << "mask2 = " << p.pass2.scale_mask << "\n"
                  << "latency_formula = " << p.latency() << "\n"
                  << "first_valid = " << first << "\n"
                  << "last_valid = " << last_valid << "\n"
                  << "samples = " << a.size() / 2 << "\n"
                  << "output_gaps = " << gaps << "\n"
                  << "o_last = " << n_last << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "fft2d_dump: " << e.what() << "\n";
        return 1;
    }
#endif
}

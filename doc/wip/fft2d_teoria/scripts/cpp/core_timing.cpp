// core_timing.cpp -- chequeo de tiempos del nucleo fft1d_r22sdf (modelo C++
// ciclo a ciclo del repo) con huecos al azar ENTRE tramas: para cada trama,
// el primer o_valid sale exactamente LATENCY ciclos despues de su primera
// entrada y las N salidas son contiguas. Incluye la ultima trama seguida de
// nada (la salida se vacia sola, sin que entre otra trama).
// Se corre desde modules/fft2d (rutas de twiddles de fft2d::Params).
#include "fft2d.hpp"
#include <cstdio>
#include <random>
#include <vector>

static int run(const fft1d_r22sdf::Params& p, int frames, int max_gap, unsigned seed)
{
    rtl::ClockDomain clk;
    fft1d_r22sdf::fft1d_r22sdf_model dut(p);
    rtl::Simulator sim(clk);
    sim.add(dut);
    sim.init();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gap(0, max_gap);
    const int n = p.n;
    // agenda de entrada: trama f arranca en start[f]
    std::vector<long> start(frames);
    long c = 0;
    for (int f = 0; f < frames; ++f) { c += (f == 0 ? 3 : gap(rng)); start[f] = c; c += n; }
    const long end = c + p.latency() + 4 * n;
    std::vector<long> out_cycles;
    int f = 0; long k = 0;
    for (long cyc = 0; cyc < end; ++cyc) {
        bool v = false;
        if (f < frames && cyc >= start[f]) { v = true; if (++k == n) { k = 0; ++f; } }
        dut.i_valid = fft1d_r22sdf::bit_t(v);
        dut.i_re = fft1d_r22sdf::in_t(v ? 0.25 : 0.0);
        dut.i_im = fft1d_r22sdf::in_t(0.0);
        sim.cycle();
        if (dut.o_valid.value().to_bool()) out_cycles.push_back(cyc);
    }
    int bad = 0;
    if ((long)out_cycles.size() != (long)frames * n) { std::printf("  salidas %zu != %ld\n", out_cycles.size(), (long)frames * n); return 1; }
    for (int g = 0; g < frames; ++g) {
        long first = out_cycles[(size_t)g * n];
        if (first - start[g] != p.latency()) ++bad;
        for (int i = 1; i < n; ++i) if (out_cycles[(size_t)g * n + i] != first + i) { ++bad; break; }
    }
    std::printf("  N=%d latency=%ld frames=%d max_gap=%d seed=%u: %s\n", n, p.latency(), frames, max_gap, seed,
                bad ? "FALLA" : "OK (o_valid = entrada + LATENCY, N salidas contiguas por trama)");
    return bad ? 1 : 0;
}

int main()
{
    fft2d::Params P;
    int bad = 0;
    for (unsigned s = 1; s <= 3; ++s) {
        bad += run(P.pass1, 12, s == 1 ? 0 : (s == 2 ? 5 : 3000), s);
        bad += run(P.pass2, 12, s == 1 ? 0 : (s == 2 ? 5 : 3000), s);
    }
    std::printf(bad ? "== FALLA\n" : "== OK\n");
    return bad ? 1 : 0;
}

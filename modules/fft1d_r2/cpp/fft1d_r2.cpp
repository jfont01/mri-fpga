#include "fft1d_r2.hpp"

#include <cmath>

namespace fft1d_r2 {

int fft1d_r2_model::bit_reverse(int x, int bits)
{
    int r = 0;

    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }

    return r;
}

void fft1d_r2_model::connect_clocks(rtl::ClockDomain& clk)
{
    clk.add(r_state);
    clk.add(r_count);
    clk.add(r_stage);
    clk.add(r_btfly);
}

void fft1d_r2_model::init()
{
    i_valid = false;
    i_re    = in_t(0);
    i_im    = in_t(0);

    o_valid = false;
    o_last  = false;
    o_re    = in_t(0);
    o_im    = in_t(0);

    r_state.set_initial_value(LOADING);
    r_count.set_initial_value(0);
    r_stage.set_initial_value(0);
    r_btfly.set_initial_value(0);

    for (int i = 0; i < N; ++i) {
        mem_re_[i] = in_t(0);
        mem_im_[i] = in_t(0);
    }

    /*
     * ROM de twiddles: W_N^{-k} = exp(-j*2*pi*k/N), k=0..N/2-1.
     * Para IFFT en vez de FFT, invertir el signo de "angle".
     */
    for (int k = 0; k < NH; ++k) {
        const double angle = -2.0 * PI * static_cast<double>(k) / static_cast<double>(N);
        twiddle_re_[k] = in_t(std::cos(angle));
        twiddle_im_[k] = in_t(std::sin(angle));
    }
}

void fft1d_r2_model::combinational()
{
    const bool in_output = (r_state.o == OUTPUT);

    o_valid = bit_t(in_output);
    o_last  = bit_t(in_output && (r_count.o == N - 1));
    o_re    = in_output ? mem_re_[r_count.o] : in_t(0);
    o_im    = in_output ? mem_im_[r_count.o] : in_t(0);
}

void fft1d_r2_model::sequential()
{
    switch (r_state.o) {

    case LOADING: {
        if (i_valid.value()) {
            const int addr = bit_reverse(r_count.o, LOG2N);
            mem_re_[addr] = i_re.value();
            mem_im_[addr] = i_im.value();

            if (r_count.o == N - 1) {
                r_state.i = COMPUTE;
                r_count.i = 0;
                r_stage.i = 0;
                r_btfly.i    = 0;
            }
            else {
                r_state.i = LOADING;
                r_count.i = r_count.o + 1;
            }
        }
        else {
            // sin backpressure real: si i_valid=0, simplemente esperamos
            r_state.i = LOADING;
            r_count.i = r_count.o;
        }
        break;
    }

    case COMPUTE: {
        const int half = 1 << r_stage.o;
        const int m    = half << 1;
        const int block = r_btfly.o / half;
        const int pos   = r_btfly.o % half;
        const int idx1  = block * m + pos;
        const int idx2  = idx1 + half;
        const int tw_idx = pos * (N / m);

        const in_t tw_re = twiddle_re_[tw_idx];
        const in_t tw_im = twiddle_im_[tw_idx];

        const in_t a_re = mem_re_[idx1];
        const in_t a_im = mem_im_[idx1];
        const in_t b_re = mem_re_[idx2];
        const in_t b_im = mem_im_[idx2];

        // multiplicacion compleja: t = tw * b
        const in_t t_re = tw_re * b_re - tw_im * b_im;
        const in_t t_im = tw_re * b_im + tw_im * b_re;

        // mariposa + escalado (>>1) para no desbordar [-1,1)
        mem_re_[idx1] = (a_re + t_re) * in_t(0.5);
        mem_im_[idx1] = (a_im + t_im) * in_t(0.5);
        mem_re_[idx2] = (a_re - t_re) * in_t(0.5);
        mem_im_[idx2] = (a_im - t_im) * in_t(0.5);

        if (r_btfly.o == NH - 1) {
            r_btfly.i = 0;

            if (r_stage.o == LOG2N - 1) {
                r_state.i = OUTPUT;
                r_count.i = 0;
                r_stage.i = 0;
            }
            else {
                r_state.i = COMPUTE;
                r_stage.i = r_stage.o + 1;
            }
        }
        else {
            r_state.i = COMPUTE;
            r_btfly.i    = r_btfly.o + 1;
            r_stage.i = r_stage.o;
        }
        break;
    }

    case OUTPUT: {
        if (r_count.o == N - 1) {
            r_state.i = LOADING;
            r_count.i = 0;
        }
        else {
            r_state.i = OUTPUT;
            r_count.i = r_count.o + 1;
        }
        break;
    }

    default: {
        r_state.i = LOADING;
        r_count.i = 0;
        break;
    }
    }
}

} // namespace fft1d_r2
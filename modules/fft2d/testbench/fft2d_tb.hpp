#pragma once

#include "../cpp/fft2d.hpp"

#include <filesystem>

namespace fft2d_tb {

struct tb_args {
    std::filesystem::path case_dir {"."};
    int n_cycles {0};        // 0: automatico (latencia + N_COILS * T + margen)
};

tb_args parse_args(int argc, char** argv);
void run_tb(const tb_args& args);

} // namespace fft2d_tb

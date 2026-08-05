#pragma once

#include "../cpp/tile_transpose.hpp"

#include <filesystem>

namespace tile_transpose_tb {

struct tb_args {
    std::filesystem::path case_dir {"."};
    int n_cycles {16};
};

tb_args parse_args(int argc, char** argv);
void run_tb(const tb_args& args);

} // namespace tile_transpose_tb

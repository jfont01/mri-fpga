#pragma once

#include "cast.hpp"

#include <filesystem>

namespace cast_tb {

struct tb_args {
    std::filesystem::path case_dir {"."};
    int n_cycles {0};
};

tb_args parse_args(int argc, char** argv);

void run_tb(const tb_args& args);

} // namespace cast_tb
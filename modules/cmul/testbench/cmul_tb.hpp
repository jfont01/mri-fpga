#pragma once

#include "cmul.hpp"

#include <filesystem>

namespace cmul_tb {

struct tb_args {
    std::filesystem::path case_dir {"."};
    int n_cycles {0};
};

tb_args parse_args(int argc, char** argv);

void run_tb(const tb_args& args);

} // namespace cmul_tb
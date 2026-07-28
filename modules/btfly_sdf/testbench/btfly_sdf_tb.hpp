#pragma once

#include "../cpp/btfly_sdf.hpp"

#include <filesystem>

namespace btfly_sdf_tb {

struct tb_args {
    std::filesystem::path case_dir {"."};
    int n_cycles {16};
};

tb_args parse_args(int argc, char** argv);
void run_tb(const tb_args& args);

} // namespace btfly_sdf_tb

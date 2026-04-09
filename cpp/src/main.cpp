#include <iostream>
#include <string>
#include <filesystem>

#include "load_stim.h"
#include "main.h"
#include "Aij.h"
#include "helpers.h"
#include "A.h"
#include "config_loader.h"

using std::string;
using std::vector;


int main() {

    string S_npy_path = get_S_npy_path();
    string y_npy_path = get_y_npy_path();
    string A_dat_path = get_A_dat_path();

    ComplexTensor y_f = load_complex_npy(y_npy_path);
    auto y_q = quantize_complex_tensor<NB_Y, NBI_Y>(y_f);

    ComplexTensor S_f = load_complex_npy(S_npy_path);
    auto S_q = quantize_complex_tensor<NB_S, NBI_S>(S_f);

    vector<size_t> A_shape;
    auto A_q = compute_A<NB_S, NBI_S, NB_A, NBI_A>(S_q, S_f.shape, A_shape);

    save_A_hex_dat(A_q, A_shape, A_dat_path);

    return 0;


}
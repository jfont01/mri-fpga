#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include "cnpy/cnpy.h"

template<typename T>
void print_shape(const std::vector<T>& shape) {
    std::cout << "shape = (";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " archivo.npy\n";
        return 1;
    }

    const std::string path = argv[1];

    cnpy::NpyArray arr = cnpy::npy_load(path);

    std::cout << "word_size = " << arr.word_size << "\n";
    std::cout << "fortran_order = " << arr.fortran_order << "\n";
    print_shape(arr.shape);

    // Caso típico: complejo128 de NumPy
    if (arr.word_size == sizeof(std::complex<double>)) {
        std::complex<double>* data = arr.data<std::complex<double>>();

        size_t total = 1;
        for (size_t d : arr.shape) total *= d;

        std::cout << "dtype asumido: complex128\n";
        std::cout << "total elems = " << total << "\n";

        size_t nprint = std::min<size_t>(total, 8);
        for (size_t i = 0; i < nprint; i++) {
            std::cout << "data[" << i << "] = " << data[i] << "\n";
        }
    } else if (arr.word_size == sizeof(double)) {
        double* data = arr.data<double>();

        size_t total = 1;
        for (size_t d : arr.shape) total *= d;

        std::cout << "dtype asumido: float64\n";
        std::cout << "total elems = " << total << "\n";

        size_t nprint = std::min<size_t>(total, 8);
        for (size_t i = 0; i < nprint; i++) {
            std::cout << "data[" << i << "] = " << data[i] << "\n";
        }
    } else {
        std::cout << "word_size no manejado en este ejemplo.\n";
    }

    return 0;
}
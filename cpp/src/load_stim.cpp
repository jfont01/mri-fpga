#include "load_stim.h"
#include "cnpy.h"

using namespace std;
using namespace cnpy;

size_t numel_from_shape(const vector<size_t>& shape) {
    size_t total = 1;
    for (size_t d : shape) total *= d;
    return total;
}

ComplexTensor load_complex_npy(const string& path) {
    NpyArray arr = npy_load(path);

    ComplexTensor out;
    out.shape = arr.shape;

    cout << "Loaded file: " << path << "\n";
    cout << "word_size = " << arr.word_size << "\n";
    print_shape(arr.shape);

    if (arr.word_size == sizeof(complex<double>)) {
        complex<double>* data_ptr = arr.data<complex<double>>();
        size_t total = numel_from_shape(arr.shape);

        out.data.assign(data_ptr, data_ptr + total);

        cout << "dtype asumido: complex128\n";
        cout << "total elems = " << total << "\n";
    }
    else if (arr.word_size == sizeof(double)) {
        double* data_ptr = arr.data<double>();
        size_t total = numel_from_shape(arr.shape);

        out.data.resize(total);
        for (size_t i = 0; i < total; i++) {
            out.data[i] = complex<double>(data_ptr[i], 0.0);
        }

        cout << "dtype asumido: float64\n";
        cout << "total elems = " << total << "\n";
    }
    else {
        throw runtime_error("dtype no soportado en este ejemplo");
    }

    return out;
}
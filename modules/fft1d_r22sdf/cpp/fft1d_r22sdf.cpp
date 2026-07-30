#include "fft1d_r22sdf.hpp"

// -----------------------------------------------------------------------------
// fft1d_r22sdf.cpp
//
// El modelo fft1d_r22sdf_model esta definido inline en el header (metodos y
// recursion radix-2^2). Esta unidad de traduccion existe para (a) dar un punto
// de compilacion del modelo consistente con el resto de los modulos, y (b)
// alojar cualquier tabla o helper no-inline que se agregue mas adelante (por
// ejemplo, cuando se cierre el modelo bit-exacto ciclo-a-ciclo del pipeline SDF
// y haga falta estado adicional).
//
// Ver la NOTA DE ESTADO en fft1d_r22sdf.hpp: hoy este modelo es el golden
// ALGORITMICO (FFT radix-2^2 verificada vs DFT), no el bit-exacto del RTL.
// -----------------------------------------------------------------------------

namespace fft1d_r22sdf {

// Sin definiciones fuera de linea por ahora: el modelo es header-only inline.

} // namespace fft1d_r22sdf
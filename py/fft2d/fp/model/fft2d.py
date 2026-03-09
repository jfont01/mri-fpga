import numpy as np
from matplotlib import pyplot as plt

def dft(x):
    """
    DFT directa:
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j 2 pi k n / N)
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    n = np.arange(N)                  # [0, 1, ..., N-1]
    k = n.reshape((N, 1))             # columna
    W = np.exp(-2j * np.pi * k * n / N)  # matriz NxN de twiddles
    X = W @ x                         # producto matricial
    return X

def fft_radix2(x):
    """
    FFT radix-2 recursiva en punto flotante.
    Requiere que N sea potencia de 2.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N == 1:
        return x
    if (N & (N - 1)) != 0:
        raise ValueError("La longitud N debe ser potencia de 2")

    # Separar pares e impares
    X_even = fft_radix2(x[0::2])
    X_odd  = fft_radix2(x[1::2])

    # Twiddles
    k = np.arange(N // 2)
    W_N = np.exp(-2j * np.pi * k / N)

    # Butterflies
    upper = X_even + W_N * X_odd
    lower = X_even - W_N * X_odd

    return np.concatenate([upper, lower])





if __name__ == "__main__":
    for N in [8, 16, 32]:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        X_fft = fft_radix2(x)
        X_np  = np.fft.fft(x)
        err = np.max(np.abs(X_fft - X_np))
        print(f"N={N}, error máximo = {err:e}")


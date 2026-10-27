import numpy as np
rng = np.random.default_rng(1)
def crand(*s): return rng.standard_normal(s) + 1j*rng.standard_normal(s)
swap = lambda z: z.imag + 1j*z.real

# (a) IFFT con el mismo nucleo: swap(FFT(swap(X))) = N * IFFT(X)
for N in (512, 1024):
    X = crand(N)
    print(f"(a) N={N}: max|swap(FFT(swap X)) - N*ifft(X)| =", np.abs(swap(np.fft.fft(swap(X))) - N*np.fft.ifft(X)).max())

# (b) R2^2 DIF a nivel de arreglo, con el direccionamiento de twiddles del RTL,
#     generalizado a log2N impar con una etapa radix-2 final (span 1, sin twiddle)
def bitrev(i, b): return int(format(i, f'0{b}b')[::-1], 2)
def r22_dif(x, N):
    L = N.bit_length()-1; y = x.astype(complex).copy()
    W = np.exp(-2j*np.pi*np.arange(N)/N)
    LOG_M = L
    while LOG_M >= 2:
        M = 1 << LOG_M
        for b0 in range(0, N, M):
            blk = y[b0:b0+M].copy()
            # BF2II (primera): span M/2, despues -j en el cuarto i in [3M/4, M)
            a, c = blk[:M//2].copy(), blk[M//2:].copy()
            blk[:M//2], blk[M//2:] = a + c, a - c
            blk[3*M//4:] *= -1j
            # BF2I (segunda): span M/4 dentro de cada mitad
            for h in (0, M//2):
                a, c = blk[h:h+M//4].copy(), blk[h+M//4:h+M//2].copy()
                blk[h:h+M//4], blk[h+M//4:h+M//2] = a + c, a - c
            # multiplicador (si no es la ultima unidad M = 4)
            if LOG_M != 2:
                for i in range(M):
                    q_hi, q_lo = (i >> (LOG_M-1)) & 1, (i >> (LOG_M-2)) & 1
                    sel = 2*q_lo + q_hi                      # {bit LOG_M-2, bit LOG_M-1}
                    num = ((i % (M//4)) << (L - LOG_M)) & ((1 << (L-2)) - 1)
                    blk[i] *= W[(num*sel) % N]
            y[b0:b0+M] = blk
        LOG_M -= 2
    if LOG_M == 1:   # log2N impar: etapa radix-2 final, span 1
        a, c = y[0::2].copy(), y[1::2].copy()
        y[0::2], y[1::2] = a + c, a - c
    return y   # salida en orden bit-reversed
for N in (8, 32, 64, 128, 512, 1024, 2048):
    x = crand(N); L = N.bit_length()-1
    y = r22_dif(x, N); nat = np.empty_like(y)
    for j in range(N): nat[bitrev(j, L)] = y[j]
    print(f"(b) N={N:5d} (log2N={L}): max|R2^2 - fft| = {np.abs(nat - np.fft.fft(x)).max():.2e}"
          f"   multiplicadores = {sum(1 for lm in range(L, 1, -2) if lm != 2)}")

# (c) fftshift como permutacion o como modulacion (-1)^n, N multiplo de 4
for N in (512, 1024):
    X = crand(N); n = np.arange(N); s = (-1.0)**n
    ref = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X)))
    print(f"(c) N={N}: modulacion (-1)^k entrada y (-1)^n salida: {np.abs(s*np.fft.ifft(s*X) - ref).max():.2e}")

# (d) 2D separable: IFFT por filas (readout 1024) y despues por columnas (PE 512);
#     el orden bit-reversed de cada pasada es solo una permutacion de indices
K = crand(512, 1024)                    # [ky, kx]: 512 lineas PE x 1024 readout
p1 = np.fft.ifft(K, axis=1)             # pasada 1: filas de 1024
p2 = np.fft.ifft(p1, axis=0)            # pasada 2: columnas de 512
print("(d) separabilidad:", np.abs(p2 - np.fft.ifft2(K)).max())
br1 = np.array([bitrev(i, 10) for i in range(1024)]); br2 = np.array([bitrev(i, 9) for i in range(512)])
p1b = p1[:, br1]                        # salida de la pasada 1 en bit-reversed por columna
p2b = np.fft.ifft(p1b, axis=0)[br2, :]  # la pasada 2 opera columna por columna: el orden de columnas no importa
print("(d) desordenar columnas antes de la pasada 2 y ordenar al final:", np.abs(p2b[np.argsort(br2)][:, np.argsort(br1)] - p2).max())

# (e) numeros de presupuesto y memoria
print("(e) muestras por pasada por bobina:", 512*1024, " ciclos -> ms a 200 MHz:", 512*1024/200e6*1e3)
print("    8 bobinas x 2 pasadas, 1 nucleo: ms =", 8*2*512*1024/200e6*1e3)
print("    intermedio de una bobina (16b complejo):", 512*1024*4/2**20, "MiB;  64 URAM288:", 64*288*1024/8/2**20, "MiB")
import math
import numpy as np

def dft(x, debug=False):
    total_ops = {"cmul": 0, "caddsub": 0, "mul": 0, "addsub": 0}

    N = len(x)
    X = []
    for k in range(N):
        acc = 0 + 0j
        first = True  # para no contar suma compleja en el primer término

        for n in range(N):
            # twiddle en float
            W = complex(
                math.cos(-2 * math.pi * k * n / N),
                math.sin(-2 * math.pi * k * n / N),
            )

            # multiplicación compleja x[n] * W
            prod = x[n] * W
            total_ops["cmul"]   += 1
            total_ops["mul"]    += 4   # 4 mul reales por cmul
            total_ops["addsub"] += 2   # 2 sumas/restas reales dentro de la cmul

            # acumulación acc += prod
            if first:
                acc = prod
                first = False
            else:
                acc += prod
                total_ops["caddsub"] += 1
                total_ops["addsub"]  += 2  # 2 sumas reales por cadd/sub

        X.append(acc)

    # OJO: el return va fuera del for k
    if debug:
        return X, total_ops
    else:
        return X


def np_fft(x):
    arr = np.array(x, dtype=np.complex128)
    X = np.fft.fft(arr)
    return list(X)

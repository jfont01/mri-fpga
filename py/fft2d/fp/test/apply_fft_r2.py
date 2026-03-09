import numpy as np
from matplotlib import pyplot as plt
from fp.model.fft2d import fft_radix2

img_path = "fp/sin_2d.npy"
sin_2d = np.load(img_path)


img = np.asarray(sin_2d, dtype=np.complex128)
M, N = img.shape

# 1) FFT 1D por filas
fft_1d_rows = np.zeros_like(img, dtype=np.complex128)
for i in range(M):
    fft_1d_rows[i, :] = fft_radix2(img[i, :])

# 2) FFT 1D por columnas
fft_1d_columns = np.zeros_like(fft_1d_rows, dtype=np.complex128)
for j in range(N):
    fft_1d_columns[:, j] = fft_radix2(fft_1d_rows[:, j])
  
fft2d = fft_1d_columns

Fshift = np.fft.fftshift(fft2d)
mag = np.abs(Fshift)

plt.imshow(np.log1p(mag), cmap='jet')
plt.title("Espectro de amplitudes (radix 2)")
plt.axis("off")
plt.show()

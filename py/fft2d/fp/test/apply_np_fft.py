import numpy as np
from matplotlib import pyplot as plt

img_path = "fp/sin_2d.npy"
sin_2d = np.load(img_path)

# FFT 2D
F = np.fft.fft2(sin_2d)
Fshift = np.fft.fftshift(F)
mag = np.abs(Fshift)

plt.imshow(np.log1p(mag), cmap='jet')
plt.title("Espectro de amplitudes (numpy)")
plt.axis("off")
plt.show()

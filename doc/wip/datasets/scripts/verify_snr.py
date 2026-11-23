# SNR "en potencia" (definición de §4.1) estimado desde k-space medido y sigma_k conocido; y sigma_k desde el fondo de la imagen.
import numpy as np
rng = np.random.default_rng(1)
NRO, NPE = 640, 368
x = np.linspace(-1, 1, NRO, endpoint=False)[:, None]; y = np.linspace(-1, 1, NPE, endpoint=False)[None, :]
obj = ((x / 0.45) ** 2 + (y / 0.8) ** 2 < 1) * (1 + 0.3 * np.cos(6 * y))
K = np.fft.fft2(obj); N = K.size
for target_db in (24, 32, 40):
    # sigma_k tal que Σ|K|² / (N·2σ²) = 10^(dB/10)
    sk = np.sqrt((np.abs(K) ** 2).sum() / (N * 2 * 10 ** (target_db / 10)))
    Km = K + sk * (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape))
    # sigma_k desde el fondo: imagen ortho (unitaria) -> sigma_img = sigma_k; ROI en la banda externa del RO (aire)
    img = np.fft.ifft2(Km, norm="ortho")
    roi = np.concatenate([img[:NRO // 8].ravel(), img[-NRO // 8:].ravel()])
    sk_hat = np.sqrt((roi.real.var() + roi.imag.var()) / 2)
    snr_hat = 10 * np.log10(((np.abs(Km) ** 2).sum() - N * 2 * sk_hat ** 2) / (N * 2 * sk_hat ** 2))
    lsb = sk_hat * 0.9 / np.abs(Km).max() * 2 ** 15
    print(f"SNR real {target_db} dB -> estimado {snr_hat:.2f} dB; sigma_k est/real {sk_hat / sk:.3f}; sigma en LSB (g=0.9/pico) {lsb:.3f}")

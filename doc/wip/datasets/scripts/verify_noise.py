# Verificación NumPy de las reglas de propagación de ruido usadas en el informe.
import numpy as np
rng = np.random.default_rng(0)

def cnoise(shape, s):  # ruido complejo, sigma s por componente
    return s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

# Objeto: elipse suave en una grilla "nativa" de 640 (RO, con sobremuestreo x2) x 368 (PE)
NRO, NPE = 640, 368
x = np.linspace(-1, 1, NRO, endpoint=False)[:, None]
y = np.linspace(-1, 1, NPE, endpoint=False)[None, :]
obj = ((x / 0.45) ** 2 + (y / 0.8) ** 2 < 1).astype(float) * (1 + 0.3 * np.cos(6 * y))  # cabe en la mitad central del RO
K = np.fft.fft2(obj)
sk = 0.02 * np.abs(K).max() / 1000
N = K.size
nk = cnoise(K.shape, sk)

# 1) sigma de imagen con ifft2 (norm backward) = sigma_k / sqrt(N)
img_n = np.fft.ifft2(nk)
print("1) sigma_img/sigma_k*sqrt(N) =", img_n.real.std() / sk * np.sqrt(N))

# 2) recorte del sobremuestreo en RO (mitad central en imagen) y vuelta a k-space
def crop_ro(Kin):
    im = np.fft.ifft(Kin, axis=0)          # el objeto ya está centrado en el arreglo (índice NRO/2)
    c = im[NRO // 4: NRO // 4 + NRO // 2]   # mitad central del FOV de lectura
    return np.fft.fft(c, axis=0)
Kc_s = crop_ro(K); Kc_n = crop_ro(nk)
print("2) sigma_k tras recorte / sigma_k =", Kc_n.real.std() / sk, "(1/sqrt2 =", 1 / np.sqrt(2), ")",
      "; pico tras/antes =", np.abs(Kc_s).max() / np.abs(K).max(),
      "; sigma en LSB tras/antes =", (Kc_n.real.std() / np.abs(Kc_s).max()) / (sk / np.abs(K).max()))

# 3) zero-padding centrado a 1024 x 1024: pico de la IDFT sin normalizar (como el hardware, que no divide por N)
def zpad(Kin, M1, M2):
    Ks = np.fft.fftshift(Kin)
    out = np.zeros((M1, M2), complex)
    a, b = Ks.shape
    out[(M1 - a) // 2:(M1 - a) // 2 + a, (M2 - b) // 2:(M2 - b) // 2 + b] = Ks
    return np.fft.ifftshift(out)
Kp_s = zpad(K, 1024, 1024); Kp_n = zpad(nk, 1024, 1024)
a_nat = np.fft.ifft2(K) * N; a_pad = np.fft.ifft2(Kp_s) * Kp_s.size          # IDFT sin 1/N
n_nat = np.fft.ifft2(nk) * N; n_pad = np.fft.ifft2(Kp_n) * Kp_n.size
print("3) pico IDFT sin normalizar: pad/nativo =", np.abs(a_pad).max() / np.abs(a_nat).max())
print("   sigma ruido IDFT sin normalizar: pad/nativo =", n_pad.real.std() / n_nat.real.std())
print("   SNR de píxel (pico/sigma): pad/nativo =", (np.abs(a_pad).max() / n_pad.real.std()) / (np.abs(a_nat).max() / n_nat.real.std()))
print("   sigma_k en LSB (g = 0.9/pico, Q1.15): nativo", sk * 0.9 / np.abs(K + nk).max() * 2**15,
      " pad", Kp_n[Kp_n != 0].real.std() * 0.9 / np.abs(Kp_s + Kp_n).max() * 2**15)

# 4) Af = 2 retrospectivo (1 de cada 2 líneas de PE): pliegue a mitad de FOV, ruido por muestra igual
Ka = Kp_s[:, ::2]
fold = np.fft.ifft2(Ka)
full = np.fft.ifft2(Kp_s)
print("4) pliegue: max|fold - (full[:, :512] + full[:, 512:])| / max|full| =",
      np.abs(fold - (full[:, :512] + full[:, 512:])).max() / np.abs(full).max())

# 5) compresión de bobinas: blanquear con la covarianza de ruido y después SVD -> ruido blanco sigma 1 en las virtuales
L, V = 15, 8
A = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
Psi = A @ A.conj().T / L + np.eye(L)       # covarianza de ruido no diagonal
C = np.linalg.cholesky(Psi)
noise = C @ (rng.standard_normal((L, 200000)) + 1j * rng.standard_normal((L, 200000))) / np.sqrt(2)
Psi_hat = noise @ noise.conj().T / noise.shape[1]
W = np.linalg.inv(np.linalg.cholesky(Psi_hat))   # blanqueo
sig = rng.standard_normal((L, 4000)) * np.linspace(3, 0.1, L)[:, None] + 0j
U, s, _ = np.linalg.svd(W @ sig, full_matrices=False)
P = U[:, :V].conj().T                         # 8 x 15
nw = P @ (W @ noise)
cov_v = nw @ nw.conj().T / nw.shape[1]
print("5) covarianza del ruido en las 8 virtuales ~ I: max|cov - I| =", np.abs(cov_v - np.eye(V)).max(),
      "; energía de señal retenida =", (s[:V] ** 2).sum() / (s ** 2).sum())

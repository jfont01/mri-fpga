import numpy as np
import matplotlib.pyplot as plt

save_path = "fp\img\sin_2d.jpeg"
N = 256
x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)

freq = 5.6
phase = 0
theta = np.pi

# x[None, :] -> forma (1, N)
# y[:, None] -> forma (N, 1)
# broadcasting -> resultado (N, N)
sin_2d = np.sin(
    2 * np.pi * freq * (np.cos(theta) * x[None, :] +
                        np.sin(theta) * y[:, None]) + phase
)

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(sin_2d, cmap='jet', origin='lower', vmin=-1, vmax=1)
ax.set_title('Space Domain')
ax.axis('off')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([-1, 0, 1])
plt.imsave(save_path, sin_2d, cmap='jet', vmin=-1, vmax=1)

plt.show()
np.save("sin_2d.npy", sin_2d)

print("shape:", sin_2d.shape)   # (256, 256)
print("num elementos:", sin_2d.size)
print("bytes en memoria:", sin_2d.nbytes)
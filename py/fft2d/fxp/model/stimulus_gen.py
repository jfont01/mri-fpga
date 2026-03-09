import math, numpy as np
from cfxp import CFxp
from cfxp2d import CFxp2D
from typing import List, Tuple
from numpy.typing import NDArray
from numpy import float64
# ------------------------------ 1D Stimulus ------------------------------
def gen_impulse_1d(
        N: int, 
        NB: int, 
        NBF: int
) -> List[CFxp]:
    one = CFxp.quantize(1.0, 0.0, NB, NBF)
    x = [CFxp.quantize(0.0, 0.0, NB, NBF) for _ in range(N)]
    x[0] = one
    return x

def gen_tone_complex_1d(
        N: int, 
        k0: int, 
        NB: int, 
        NBF: int, 
        A: float
) -> List[CFxp]:
    x = []
    for n in range(N):
        ang = 2.0 * math.pi * k0 * n / N
        re = A * math.cos(ang)
        im = A * math.sin(ang)
        x.append(CFxp.quantize(re, im, NB, NBF))
    return x

def gen_tone_cos_real_1d(
        N: int, 
        k0: int, 
        NB: int, 
        NBF: int, 
        A: float
) -> List[CFxp]:
    
    x = []
    for n in range(N):
        ang = 2.0 * math.pi * k0 * n / N
        re = A * math.cos(ang)
        x.append(CFxp.quantize(re, 0.0, NB, NBF))
    return x

def gen_sine_real_1d(
        N: int, 
        k0: int, 
        NB: int, 
        NBF: int, 
        A: float
) -> Tuple[List[CFxp], List[float]]:
    
    x_q = []
    x_f = []
    for n in range(N):
        ang = 2.0 * math.pi * k0 * n / N
        val = A * math.sin(ang)
        x_f.append(val)
        x_q.append(CFxp.quantize(val, 0.0, NB, NBF))
    return x_q, x_f

def gen_two_sines_real_hz_1d(
        N: int, 
        Fs: float, 
        f0: float, 
        f1: float, 
        A0: float, 
        A1: float, 
        NB: int, 
        NBF: int
) -> Tuple[NDArray[float64], List[CFxp], NDArray[float64]]:

    time = np.arange(0.0, N/Fs, 1/Fs)
    x_f = A0 * np.sin( 2*np.pi*f0*time ) + A1 * np.sin( 2*np.pi*f1*time )

    # cuantización a CFxp
    x_q = []
    for xn in x_f:
        z = CFxp.quantize(float(xn), 0.0, NB, NBF)
        x_q.append(z)


    return x_f, x_q, time

# ------------------------------ 2D Stimulus ------------------------------
def gen_tone_cos_2d(
    Nx: int,
    Ny: int,
    kx0: int,
    ky0: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    phi: float = 0.0,
) -> Tuple[CFxp2D, NDArray[np.float64]]:
    """
    Genera una senoidal 2D real:
        s[x, y] = A * sin( 2*pi*(kx0*x/Nx + ky0*y/Ny) + phi )

    Devuelve:
        img_fx : CFxp2D  (contenedor Ny x Nx de CFxp)
        img_f  : np.ndarray Ny x Nx con la versión en float
    """
    img_q = []
    img_f = []

    for y in range(Ny):
        row_q = []
        row_f = []
        for x in range(Nx):
            ang = 2.0 * math.pi * (kx0 * x / Nx + ky0 * y / Ny) + phi
            val = A * math.sin(ang)   # o cos, si querés tono coseno

            # cuantización en CFxp, parte imaginaria = 0.0
            z_q = CFxp.quantize(val, 0.0, NB, NBF)
            row_q.append(z_q)
            row_f.append(val)
        img_q.append(row_q)
        img_f.append(row_f)

    # Imagen fija como clase contenedora
    img_fx = CFxp2D(img_q)

    # Imagen referencia en float como numpy array
    img_f_np = np.array(img_f, dtype=np.float64)

    return img_fx, img_f_np

def gen_impulse_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    x0: int | None = None,
    y0: int | None = None,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Impulso 2D (delta):

        img_f[y,x] = A en (x0,y0), 0 en el resto.

    Por defecto, el impulso está en el centro de la imagen.
    """
    if x0 is None:
        x0 = Nx // 2
    if y0 is None:
        y0 = Ny // 2

    img_f = np.zeros((Ny, Nx), dtype=float64)
    img_f[y0, x0] = A

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_checkerboard_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    period_x: int = 8,
    period_y: int = 8,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Patrón tipo damero (checkerboard) 2D.

    El valor alterna ±A por bloques de tamaño period_x x period_y.
    """
    img_f = np.zeros((Ny, Nx), dtype=float64)

    for y in range(Ny):
        for x in range(Nx):
            bx = (x // period_x)
            by = (y // period_y)
            # alterna signo según la paridad de bx+by
            sign = 1.0 if ((bx + by) % 2 == 0) else -1.0
            img_f[y, x] = A * sign

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_concentric_rings_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    rings_period: float = 16.0,
    phase0: float = 0.0,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Patrón de anillos concéntricos tipo 'bullseye':

        s[r] = A * cos( 2*pi * r / rings_period + phase0 )

    donde r es la distancia radial al centro en píxeles.
    """
    cx = Nx / 2.0
    cy = Ny / 2.0

    Y, X = np.mgrid[0:Ny, 0:Nx]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    pattern = A * np.cos(2.0 * math.pi * r / rings_period + phase0)

    # referencia float
    img_f = pattern.astype(float64)

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_gaussian_spots_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    variance: float = 0.01,
    centers: list[tuple[int, int]] | None = None,
    A: float = 1.0,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Suma de 'blobs' Gaussianos 2D sobre fondo 0.

    variance: varianza σ² en unidades de (píxel^2).
    centers: lista de (x, y). Si es None, dos spots horizontales.
    A: amplitud máxima de cada gaussiana (antes de cuantizar).
    """
    if centers is None:
        cx = Nx // 2
        cy = Ny // 2
        dx = Nx // 4
        centers = [(cx - dx, cy), (cx + dx, cy)]

    sigma2 = float(variance)
    sigma2 = sigma2 if sigma2 > 0.0 else 1e-6

    Y, X = np.mgrid[0:Ny, 0:Nx]
    img_f = np.zeros((Ny, Nx), dtype=float64)

    for (cx, cy) in centers:
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        img_f += A * np.exp(-r2 / (2.0 * sigma2))

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_dots_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    coords: list[tuple[int, int]] | None = None,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Imagen con uno o varios 'dots' (píxeles puntuales) de valor A sobre fondo 0.

    coords: lista de (x, y). Si es None, se crean dos puntos simétricos
            horizontales alrededor del centro.
    """
    if coords is None:
        cx = Nx // 2
        cy = Ny // 2
        # Dos puntos a izquierda y derecha del centro
        coords = [(cx - Nx // 6, cy), (cx + Nx // 6, cy)]

    img_f = np.zeros((Ny, Nx), dtype=float64)
    for (x0, y0) in coords:
        if 0 <= x0 < Nx and 0 <= y0 < Ny:
            img_f[y0, x0] = A

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_circles_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    radius: int = 5,
    centers: list[tuple[int, int]] | None = None,
    filled: bool = True,
    thickness: int = 1,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Genera uno o varios círculos (discos) de valor A sobre fondo 0.

    radius: radio en píxeles.
    centers: lista de (x, y). Si es None, dos círculos simétricos horizontales.
    filled:  True  -> discos rellenos
             False -> solo circunferencia (con 'thickness' píxeles de ancho)
    """
    if centers is None:
        cx = Nx // 2
        cy = Ny // 2
        dx = Nx // 4
        centers = [(cx - dx, cy), (cx + dx, cy)]

    Y, X = np.mgrid[0:Ny, 0:Nx]
    img_f = np.zeros((Ny, Nx), dtype=float64)

    for (cx, cy) in centers:
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        if filled:
            mask = r <= radius
        else:
            mask = np.logical_and(r >= radius - thickness / 2.0,
                                  r <= radius + thickness / 2.0)
        img_f[mask] = A

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

def gen_squares_2d(
    Nx: int,
    Ny: int,
    NB: int,
    NBF: int,
    A: float = 1.0,
    size: int = 8,
    centers: list[tuple[int, int]] | None = None,
) -> Tuple[CFxp2D, NDArray[float64]]:
    """
    Genera uno o varios cuadrados blancos (valor A) sobre fondo 0.

    size: lado del cuadrado en píxeles.
    centers: lista de (x, y) centros de cada cuadrado.
             Si es None, dos cuadrados horizontales.
    """
    if centers is None:
        cx = Nx // 2
        cy = Ny // 2
        dx = Nx // 4
        centers = [(cx - dx, cy), (cx + dx, cy)]

    img_f = np.zeros((Ny, Nx), dtype=float64)

    for (cx, cy) in centers:
        x_start = max(0, cx - size // 2)
        x_end   = min(Nx, cx + size // 2)
        y_start = max(0, cy - size // 2)
        y_end   = min(Ny, cy + size // 2)
        img_f[y_start:y_end, x_start:x_end] = A

    # cuantizar
    img_q: list[list[CFxp]] = []
    for y in range(Ny):
        row_q: list[CFxp] = []
        for x in range(Nx):
            val = float(img_f[y, x])
            row_q.append(CFxp.quantize(val, 0.0, NB, NBF))
        img_q.append(row_q)

    img_fx = CFxp2D(img_q)
    return img_fx, img_f

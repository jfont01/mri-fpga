import numpy as np
from numpy.typing import NDArray


def show_npy_min_max(path: str) -> None:
    """
    Carga un archivo .npy y printea información básica:
    - shape
    - dtype
    - valor mínimo
    - valor máximo
    """
    arr: NDArray = np.load(path)

    vmin = np.min(arr.real)
    vmax = np.max(arr.real)

    print(f"Archivo: {path}")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  min   : {vmin}")
    print(f"  max   : {vmax}")


if __name__ == "__main__":
    show_npy_min_max("coil_imagesN32.npy")
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from cfxp import CFxp


@dataclass
class CFxp2D:
    data: List[List[CFxp]]  # Ny x Nx

    def __post_init__(self):
        if not self.data or not self.data[0]:
            raise ValueError("CFxp2D: data vacío")

        self.Ny = len(self.data)
        self.Nx = len(self.data[0])
        if self.Ny != self.Nx:
            raise ValueError("CFxp2D: la imágen debe tener las mismas dimensiones")

        # Comprobar que todas las filas tengan misma longitud
        for row in self.data:
            if len(row) != self.Nx:
                raise ValueError("CFxp2D: todas las filas deben tener la misma longitud")

        # Tomar NB/NBF del primer píxel
        self.NB  = self.data[0][0].re.NB
        self.NBF = self.data[0][0].re.NBF

    def __getitem__(self, idx):
        """Permite hacer img[y][x] o img[y] como si fuera una lista 2D."""
        return self.data[idx]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.Ny, self.Nx)
    
    def to_float(self) -> np.ndarray:
        """
        Devuelve un numpy array Ny x Nx con la parte real en float.
        """
        return np.array(
            [[z.to_complex().real for z in row] for row in self.data],
            dtype=np.float64
        )

    def quant_metrics_real(self, ref_float: np.ndarray) -> dict:
        """
        Calcula métricas de cuantización respecto a una imagen float Nx x Ny.

        ref_float: numpy array Ny x Nx (referencia en punto flotante).

        Devuelve un dict con:
            MSE, RMSE, MAE, MAX_ABS_ERR, SNR_dB, PSNR_dB
        """
        arr_f = np.array(ref_float, dtype=np.float64)
        arr_q = self.to_float()

        if arr_f.shape != arr_q.shape:
            raise ValueError(f"Dim mismatch en quant_metrics: ref={arr_f.shape}, fxp={arr_q.shape}")

        err = arr_q - arr_f

        mse = np.mean(err ** 2)
        rmse = math.sqrt(mse) if mse > 0.0 else 0.0
        mae = np.mean(np.abs(err))
        max_abs_err = np.max(np.abs(err))

        signal_power = np.mean(arr_f ** 2)
        noise_power  = np.mean(err ** 2)

        if noise_power > 0.0 and signal_power > 0.0:
            snr_db = 10.0 * math.log10(signal_power / noise_power)
        else:
            snr_db = float("inf")

        peak = float(np.max(np.abs(arr_f)))
        if mse > 0.0 and peak > 0.0:
            psnr_db = 10.0 * math.log10((peak ** 2) / mse)
        else:
            psnr_db = float("inf")

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAX_ABS_ERR": max_abs_err,
            "SNR_dB": snr_db,
            "PSNR_dB": psnr_db,
        }
    
    def __repr__(self) -> str:
        return f"CFxp2D(shape=({self.Ny},{self.Nx}), S({self.NB},{self.NBF}))"

    @classmethod
    def from_float(cls, arr: np.ndarray, NB: int, NBF: int) -> "CFxp2D":
        """
        Construye un CFxp2D a partir de una imagen en float (np.ndarray).

        arr : np.ndarray de shape (Ny, Nx)
        NB  : número total de bits
        NBF : número de bits fraccionales
        """
        # Aseguramos que sea array 2D de float
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"from_float espera un array 2D, recibió ndim={arr.ndim}")

        Ny, Nx = arr.shape

        # Creamos la lista 2D de CFxp (Ny x Nx)
        data: list[list[CFxp]] = []
        for y in range(Ny):
            row: list[CFxp] = []
            for x in range(Nx):
                value = float(arr[y, x])        # valor en punto flotante
                z_q   = CFxp.quantize(value,    # parte real
                                      0.0,      # parte imaginaria
                                      NB,
                                      NBF)
                row.append(z_q)
            data.append(row)

        # Devolvemos la instancia de CFxp2D
        return cls(data)


    @classmethod
    def from_complex(cls, arr: np.ndarray, NB: int, NBF: int,
                     mode: str = "round", signed: bool = True) -> "CFxp2D":
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"from_complex espera 2D, got shape={arr.shape}")

        Ny, Nx = arr.shape
        data = []
        for y in range(Ny):
            row = []
            for x in range(Nx):
                z = complex(arr[y, x])
                fx = CFxp.from_complex(z, NB, NBF, mode=mode, signed=signed)
                row.append(fx)
            data.append(row)


        return cls(data)

            
    def conj(self, inplace: bool = True) -> "CFxp2D":
        Ny, Nx = self.shape
        if inplace:
            for y in range(Ny):
                for x in range(Nx):
                    self.data[y][x] = self.data[y][x].conj()
            return self
        else:
            out = self.copy()
            Ny, Nx = out.shape
            for y in range(Ny):
                for x in range(Nx):
                    out.data[y][x] = out.data[y][x].conj()
            return out
    
    def max_abs_components(self) -> Tuple[float, float]:
        """
        Devuelve (max_abs_re, max_abs_im) sobre toda la imagen,
        tomando el máximo de |Re| y de |Im| en todos los píxeles.
        """
        max_re = 0.0
        max_im = 0.0

        for row in self.data:
            for z in row:
                c = z.to_complex()
                abs_re = abs(c.real)
                abs_im = abs(c.imag)
                if abs_re > max_re:
                    max_re = abs_re
                if abs_im > max_im:
                    max_im = abs_im

        return max_re, max_im

    def max_abs_value(self) -> float:
        """
        Devuelve max( |Re|, |Im| ) global en toda la imagen.
        Útil para estimar cuántos bits enteros se necesitan.
        """
        max_re, max_im = self.max_abs_components()
        return max(max_re, max_im)
    

    def to_complex_array(self) -> np.ndarray:
        """
        Devuelve un array complejo Ny x Nx con cada píxel como complex128.
        """
        return np.array(
            [[z.to_complex() for z in row] for row in self.data],
            dtype=np.complex128,
        )
    

    def quant_metrics_complex(self, ref_cplx: np.ndarray) -> dict:
        """
        Métricas de error para datos complejos 

        ref_cplx: np.ndarray complejo Ny x Nx

        Devuelve un dict con:
            - MSE, RMSE, MAE, MAX_ABS_ERR, SNR_dB, PSNR_dB
            - MSE_RE, MSE_IM 
        """
        arr_f = np.asarray(ref_cplx, dtype=np.complex128)
        arr_q = self.to_complex_array()

        if arr_f.shape != arr_q.shape:
            raise ValueError(
                f"Dim mismatch en quant_metrics_complex: "
                f"ref={arr_f.shape}, fxp={arr_q.shape}"
            )

        # Error complejo
        err = arr_q - arr_f

        # Métricas sobre el valor complejo (norma)
        abs_err = np.abs(err)
        mse = float(np.mean(abs_err**2))
        rmse = math.sqrt(mse) if mse > 0.0 else 0.0
        mae = float(np.mean(abs_err))
        max_abs_err = float(np.max(abs_err))

        # MSE separado real/imag
        mse_re = float(np.mean((arr_q.real - arr_f.real) ** 2))
        mse_im = float(np.mean((arr_q.imag - arr_f.imag) ** 2))

        # Potencias para SNR
        signal_power = float(np.mean(np.abs(arr_f) ** 2))
        noise_power  = float(np.mean(abs_err ** 2))

        if noise_power > 0.0 and signal_power > 0.0:
            snr_db = 10.0 * math.log10(signal_power / noise_power)
        else:
            snr_db = float("inf")

        # PSNR usando máximo de la referencia
        peak = float(np.max(np.abs(arr_f)))
        if mse > 0.0 and peak > 0.0:
            psnr_db = 10.0 * math.log10((peak ** 2) / mse)
        else:
            psnr_db = float("inf")

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAX_ABS_ERR": max_abs_err,
            "MSE_RE": mse_re,
            "MSE_IM": mse_im,
            "SNR_dB": snr_db,
            "PSNR_dB": psnr_db,
        }



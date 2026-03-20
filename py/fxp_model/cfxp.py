from fxp import Fxp
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class CFxp:
    re: Fxp
    im: Fxp

    def __add__(self, sumando: "CFxp") -> "CFxp":
        if not isinstance(sumando, CFxp):
            return NotImplemented
        return CFxp(self.re + sumando.re, self.im + sumando.im)

    def __sub__(self, restando: "CFxp") -> "CFxp":
        if not isinstance(restando, CFxp):
            return NotImplemented
        return CFxp(self.re - restando.re, self.im - restando.im)
    
    def __mul__(self, other: "CFxp") -> "CFxp":
        """Multiplicación compleja: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)"""
        if not isinstance(other, CFxp):
            return NotImplemented

        ac = self.re * other.re
        bd = self.im * other.im
        ad = self.re * other.im
        bc = self.im * other.re

        return CFxp(ac - bd, ad + bc)


    def conj(self) -> "CFxp":
        return CFxp(self.re, -self.im)
    

    def to_complex(self) -> complex:
        return complex(self.re.get_val(), self.im.get_val())
    
    @classmethod
    def from_uint_pair(
        cls,
        re_raw: int,
        im_raw: int,
        NB: int,
        NBF: int,
        signed: bool = True,
    ) -> "CFxp":
        re_fxp = Fxp.from_uint_raw(re_raw, NB, NBF, signed=signed)
        im_fxp = Fxp.from_uint_raw(im_raw, NB, NBF, signed=signed)
        return cls(re=re_fxp, im=im_fxp)

    @classmethod
    def from_complex(cls, z: complex, NB: int, NBF: int,
            mode: str = "round", signed: bool = True) -> "CFxp":
        """
        Construye un CFxp a partir de un número complejo de Python,
        cuantizando real e imag en formato S(NB, NBF).
        """
        re_fxp = Fxp.quantize(z.real, NB=NB, NBF=NBF, mode=mode, signed=signed)
        im_fxp = Fxp.quantize(z.imag, NB=NB, NBF=NBF, mode=mode, signed=signed)
        return cls(re=re_fxp, im=im_fxp)

    @classmethod
    def quantize(cls, re_f: float, im_f: float, NB: int, NBF: int, mode="round", signed=True) -> "CFxp":
        return cls(
            Fxp.quantize(re_f, NB, NBF, mode=mode, signed=signed),
            Fxp.quantize(im_f, NB, NBF, mode=mode, signed=signed),
        )
    

    def cast(self, NB_out: int, NBF_out: int, mode: str = "round", overflow: str = "saturate") -> "CFxp":
        return CFxp(
            self.re.cast(NB_out, NBF_out, mode=mode, overflow=overflow),
            self.im.cast(NB_out, NBF_out, mode=mode, overflow=overflow),
        )
    

    def shift_right(self, k: int = 1) -> "CFxp":
        """
        Shift aritmético a la derecha de la parte real e imaginaria.
        Usa el shift_right de Fxp en ambos componentes.
        """
        if k == 0:
            return self

        return CFxp(
            self.re.shift_right(k),
            self.im.shift_right(k),
        )

    def __rshift__(self, k: int) -> "CFxp":
        """
        Permite hacer y = x >> k directamente sobre CFxp.
        """
        if not isinstance(k, int):
            raise TypeError("shift amount debe ser int")
        return self.shift_right(k)
    
    @staticmethod
    def quant_metrics_vec(
        vec_fx: list["CFxp"],
        ref: np.ndarray | list[complex] | list[float],
    ) -> dict:
        """
        Calcula métricas de cuantización para un vector 1D de CFxp
        respecto a un vector de referencia (float o complex).

        vec_fx : lista de CFxp (señal cuantizada)
        ref    : iterable de float o complex (referencia)

        Devuelve un dict con:
            MSE, RMSE, MAE, MAX_ABS_ERR, SNR_dB, PSNR_dB
        """
        # referencia como array 1D de complex (o float)
        arr_f = np.asarray(ref)
        if arr_f.ndim != 1:
            raise ValueError(f"quant_metrics_vec espera ref 1D, recibió ndim={arr_f.ndim}")

        # vector cuantizado a complex
        arr_q = np.array([z.to_complex() for z in vec_fx], dtype=np.complex128)

        if arr_f.shape != arr_q.shape:
            raise ValueError(
                f"Dim mismatch en quant_metrics_vec: ref={arr_f.shape}, fxp={arr_q.shape}"
            )

        # si la referencia es real, la convertimos a complex explícitamente
        if not np.iscomplexobj(arr_f):
            arr_f = arr_f.astype(np.complex128)

        # error complejo
        err = arr_q - arr_f
        abs_err = np.abs(err)

        mse = np.mean(abs_err**2)
        rmse = math.sqrt(mse) if mse > 0.0 else 0.0
        mae = np.mean(abs_err)
        max_abs_err = np.max(abs_err)

        signal_power = np.mean(np.abs(arr_f) ** 2)
        noise_power  = mse  # = mean(|err|^2)

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

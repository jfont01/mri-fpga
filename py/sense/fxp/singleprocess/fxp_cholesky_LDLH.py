import numpy as np
from dataclasses import dataclass

from fxp import Fxp
from cfxp import CFxp


@dataclass
class FXP_LDLH_2x2:
    """
    Representación compacta de A = L D L^H para bloque 2x2.

    L = [[1,   0],
         [l10, 1]]

    D = diag(d0, d1)

    d0, d1 : Fxp   (reales positivos)
    l10    : CFxp  (complejo)
    """
    d0: Fxp
    d1: Fxp
    l10: CFxp


# ------------------------------------------------------------------
# HELPERS DE DIVISIÓN
# ------------------------------------------------------------------
def fxp_div_fxp(
    num: Fxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    signed: bool = True,
) -> Fxp:
    """
    Placeholder: división real fixed num / den.

    Reemplazar por tu implementación real de división o recíproco.
    """
    raise NotImplementedError("Falta implementar fxp_div_fxp según tu librería fixed-point")


def fxp_div_cfxp_by_real(
    num: CFxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    signed: bool = True,
) -> CFxp:
    """
    Placeholder: división compleja por real positivo.

    num / den = (num.re / den) + j (num.im / den)
    """
    re_q = fxp_div_fxp(num.re, den, NB_out, NBF_out, mode=mode, signed=signed)
    im_q = fxp_div_fxp(num.im, den, NB_out, NBF_out, mode=mode, signed=signed)
    return CFxp(re_q, im_q)


# ------------------------------------------------------------------
# FACTORIZACIÓN LDLH
# ------------------------------------------------------------------
def fxp_compute_LD(
    Aij: np.ndarray,   # shape (2,2), idealmente entradas CFxp
    NB_L: int,
    NBF_L: int,
    NB_D: int,
    NBF_D: int,
    signed: bool = True,
    eps: float = 1e-12,
) -> FXP_LDLH_2x2:
    """
    Factorización LDL^H en fixed-point para bloque 2x2.

    Input
    -----
    Aij : np.ndarray, shape (2,2)
        Se espera:
            Aij[0,0], Aij[1,1] : reales (pueden venir como CFxp con imag=0)
            Aij[1,0]           : complejo
    NB_L, NBF_L : formato de salida para l10
    NB_D, NBF_D : formato de salida para d0 y d1

    Output
    ------
    FXP_LDLH_2x2(d0, d1, l10)

    Significado matemático
    ----------------------
        d0  = a00
        l10 = a10 / d0
        d1  = a11 - |a10|^2 / d0
    """

    if Aij.shape != (2, 2):
        raise ValueError(f"Se esperaba Aij con shape (2,2), recibido {Aij.shape}")

    a00 = Aij[0, 0]
    a10 = Aij[1, 0]
    a11 = Aij[1, 1]

    # Asumimos que las diagonales son reales.
    # Si te llegan como CFxp, tomamos la parte real.
    if isinstance(a00, CFxp):
        d0 = a00.re.cast(NB_D, NBF_D, mode="round")
    elif isinstance(a00, Fxp):
        d0 = a00.cast(NB_D, NBF_D, mode="round")
    else:
        raise TypeError("Aij[0,0] debe ser Fxp o CFxp")

    if isinstance(a11, CFxp):
        a11_re = a11.re.cast(NB_D, NBF_D, mode="round")
    elif isinstance(a11, Fxp):
        a11_re = a11.cast(NB_D, NBF_D, mode="round")
    else:
        raise TypeError("Aij[1,1] debe ser Fxp o CFxp")

    if not isinstance(a10, CFxp):
        raise TypeError("Aij[1,0] debe ser CFxp")

    # Chequeo HPD básico sobre d0 usando valor float
    if float(d0.get_val()) <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD: d0=a00={float(d0.get_val())} no es estrictamente positivo"
        )

    # l10 = a10 / d0
    l10 = fxp_div_cfxp_by_real(
        a10,
        d0,
        NB_out=NB_L,
        NBF_out=NBF_L,
        mode="round",
        signed=signed,
    )

    # |a10|^2 = a10.re^2 + a10.im^2
    # y luego term = |a10|^2 / d0
    abs_a10_sq = (a10.re * a10.re + a10.im * a10.im).cast(NB_D, NBF_D, mode="round")

    term = fxp_div_fxp(
        abs_a10_sq,
        d0,
        NB_out=NB_D,
        NBF_out=NBF_D,
        mode="round",
        signed=signed,
    )

    d1 = (a11_re - term).cast(NB_D, NBF_D, mode="round")

    if float(d1.get_val()) <= eps:
        raise np.linalg.LinAlgError(
            f"A no es HPD o está muy mal condicionada: d1={float(d1.get_val())}"
        )

    return FXP_LDLH_2x2(d0=d0, d1=d1, l10=l10)


# ------------------------------------------------------------------
# FORWARD
# ------------------------------------------------------------------
def fxp_forward_subst_ldlh(
    LD: FXP_LDLH_2x2,
    b: np.ndarray,   # shape (2,), entradas CFxp
    NB_Y: int,
    NBF_Y: int,
) -> np.ndarray:
    """
    Resuelve L y = b con L = [[1,0],[l10,1]]

    y0 = b0
    y1 = b1 - l10*y0

    Output
    ------
    y : np.ndarray, shape (2,), dtype=object, entradas CFxp
    """
    if b.shape != (2,):
        raise ValueError(f"Se esperaba b con shape (2,), recibido {b.shape}")

    b0 = b[0]
    b1 = b[1]

    if not isinstance(b0, CFxp) or not isinstance(b1, CFxp):
        raise TypeError("b debe contener CFxp en ambas entradas")

    y0 = b0.cast(NB_Y, NBF_Y, mode="round")
    y1 = (b1 - (LD.l10 * y0)).cast(NB_Y, NBF_Y, mode="round")

    return np.array([y0, y1], dtype=object)


# ------------------------------------------------------------------
# DIAGONAL
# ------------------------------------------------------------------
def fxp_diagonal_subst(
    LD: FXP_LDLH_2x2,
    y: np.ndarray,   # shape (2,), CFxp
    NB_Z: int,
    NBF_Z: int,
    signed: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Resuelve D z = y con D = diag(d0, d1)

    z0 = y0 / d0
    z1 = y1 / d1
    """
    if y.shape != (2,):
        raise ValueError(f"Se esperaba y con shape (2,), recibido {y.shape}")

    y0 = y[0]
    y1 = y[1]

    if float(LD.d0.get_val()) <= eps or float(LD.d1.get_val()) <= eps:
        raise np.linalg.LinAlgError(
            f"D no es invertible o no es positiva: d0={float(LD.d0.get_val())}, d1={float(LD.d1.get_val())}"
        )

    z0 = fxp_div_cfxp_by_real(
        y0, LD.d0,
        NB_out=NB_Z, NBF_out=NBF_Z,
        mode="round", signed=signed
    )

    z1 = fxp_div_cfxp_by_real(
        y1, LD.d1,
        NB_out=NB_Z, NBF_out=NBF_Z,
        mode="round", signed=signed
    )

    return np.array([z0, z1], dtype=object)


# ------------------------------------------------------------------
# BACKWARD
# ------------------------------------------------------------------
def fxp_backward_subst_ldlh(
    LD: FXP_LDLH_2x2,
    z: np.ndarray,   # shape (2,), CFxp
    NB_M: int,
    NBF_M: int,
) -> np.ndarray:
    """
    Resuelve L^H m = z

    m1 = z1
    m0 = z0 - conj(l10)*m1
    """
    if z.shape != (2,):
        raise ValueError(f"Se esperaba z con shape (2,), recibido {z.shape}")

    z0 = z[0]
    z1 = z[1]

    m1 = z1.cast(NB_M, NBF_M, mode="round")
    m0 = (z0 - (LD.l10.conj() * m1)).cast(NB_M, NBF_M, mode="round")

    return np.array([m0, m1], dtype=object)


# ------------------------------------------------------------------
# SOLVER COMPLETO
# ------------------------------------------------------------------
def fxp_cholesky_ldlh(
    Aij: np.ndarray,   # shape (2,2), Fxp/CFxp
    bij: np.ndarray,   # shape (2,), CFxp
    NB_L: int,
    NBF_L: int,
    NB_D: int,
    NBF_D: int,
    NB_Y: int,
    NBF_Y: int,
    NB_Z: int,
    NBF_Z: int,
    NB_M: int,
    NBF_M: int,
    signed: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Resuelve Aij m = bij usando LDL^H manual 2x2 en fixed-point.

    Output
    ------
    m_hat : np.ndarray, shape (2,), dtype=object, entradas CFxp
    """
    LD = fxp_compute_LD(
        Aij,
        NB_L=NB_L, NBF_L=NBF_L,
        NB_D=NB_D, NBF_D=NBF_D,
        signed=signed,
        eps=eps,
    )

    y = fxp_forward_subst_ldlh(LD, bij, NB_Y=NB_Y, NBF_Y=NBF_Y)
    z = fxp_diagonal_subst(LD, y, NB_Z=NB_Z, NBF_Z=NBF_Z, signed=signed, eps=eps)
    m_hat = fxp_backward_subst_ldlh(LD, z, NB_M=NB_M, NBF_M=NBF_M)

    return m_hat
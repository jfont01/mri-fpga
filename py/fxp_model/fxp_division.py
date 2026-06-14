from __future__ import annotations

import os
from typing import Literal

from apytypes import APyFixed

from fxp import Fxp, FXP_STATS


FxpDivMethod = Literal[
    "apy",
    "restoring",
    "newton_raphson",
]


DEFAULT_FXP_DIV_METHOD = "restoring"


def get_default_div_method() -> str:
    return os.environ.get("FXP_DIV_METHOD", DEFAULT_FXP_DIV_METHOD)


def _check_fxp_operands(num: Fxp, den: Fxp) -> None:
    if not isinstance(num, Fxp):
        raise TypeError(f"num debe ser Fxp, recibido {type(num)}")
    if not isinstance(den, Fxp):
        raise TypeError(f"den debe ser Fxp, recibido {type(den)}")

    if float(den.get_val()) == 0.0:
        raise ZeroDivisionError("División por cero")


def _increment_div_stats() -> None:
    if "fxp_div" in FXP_STATS:
        FXP_STATS["fxp_div"] += 1


def _udiv_restoring(dividend: int, divisor: int, nbits: int) -> tuple[int, int]:
    if divisor == 0:
        raise ZeroDivisionError("División restoring unsigned por cero")

    remainder = 0
    quotient = 0

    for i in range(nbits - 1, -1, -1):
        remainder = (remainder << 1) | ((dividend >> i) & 1)

        trial = remainder - divisor
        if trial >= 0:
            remainder = trial
            quotient |= (1 << i)

    return quotient, remainder


def div_apy(
    num: Fxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    overflow: str = "saturate",
    signed_out: bool | None = None,
) -> Fxp:
    """
    División de referencia usando APyFixed.
    No representa una arquitectura RTL específica.
    """

    _check_fxp_operands(num, den)

    if signed_out is None:
        signed_out = bool(num.signed or den.signed)

    _increment_div_stats()

    res_val = num._val / den._val
    res_fxp = Fxp.from_apyfixed(res_val, signed=signed_out)

    return res_fxp.cast(
        NB_out=NB_out,
        NBF_out=NBF_out,
        mode=mode,
        overflow=overflow,
    )


def div_restoring(
    num: Fxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    overflow: str = "saturate",
    signed_out: bool | None = None,
) -> Fxp:
    """
    División fixed-point por algoritmo restoring.

    Esta función replica el comportamiento del RTL div_restoring:
    - signed por magnitudes absolutas
    - escalado por shift = NBF_out + den.NBF - num.NBF
    - corrección para cociente negativo con resto no nulo
    - saturación final mediante cast
    """

    _check_fxp_operands(num, den)

    if signed_out is None:
        signed_out = bool(num.signed or den.signed)

    _increment_div_stats()

    sign_q_neg = num.is_negative() ^ den.is_negative()

    num_mag = abs(num.to_sint())
    den_mag = abs(den.to_sint())

    shift = NBF_out + den.NBF - num.NBF

    if shift >= 0:
        dividend = num_mag << shift
        divisor = den_mag
        nbits = max(1, num.NB + shift)
    else:
        dividend = num_mag
        divisor = den_mag << (-shift)
        nbits = max(1, num.NB)

    q_raw_trunc, rem = _udiv_restoring(dividend, divisor, nbits)

    # Corrección para truncado negativo tipo floor.
    if sign_q_neg and rem != 0:
        q_raw_trunc += 1

    # Se agregan dos bits fraccionales temporales para que el cast final
    # aplique la política mode/overflow de forma centralizada.
    NBF_tmp = NBF_out + 2

    if sign_q_neg:
        q_tmp_sint = -(q_raw_trunc << 2)
    else:
        q_tmp_sint = q_raw_trunc << 2

    mag_bits = 1 if q_tmp_sint == 0 else abs(q_tmp_sint).bit_length()
    NB_tmp = max(NB_out + 2, mag_bits + 1)

    tmp = Fxp.from_sint(
        q_tmp_sint,
        NB=NB_tmp,
        NBF=NBF_tmp,
        signed=signed_out,
    )

    return tmp.cast(
        NB_out=NB_out,
        NBF_out=NBF_out,
        mode=mode,
        overflow=overflow,
    )


def div_newton_raphson(
    num: Fxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    overflow: str = "saturate",
    signed_out: bool | None = None,
    n_iter: int = 2,
) -> Fxp:
    """
    Placeholder para reciprocal + multiply con Newton-Raphson.

    Todavía no debe usarse para resultados finales hasta definir:
    - normalización de den
    - formato interno del recíproco
    - x0
    - política de cast entre multiplicaciones
    """

    raise NotImplementedError(
        "div_newton_raphson todavía no está implementado. "
        "Usar method='restoring' o method='apy'."
    )


def divide(
    num: Fxp,
    den: Fxp,
    NB_out: int,
    NBF_out: int,
    mode: str = "round",
    overflow: str = "saturate",
    signed_out: bool | None = None,
    method: str | None = None,
) -> Fxp:
    """
    Dispatcher único de división fixed-point.
    """

    if method is None:
        method = get_default_div_method()

    method = method.lower().strip()

    if method in ("apy", "apfixed", "apyfixed"):
        return div_apy(
            num=num,
            den=den,
            NB_out=NB_out,
            NBF_out=NBF_out,
            mode=mode,
            overflow=overflow,
            signed_out=signed_out,
        )

    if method in ("restoring", "div_restoring"):
        return div_restoring(
            num=num,
            den=den,
            NB_out=NB_out,
            NBF_out=NBF_out,
            mode=mode,
            overflow=overflow,
            signed_out=signed_out,
        )

    if method in ("newton_raphson", "nr", "reciprocal_nr"):
        return div_newton_raphson(
            num=num,
            den=den,
            NB_out=NB_out,
            NBF_out=NBF_out,
            mode=mode,
            overflow=overflow,
            signed_out=signed_out,
        )

    raise ValueError(
        f"Método de división inválido: {method}. "
        "Opciones: 'apy', 'restoring', 'newton_raphson'."
    )
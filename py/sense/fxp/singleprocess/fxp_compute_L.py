import numpy as np
import os
import sys
from typing import Tuple, Dict

# ------------------------- ENVIRONMENT SET -------------------------
FXP_MODEL_ROOT = os.environ.get("FXP_MODEL_ROOT")
if FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] FXP_MODEL_ROOT not defined")

sys.path.insert(0, FXP_MODEL_ROOT)

from fxp import Fxp
from cfxp import CFxp
from cfxptensor import CFxpTensor

SENSE_FXP_DIR = os.environ.get("SENSE_FXP_DIR")
if SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(SENSE_FXP_DIR, "helpers"))
from fxp_stats import update_acc_stats
# ------------------------------------------------------------------


def fxp_compute_L_i(
    Aij: CFxpTensor,
    Dij: CFxpTensor,
    stats_L: Dict | None = None,
    eps: float = 1e-12,
) -> CFxpTensor:
    """
    Construye L local de una factorización LDL^H 2x2.

    Input
    -----
    Aij : CFxpTensor, shape (2,2)
        Bloque local A.
    Dij : CFxpTensor, shape (2,2)
        Bloque local D, con d0 en [0,0].

    Output
    ------
    Lij : CFxpTensor, shape (2,2)
        Matriz triangular inferior unitaria:
            [[1, 0],
             [l10, 1]]
    """

    if Aij.shape != (2, 2):
        raise ValueError(f"Aij debe tener shape (2,2), recibió {Aij.shape}")

    if Dij.shape != (2, 2):
        raise ValueError(f"Dij debe tener shape (2,2), recibió {Dij.shape}")

    NB = Aij.NB
    NBF = Aij.NBF
    signed = Aij.signed

    if Dij.NB != NB or Dij.NBF != NBF or Dij.signed != signed:
        raise ValueError(
            "Aij y Dij deben tener el mismo formato fixed"
        )

    a10 = Aij[1, 0]
    d0_c = Dij[0, 0]
    d0 = d0_c.re.cast(NB, NBF, mode="round")

    d0_f = float(d0.get_val())
    if d0_f <= eps:
        raise np.linalg.LinAlgError(
            f"D no es válido para construir L: d0={d0_f} no es estrictamente positivo"
        )

    zero = Fxp.quantize(0.0, NB, NBF, signed=signed)
    one = Fxp.quantize(1.0, NB, NBF, signed=signed)

    l10 = CFxp.div_by_real(a10, d0, NB_out=NB, NBF_out=NBF, mode="round")

    if stats_L is not None:
        update_acc_stats(stats_L, "L10", l10)

    Lij = CFxpTensor.zeros(
        shape=(2, 2),
        NB=NB,
        NBF=NBF,
        signed=signed,
    )

    Lij[0, 0] = CFxp(one, zero)
    Lij[0, 1] = CFxp(zero, zero)
    Lij[1, 0] = l10.cast(NB, NBF, mode="round")
    Lij[1, 1] = CFxp(one, zero)

    return Lij
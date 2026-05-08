import os
import sys
import random
from pathlib import Path

PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from fxp import Fxp


def get_range_from(NB: int, NBF: int, signed: bool) -> tuple[float, float]:
    if signed:
        max_val = (2 ** (NB - NBF - 1)) - (2 ** (-NBF))
        min_val = -(2 ** (NB - NBF - 1))
    else:
        max_val = (2 ** (NB - NBF)) - (2 ** (-NBF))
        min_val = 0.0
    return min_val, max_val


def gen_random_value(rng: random.Random, NB: int, NBF: int, signed: bool) -> float:
    min_val, max_val = get_range_from(NB, NBF, signed)
    return rng.uniform(min_val, max_val)


def generate_fixed_case_dat(
    out_dir: str,
    n_cases: int,
    mode: str,
    NB_NUM: int,
    NBF_NUM: int,
    NB_DEN: int,
    NBF_DEN: int,
    NB_QUOTIENT: int,
    NBF_QUOTIENT: int,
    signed: bool = True,
    seed: int = 1234,
    use_restoring_as_ref: bool = True,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    in_file = out_path / f"div_restoring_{mode}_in.dat"
    out_file = out_path / f"div_restoring_{mode}_out.dat"

    rng = random.Random(seed)

    with in_file.open("w", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
        written = 0

        while written < n_cases:
            num_f = gen_random_value(rng, NB_NUM, NBF_NUM, signed)
            den_f = gen_random_value(rng, NB_DEN, NBF_DEN, signed)

            if abs(den_f) < 2.0 ** (-max(NBF_DEN - 2, 0)):
                den_f = 0.5 if den_f >= 0 else -0.5

            num = Fxp.quantize(num_f, NB=NB_NUM, NBF=NBF_NUM, mode=mode, signed=signed)
            den = Fxp.quantize(den_f, NB=NB_DEN, NBF=NBF_DEN, mode=mode, signed=signed)

            if den.get_val() == 0.0:
                continue

            min_neg_num = -(1 << (NB_NUM - 1))
            min_neg_den = -(1 << (NB_DEN - 1))

            if signed and num.to_sint() == min_neg_num:
                continue
            if signed and den.to_sint() == min_neg_den:
                continue

            if use_restoring_as_ref:
                q = Fxp.div_restoring(
                    num=num,
                    den=den,
                    NB_out=NB_QUOTIENT,
                    NBF_out=NBF_QUOTIENT,
                    mode=mode,
                    overflow="saturate",
                    signed_out=signed,
                )
            else:
                q = Fxp.div(
                    num=num,
                    den=den,
                    NB_out=NB_QUOTIENT,
                    NBF_out=NBF_QUOTIENT,
                    mode=mode,
                    overflow="saturate",
                    signed_out=signed,
                )

            fin.write(f"{num.to_hex().lower()} {den.to_hex().lower()}\n")
            fout.write(f"{q.to_hex().lower()}\n")

            written += 1

    print(f"[OK] Generated {written} cases")
    print(f"[OK] Input : {in_file}")
    print(f"[OK] Output: {out_file}")


if __name__ == "__main__":
    generate_fixed_case_dat(
        out_dir="./dat_div_restoring_16b",
        n_cases=1000,
        mode="trunc",
        NB_NUM=16,
        NBF_NUM=15,
        NB_DEN=16,
        NBF_DEN=15,
        NB_QUOTIENT=16,
        NBF_QUOTIENT=15,
        signed=True,
        seed=5678,
        use_restoring_as_ref=True,
    )
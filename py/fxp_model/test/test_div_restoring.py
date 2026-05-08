import unittest
import sys
import os
import random, math

PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from fxp import Fxp


N_INDIVIDUAL_TEST = 100000

class TestFxpDivRestoring(unittest.TestCase):


    def assert_fxp(self, dut: Fxp, ref: Fxp, ref_float: float):
        self.assertEqual(dut.NB, ref.NB)
        self.assertEqual(dut.NBF, ref.NBF)
        self.assertEqual(dut.signed, ref.signed)

        if dut.signed:
            dut_raw = dut.to_sint()
            ref_raw = ref.to_sint()
        else:
            dut_raw = dut.to_uint()
            ref_raw = ref.to_uint()

        diff_raw = abs(dut_raw - ref_raw)

        tol_raw = max(
            1,
            math.ceil(abs(ref_float) * (2.0 ** (dut.NBF - 52)))
        )

        self.assertLessEqual(
            diff_raw,
            tol_raw,
            msg=(
                "Mismatch beyond adaptive tolerance\n"
                f"dut={dut}\n"
                f"ref={ref}\n"
                f"ref_float={ref_float}\n"
                f"dut_raw={dut_raw}\n"
                f"ref_raw={ref_raw}\n"
                f"diff_raw={diff_raw}\n"
                f"tol_raw={tol_raw}"
            ),
        )

    def run_case(
        self,
        num_f: float,
        den_f: float,
        NB_in: int = 16,
        NBF_in: int = 12,
        NB_out: int = 16,
        NBF_out: int = 12,
        mode: str = "round",
        overflow: str = "saturate",
        signed: bool = True,
    ):
        if den_f == 0.0:
            self.skipTest("den_f = 0.0 no permitido")

        num = Fxp.quantize(num_f, NB=NB_in, NBF=NBF_in, mode=mode, signed=signed)
        den = Fxp.quantize(den_f, NB=NB_in, NBF=NBF_in, mode=mode, signed=signed)

        if den.get_val() == 0.0:
            self.skipTest(f"den cuantizado a cero: den_f={den_f}")

        min_neg_raw = -(1 << (NB_in - 1))
        if signed and num.to_sint() == min_neg_raw:
            return
        if signed and den.to_sint() == min_neg_raw:
            return

        dut = Fxp.div_restoring(
            num=num,
            den=den,
            NB_out=NB_out,
            NBF_out=NBF_out,
            mode=mode,
            overflow=overflow,
            signed_out=signed,
        )

        ref_float = num.get_val() / den.get_val()
        ref = Fxp.quantize(
            ref_float,
            NB=NB_out,
            NBF=NBF_out,
            mode=mode,
            signed=signed,
        )

        self.assert_fxp(dut, ref, ref_float)
    


    def gen_random_format(self, rng: random.Random) -> tuple[int, int]:
        NB = rng.randint(4, 64)
        NBF = rng.randint(2, NB - 2)
        return NB, NBF

    def get_range_from(
        self,
        NB: int,
        NBF: int,
        signed: bool,
    ) -> tuple[float, float]:
        if signed:
            max_val = (2 ** (NB - NBF - 1)) - (2 ** (-NBF))
            min_val = -(2 ** (NB - NBF - 1))
        else:
            max_val = (2 ** (NB - NBF)) - (2 ** (-NBF))
            min_val = 0.0
        return min_val, max_val
    
    def gen_random_value(
        self,
        rng: random.Random,
        NB: int,
        NBF: int,
        signed: bool,
    ) -> float:
        min_val, max_val = self.get_range_from(NB, NBF, signed)
        return rng.uniform(min_val, max_val)


    def test_rng(self):
        rng = random.Random(5678)

        for i in range(N_INDIVIDUAL_TEST):
            NB_in, NBF_in = self.gen_random_format(rng)
            NB_out, NBF_out = self.gen_random_format(rng)

            num_f = self.gen_random_value(rng, NB_in, NBF_in, True)
            den_f = self.gen_random_value(rng, NB_in, NBF_in, True)

            if abs(den_f) < 2.0 ** (-max(NBF_in - 2, 0)):
                den_f = 0.5 if den_f >= 0 else -0.5
            
            modes = ["round", "trunc"]
            for mode in modes:
                with self.subTest(
                    i=i,
                    num_f=num_f,
                    den_f=den_f,
                    NB_in=NB_in,
                    NBF_in=NBF_in,
                    NB_out=NB_out,
                    NBF_out=NBF_out,
                ):
                    self.run_case(
                        num_f=num_f,
                        den_f=den_f,
                        NB_in=NB_in,
                        NBF_in=NBF_in,
                        NB_out=NB_out,
                        NBF_out=NBF_out,
                        mode=mode,
                        overflow="saturate",
                        signed=True,
                    )



if __name__ == "__main__":
    unittest.main(verbosity=2)
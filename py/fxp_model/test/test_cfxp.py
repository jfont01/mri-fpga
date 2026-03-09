# test_cfxp.py
import unittest
import math

from cfxp import CFxp


def assert_complex_close(tc: unittest.TestCase, z1: complex, z2: complex, tol: float, msg=""):
    tc.assertTrue(math.isclose(z1.real, z2.real, abs_tol=tol), msg or f"Re: {z1.real} != {z2.real}")
    tc.assertTrue(math.isclose(z1.imag, z2.imag, abs_tol=tol), msg or f"Im: {z1.imag} != {z2.imag}")


class TestCFxp(unittest.TestCase):
    def test_add_sub(self):
        NB, NBF = 8, 6
        a = CFxp.quantize(0.75,  -0.50, NB, NBF)
        b = CFxp.quantize(0.25,   0.50, NB, NBF)

        # referencia: operar sobre los valores YA cuantizados
        A = a.to_complex()
        B = b.to_complex()

        s = a + b
        d = a - b

        tol = 1e-9
        assert_complex_close(self, s.to_complex(), A + B, tol, "add")
        assert_complex_close(self, d.to_complex(), A - B, tol, "sub")

        # chequeo identidad: (a+b)-b == a (en valor)
        back = (a + b) - b
        assert_complex_close(self, back.to_complex(), A, tol, "add/sub identity")

    def test_conj(self):
        NB, NBF = 8, 6
        a = CFxp.quantize(-1.25, 0.75, NB, NBF)
        A = a.to_complex()

        c = a.conj()
        C = c.to_complex()

        tol = 1e-9
        assert_complex_close(self, C, complex(A.real, -A.imag), tol, "conj")

        # conj(conj(a)) == a (en valor)
        cc = c.conj()
        assert_complex_close(self, cc.to_complex(), A, tol, "double conj")

    def test_mul(self):
        NB, NBF = 8, 6
        a = CFxp.quantize(0.75,  -0.50, NB, NBF)
        b = CFxp.quantize(0.25,   0.50, NB, NBF)

        A = a.to_complex()
        B = b.to_complex()

        p = a * b
        P = p.to_complex()

        tol = 1e-9
        assert_complex_close(self, P, A * B, tol, "mul")

        # (a*b).conj == a.conj * b.conj (propiedad)
        left = (a * b).conj().to_complex()
        right = (a.conj() * b.conj()).to_complex()
        assert_complex_close(self, left, right, tol, "conj mul property")


if __name__ == "__main__":
    unittest.main(verbosity=2)

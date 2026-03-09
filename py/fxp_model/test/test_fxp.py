import unittest, os, random, math
from fxp import Fxp
from fractions import Fraction

# --- progress bar (opcional) ---
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _tqdm(total=None, desc=""):
    """
    Barra de progreso opcional.
    Activala con: TQDM=1 python3 -m unittest -v
    """
    enabled = os.getenv("TQDM", "0") not in ("0", "false", "False", "")
    if (tqdm is None) or (not enabled):
        class Dummy:
            def update(self, n=1): pass
            def close(self): pass
        return Dummy()
    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)

# ----------------------------
# Helpers de referencia
# ----------------------------

def signed_int_edges(NB: int):
    """Valores enteros típicos de borde en two's complement (signed)."""
    q_min = -(1 << (NB - 1))
    q_max =  (1 << (NB - 1)) - 1
    vals = [
        q_min, q_min + 1, q_min + 2,
        -2, -1, 0, 1, 2,
        q_max - 2, q_max - 1, q_max
    ]
    # Filtrar por si NB es muy chico (NB=1 no lo usás, pero igual)
    return [v for v in vals if q_min <= v <= q_max]

def cast_rounding_edges(NB_in: int, NBF_in: int, NB_out: int, NBF_out: int):
    """
    Genera I_in (enteros en el formato NB_in/NBF_in) que fuerzan casos borde de cast:
    - offsets 0, (2^cut-1)
    - offsets alrededor del punto medio de rounding (2^(cut-1))
    - bases cerca de 0 y cerca de los extremos
    """
    assert NBF_out <= NBF_in
    cut = NBF_in - NBF_out
    qmin_in = -(1 << (NB_in - 1))
    qmax_in =  (1 << (NB_in - 1)) - 1

    if cut == 0:
        return signed_int_edges(NB_in)

    # offsets relevantes en los bits que se cortan
    mid = 1 << (cut - 1)
    offsets = {0, (1 << cut) - 1, mid - 1, mid, mid + 1}
    offsets = {o for o in offsets if 0 <= o < (1 << cut)}

    # Elegimos algunos "kept parts" (la parte que queda después del >> cut)
    # Nota: kept está en unidades de 2^cut (o sea I = kept<<cut + offset)
    kept_bits = NB_in - cut
    kept_min = -(1 << (kept_bits - 1))
    kept_max =  (1 << (kept_bits - 1)) - 1

    kept_candidates = {
        kept_min, kept_min + 1,
        -2, -1, 0, 1, 2,
        kept_max - 1, kept_max
    }
    kept_candidates = {k for k in kept_candidates if kept_min <= k <= kept_max}

    cases = set()
    for k in kept_candidates:
        base = k << cut

        for o in offsets:
            # positivo: base + o
            I1 = base + o
            # negativo: base - o (para explorar patrones de bits bajos en negativos)
            I2 = base - o

            if qmin_in <= I1 <= qmax_in:
                cases.add(I1)
            if qmin_in <= I2 <= qmax_in:
                cases.add(I2)

    # También metemos explícitamente extremos absolutos
    cases.update(signed_int_edges(NB_in))
    return sorted(cases)

def bits_to_int(bits):
    """Two's complement bits (MSB primero) -> int con signo."""
    nb = len(bits)
    u = 0
    for b in bits:
        u = (u << 1) | b
    if bits[0] == 1:
        u -= (1 << nb)
    return u

def int_to_bits(x, nb):
    """int con signo -> two's complement bits (MSB primero) de nb bits."""
    mask = (1 << nb) - 1
    u = x & mask
    return [(u >> (nb - 1 - i)) & 1 for i in range(nb)]

def cast_ref_int(I_in, NB_in, NBF_in, NB_out, NBF_out, mode, overflow):
    """
    Referencia entera de tu cast actual (solo NBF_out <= NBF_in).
    - trunc: shift aritmético a derecha (>>), que en Python redondea hacia -inf
    - round: sumar bias positivo (1<<(cut-1)) y luego >> cut
      (tie-break hacia +inf; en negativos eso empuja hacia 0)
    - saturate: clamp
    - wrap: máscara NB_out y reinterpretar signed
    """
    assert NBF_out <= NBF_in
    cut = NBF_in - NBF_out

    if cut == 0:
        I_mid = I_in
    else:
        if mode == "trunc":
            I_mid = I_in >> cut
        elif mode == "round":
            bias = 1 << (cut - 1)
            I_mid = (I_in + bias) >> cut
        else:
            raise ValueError("mode")

    # overflow handling al reducir NB_out
    if overflow == "saturate":
        I_max = (1 << (NB_out - 1)) - 1
        I_min = -(1 << (NB_out - 1))
        I_out = min(max(I_mid, I_min), I_max)
    elif overflow == "wrap":
        # wrap en NB_out bits
        u = I_mid & ((1 << NB_out) - 1)
        if u >= (1 << (NB_out - 1)):
            u -= (1 << NB_out)
        I_out = u
    else:
        raise ValueError("overflow")

    return I_out


class TestFxp(unittest.TestCase):
    def setUp(self):
        random.seed(12345)
        self.N_test = int(os.getenv("N_TEST", "10_000"))

    # --------- básicos de representación ----------


    def test_get_val(self):
        p = _tqdm(total=self.N_test, desc="test_get_val")
        try:
            for _ in range(self.N_test):
                NB  = random.randint(4, 64)
                NBF = random.randint(0, NB-1)
                I = random.randint(-(2**(NB - 1)), (2**(NB - 1)) - 1)

                fxp = Fxp(int_to_bits(I, NB), NB, NBF, signed=True)
                ref = I / (2 ** NBF)

                with self.subTest(NB=NB, NBF=NBF, I=I):
                    self.assertTrue(math.isclose(fxp.get_val(), ref, rel_tol=1e-12, abs_tol=1e-6))
                p.update(1)
        finally:
            p.close()


    def test_quantize_saturates(self):
        p = _tqdm(total=self.N_test, desc="test_quantize_saturates")
        try:
            for _ in range(self.N_test):
                NB  = random.randint(4, 64)
                NBF = random.randint(0, NB-1)
                scale = 2**NBF

                q_max = (2**(NB-1)) - 1
                q_min = -(2**(NB-1))

                x_pos = (q_max + 1000) / scale
                x_neg = (q_min - 1000) / scale

                x_max = Fxp.quantize(x_pos,  NB, NBF, mode="round", signed=True)
                x_min = Fxp.quantize(x_neg, NB, NBF, mode="round", signed=True)

                with self.subTest(NB=NB, NBF=NBF):
                    self.assertEqual(x_max.bits, int_to_bits(q_max, NB))
                    self.assertEqual(x_min.bits, int_to_bits(q_min, NB))
                p.update(1)
        finally:
            p.close()


    # --------- suma ----------
    def test_add(self):
        cases = [(8, 6, 8, 6), (8, 4, 10, 6), (12, 3, 9, 1)]
        total = len(cases) * self.N_test
        bar = _tqdm(total=total, desc="test_add")
        try:
            for NB1, NBF1, NB2, NBF2 in cases:
                for _ in range(self.N_test):
                    I1 = random.randint(-(2**(NB1 - 1)), (2**(NB1 - 1)) - 1)
                    I2 = random.randint(-(2**(NB2 - 1)), (2**(NB2 - 1)) - 1)

                    a = Fxp(int_to_bits(I1, NB1), NB1, NBF1, signed=True)
                    b = Fxp(int_to_bits(I2, NB2), NB2, NBF2, signed=True)
                    s = a + b

                    NBF = max(NBF1, NBF2)
                    I1s = I1 << (NBF - NBF1)
                    I2s = I2 << (NBF - NBF2)
                    I_sum = I1s + I2s

                    NBI1 = NB1 - NBF1
                    NBI2 = NB2 - NBF2
                    NB_ref = NBF + (max(NBI1, NBI2) + 1)

                    self.assertEqual(s.NBF, NBF)
                    self.assertEqual(s.NB, NB_ref)
                    self.assertEqual(s.bits, int_to_bits(I_sum, NB_ref))
                    bar.update(1)
        finally:
            bar.close()


    # --------- multiplicación ----------
    def test_mul(self):
        bar = _tqdm(total=self.N_test, desc="test_mul")
        try:
            for _ in range(self.N_test):
                NB1, NBF1 = 8, 6
                NB2, NBF2 = 8, 6
                I1 = random.randint(-(2**(NB1 - 1)), (2**(NB1 - 1)) - 1)
                I2 = random.randint(-(2**(NB2 - 1)), (2**(NB2 - 1)) - 1)

                a = Fxp(int_to_bits(I1, NB1), NB1, NBF1, signed=True)
                b = Fxp(int_to_bits(I2, NB2), NB2, NBF2, signed=True)
                prod = a * b

                I_prod = I1 * I2
                NBp = NB1 + NB2
                self.assertEqual(prod.NB, NBp)
                self.assertEqual(prod.NBF, NBF1 + NBF2)
                self.assertEqual(prod.bits, int_to_bits(I_prod, NBp))
                bar.update(1)
        finally:
            bar.close()


    # --------- cast (round/trunc + sat/wrap) ----------
    def test_cast(self):
        configs = [
            (16, 8, 12, 4),
            (16, 8, 10, 6),
            (12, 6, 8,  4),
            (10, 4, 6,  2),
        ]
        total = len(configs) * 2 * 2 * self.N_test
        p = _tqdm(total=total, desc="test_cast")
        try:
            for NB_in, NBF_in, NB_out, NBF_out in configs:
                for mode in ("trunc", "round"):
                    for overflow in ("wrap", "saturate"):
                        for _ in range(self.N_test):
                            I = random.randint(-(2**(NB_in - 1)), (2**(NB_in - 1)) - 1)
                            x = Fxp(int_to_bits(I, NB_in), NB_in, NBF_in, signed=True)
                            y = x.cast(NB_out, NBF_out, mode=mode, overflow=overflow)

                            I_ref = cast_ref_int(I, NB_in, NBF_in, NB_out, NBF_out, mode, overflow)
                            with self.subTest(cfg=(NB_in, NBF_in, NB_out, NBF_out),
                                            mode=mode, overflow=overflow, I=I):
                                self.assertEqual(y.bits, int_to_bits(I_ref, NB_out))
                            p.update(1)
        finally:
            p.close()




    def test_cast_tie_break_negative_rounds_toward_zero(self):
        NB_in, NBF_in = 8, 6
        NB_out, NBF_out = 8, 4
        I_in = -50  # -0.78125 * 64

        x = Fxp(int_to_bits(I_in, NB_in), NB_in, NBF_in, signed=True)
        y = x.cast(NB_out, NBF_out, mode="round", overflow="saturate")

        # referencia:
        I_ref = cast_ref_int(I_in, NB_in, NBF_in, NB_out, NBF_out, "round", "saturate")
        self.assertEqual(I_ref, -12)  # -0.75 * 16
        self.assertEqual(y.bits, int_to_bits(I_ref, NB_out))

    def test_cast_saturate_limits(self):
        # Q2.4 (NB=6,NBF=4) -> int range [-32, 31] => real [-2.0, 1.9375]
        NB_in, NBF_in = 8, 4
        NB_out, NBF_out = 6, 4

        # algo muy grande: 3.5
        g = Fxp.quantize(3.5, NB=NB_in, NBF=NBF_in, mode="round", signed=True)
        gs = g.cast(NB_out, NBF_out, mode="round", overflow="saturate")
        self.assertEqual(gs.get_val(), 1.9375)

        # muy chico: -3.5
        n = Fxp.quantize(-3.5, NB=NB_in, NBF=NBF_in, mode="round", signed=True)
        ns = n.cast(NB_out, NBF_out, mode="round", overflow="saturate")
        self.assertEqual(ns.get_val(), -2.0)

    def test_cast_edge_cases(self):
        configs = [
            (16, 8, 12, 4),
            (16, 8, 10, 6),
            (12, 6, 8,  4),
            (10, 4, 6,  2),
        ]

        for NB_in, NBF_in, NB_out, NBF_out in configs:
            edge_I = cast_rounding_edges(NB_in, NBF_in, NB_out, NBF_out)

            for mode in ("trunc", "round"):
                for overflow in ("wrap", "saturate"):
                    for I in edge_I:
                        x = Fxp(int_to_bits(I, NB_in), NB_in, NBF_in, signed=True)
                        y = x.cast(NB_out, NBF_out, mode=mode, overflow=overflow)

                        I_ref = cast_ref_int(I, NB_in, NBF_in, NB_out, NBF_out, mode, overflow)

                        with self.subTest(cfg=(NB_in, NBF_in, NB_out, NBF_out),
                                        mode=mode, overflow=overflow, I=I):
                            self.assertEqual(y.bits, int_to_bits(I_ref, NB_out))

    def test_quantize_round_ties_and_saturation_edges(self):
        # Probamos varios NBF y NB “razonables” para que float no nos arruine el tie
        for NBF in (0, 1, 2, 7, 10, 15):
            NB = max(8, NBF + 4)
            scale = 1 << NBF

            # tie exacto: k + 0.5  -> round "away from zero" por tu implementación
            k = 5
            x_pos = float(Fraction(k, 1) / scale + Fraction(1, 2 * scale))
            x_neg = -x_pos

            y_pos = Fxp.quantize(x_pos, NB, NBF, mode="round", signed=True)
            y_neg = Fxp.quantize(x_neg, NB, NBF, mode="round", signed=True)

            self.assertEqual(bits_to_int(y_pos.bits), k + 1)
            self.assertEqual(bits_to_int(y_neg.bits), -(k + 1))

            # borde de saturación: justo afuera del rango representable
            q_max = (1 << (NB - 1)) - 1
            q_min = -(1 << (NB - 1))

            x_big  = float(Fraction(q_max + 1, scale))
            x_small = float(Fraction(q_min - 1, scale))

            y_big  = Fxp.quantize(x_big,  NB, NBF, mode="round", signed=True)
            y_small = Fxp.quantize(x_small, NB, NBF, mode="round", signed=True)

            self.assertEqual(y_big.bits, int_to_bits(q_max, NB))
            self.assertEqual(y_small.bits, int_to_bits(q_min, NB))


    def test_mul_edge_cases(self):
        # Probamos varios formatos (cambian NBF pero el producto en entero es igual)
        configs = [
            (8, 0),
            (8, 4),
            (8, 6),
            (12, 3),
            (16, 8),
        ]

        for NB, NBF in configs:
            qmin = -(1 << (NB - 1))
            qmax =  (1 << (NB - 1)) - 1

            # Casos borde típicos para multiplicación signed
            edge_vals = [
                qmin, qmin + 1,
                -2, -1, 0, 1, 2,
                qmax - 1, qmax
            ]

            for I1 in edge_vals:
                for I2 in edge_vals:
                    a = Fxp(int_to_bits(I1, NB), NB, NBF, signed=True)
                    b = Fxp(int_to_bits(I2, NB), NB, NBF, signed=True)
                    p = a * b

                    # Referencia exacta en entero (sin saturar, tu mul devuelve NB1+NB2 bits)
                    I_ref = I1 * I2
                    NBp = NB + NB

                    with self.subTest(NB=NB, NBF=NBF, I1=I1, I2=I2):
                        self.assertEqual(p.NB, NBp)
                        self.assertEqual(p.NBF, 2 * NBF)
                        self.assertEqual(p.bits, int_to_bits(I_ref, NBp))

    def test_sub(self):
        cases = [(8, 6, 8, 6), (8, 4, 10, 6), (12, 3, 9, 1)]
        for NB1, NBF1, NB2, NBF2 in cases:
            for _ in range(self.N_test):
                I1 = random.randint(-(1 << (NB1 - 1)), (1 << (NB1 - 1)) - 1)
                I2 = random.randint(-(1 << (NB2 - 1)), (1 << (NB2 - 1)) - 1)

                a = Fxp(int_to_bits(I1, NB1), NB1, NBF1, signed=True)
                b = Fxp(int_to_bits(I2, NB2), NB2, NBF2, signed=True)

                d = a - b

                # referencia numérica (alineando fracción)
                NBF = max(NBF1, NBF2)
                I1s = I1 << (NBF - NBF1)
                I2s = I2 << (NBF - NBF2)
                I_diff = I1s - I2s

                # ancho esperado: igual al add, pero si b es qmin entonces -b crece 1 bit
                NBI1 = NB1 - NBF1
                qmin2 = -(1 << (NB2 - 1))
                extra = 1 if (I2 == qmin2) else 0  # <- CLAVE

                NBI2_eff = (NB2 + extra) - NBF2
                NB_ref = NBF + (max(NBI1, NBI2_eff) + 1)

                with self.subTest(NB1=NB1, NBF1=NBF1, NB2=NB2, NBF2=NBF2, I1=I1, I2=I2):
                    self.assertEqual(d.NBF, NBF)
                    self.assertEqual(d.NB, NB_ref)
                    self.assertEqual(d.bits, int_to_bits(I_diff, NB_ref))

if __name__ == "__main__":
    unittest.main(verbosity=2)

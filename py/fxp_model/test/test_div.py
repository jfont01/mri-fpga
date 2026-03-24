import apytypes as apy
import math


def make_fx(x: float, int_bits: int, frac_bits: int):
    return apy.fx(x, int_bits=int_bits, frac_bits=frac_bits)


def safe_rel_err(ref: float, test: float) -> float:
    if ref == 0.0:
        return abs(test)
    return abs((test - ref) / ref)


def run_real_div_case(
    name: str,
    a_f: float,
    b_f: float,
    int_bits_a: int,
    frac_bits_a: int,
    int_bits_b: int,
    frac_bits_b: int,
):
    a = make_fx(a_f, int_bits_a, frac_bits_a)
    b = make_fx(b_f, int_bits_b, frac_bits_b)

    print("=" * 80)
    print(f"[REAL DIV] {name}")
    print(f"a_f = {a_f}")
    print(f"b_f = {b_f}")
    print(f"a   = {a}  | bits={a.bits} | int_bits={a.int_bits} | frac_bits={a.frac_bits}")
    print(f"b   = {b}  | bits={b.bits} | int_bits={b.int_bits} | frac_bits={b.frac_bits}")

    try:
        c = a / b
        c_f = a_f / b_f

        c_float = float(c)

        print(f"c   = {c}")
        print(f"c_f = {c_f}")
        print(f"c bits={c.bits} | int_bits={c.int_bits} | frac_bits={c.frac_bits}")
        print(f"abs_err = {abs(c_float - c_f):.12e}")
        print(f"rel_err = {safe_rel_err(c_f, c_float):.12e}")

    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")


def run_complex_by_real_case(
    name: str,
    ar_f: float,
    ai_f: float,
    d_f: float,
    int_bits_num: int,
    frac_bits_num: int,
    int_bits_den: int,
    frac_bits_den: int,
):
    ar = make_fx(ar_f, int_bits_num, frac_bits_num)
    ai = make_fx(ai_f, int_bits_num, frac_bits_num)
    d = make_fx(d_f, int_bits_den, frac_bits_den)

    print("=" * 80)
    print(f"[COMPLEX / REAL] {name}")
    print(f"a_f = {ar_f} + j{ai_f}")
    print(f"d_f = {d_f}")
    print(f"ar  = {ar} | bits={ar.bits} | int_bits={ar.int_bits} | frac_bits={ar.frac_bits}")
    print(f"ai  = {ai} | bits={ai.bits} | int_bits={ai.int_bits} | frac_bits={ai.frac_bits}")
    print(f"d   = {d}  | bits={d.bits}  | int_bits={d.int_bits}  | frac_bits={d.frac_bits}")

    try:
        qr = ar / d
        qi = ai / d

        qr_f = ar_f / d_f
        qi_f = ai_f / d_f

        qr_float = float(qr)
        qi_float = float(qi)

        print(f"qr   = {qr}")
        print(f"qi   = {qi}")
        print(f"qr_f = {qr_f}")
        print(f"qi_f = {qi_f}")

        print(f"qr bits={qr.bits} | int_bits={qr.int_bits} | frac_bits={qr.frac_bits}")
        print(f"qi bits={qi.bits} | int_bits={qi.int_bits} | frac_bits={qi.frac_bits}")

        print(f"abs_err_re = {abs(qr_float - qr_f):.12e}")
        print(f"abs_err_im = {abs(qi_float - qi_f):.12e}")
        print(f"rel_err_re = {safe_rel_err(qr_f, qr_float):.12e}")
        print(f"rel_err_im = {safe_rel_err(qi_f, qi_float):.12e}")

    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Casos básicos, en rango y representativos de tu LDLH
    # ------------------------------------------------------------------
    run_real_div_case(
        name="basic_small_over_positive",
        a_f=0.0002514,
        b_f=0.05,
        int_bits_a=1, frac_bits_a=15,
        int_bits_b=1, frac_bits_b=15,
    )

    run_real_div_case(
        name="unity_like_denominator",
        a_f=0.12,
        b_f=0.95,
        int_bits_a=2, frac_bits_a=14,
        int_bits_b=2, frac_bits_b=14,
    )

    # ------------------------------------------------------------------
    # 2) Denominador pequeño positivo
    #    Importante para ver crecimiento de palabra / rango del cociente
    # ------------------------------------------------------------------
    run_real_div_case(
        name="small_positive_denominator",
        a_f=0.01,
        b_f=0.001953125,   # 2^-9
        int_bits_a=2, frac_bits_a=14,
        int_bits_b=1, frac_bits_b=15,
    )

    # ------------------------------------------------------------------
    # 3) Numerador muy chico
    #    Importante para detectar pérdida de precisión / underflow numérico
    # ------------------------------------------------------------------
    run_real_div_case(
        name="tiny_numerator",
        a_f=2.0**-20,
        b_f=0.5,
        int_bits_a=1, frac_bits_a=24,
        int_bits_b=1, frac_bits_b=15,
    )

    # ------------------------------------------------------------------
    # 4) Casos con signo
    #    Útiles para verificar convención de división signed
    # ------------------------------------------------------------------
    run_real_div_case(
        name="negative_over_positive",
        a_f=-0.125,
        b_f=0.25,
        int_bits_a=2, frac_bits_a=14,
        int_bits_b=2, frac_bits_b=14,
    )

    run_real_div_case(
        name="positive_over_negative",
        a_f=0.125,
        b_f=-0.25,
        int_bits_a=2, frac_bits_a=14,
        int_bits_b=2, frac_bits_b=14,
    )

    run_real_div_case(
        name="negative_over_negative",
        a_f=-0.125,
        b_f=-0.25,
        int_bits_a=2, frac_bits_a=14,
        int_bits_b=2, frac_bits_b=14,
    )

    # ------------------------------------------------------------------
    # 5) Casos cerca del máximo rango
    #    Para ver qué formato devuelve APyTypes y si aparece overflow
    # ------------------------------------------------------------------
    run_real_div_case(
        name="near_range_limit",
        a_f=0.95,
        b_f=0.125,
        int_bits_a=1, frac_bits_a=15,
        int_bits_b=1, frac_bits_b=15,
    )

    # ------------------------------------------------------------------
    # 6) División compleja por real positiva
    #    Emula exactamente lo que te interesa para l10, z0, z1
    # ------------------------------------------------------------------
    run_complex_by_real_case(
        name="complex_over_positive_real_1",
        ar_f=0.02,
        ai_f=-0.01,
        d_f=0.5,
        int_bits_num=2, frac_bits_num=14,
        int_bits_den=2, frac_bits_den=14,
    )

    run_complex_by_real_case(
        name="complex_over_small_positive_real",
        ar_f=0.005,
        ai_f=0.003,
        d_f=0.03125,   # 2^-5
        int_bits_num=2, frac_bits_num=18,
        int_bits_den=1, frac_bits_den=15,
    )
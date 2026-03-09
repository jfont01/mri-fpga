import math


class Fxp:
    def __init__(self, bits, NB, NBF, signed=True):
        self.NB = NB
        self.NBF = NBF
        self.NBI = NB - NBF
        self.signed = signed

        if isinstance(bits, list):
            s = ''.join(str(b) for b in bits)
        else:
            s = str(bits).replace('_', '')

        if len(s) != NB:
            raise ValueError(f"Se esperaban {NB} bits, recibidos {len(s)}")

        self.bits = [int(b) for b in s]

    def __repr__(self):
        s = ''.join(str(b) for b in self.bits)
        return f"Fxp(bits='{s}', NB={self.NB}, NBF={self.NBF}, signed={self.signed}, val={self.get_val():.4f})"

    def get_val(self):
        NB  = self.NB
        NBF = self.NBF
        val = 0.0

        for i in range(NB):
            j = NB - i - 1
            b = self.bits[i]

            if self.signed and i == 0:
                val -= b * 2**(j - NBF)
            else:
                val += b * 2**(j - NBF)

        return val
    
    def get_bits_string(self):
        bits = self.bits[:]
        s = "".join(str(b) for b in bits)
        
        if self.NBF > 0:
            s = s[:self.NBI] + "_" + s[self.NBI:]
        
        return s


    ################## Suma binaria ##################
    @staticmethod
    def _sum_bits(num1, NB1, NBF1, num2, NB2, NBF2, signed=True):

        assert 0 <= NBF1 <= NB1 and 0 <= NBF2 <= NB2
        Fxp._assert_bits(num1, NB1)
        Fxp._assert_bits(num2, NB2)
        assert isinstance(signed, bool)

        x = num1[:]
        y = num2[:]

        NBI1 = NB1 - NBF1
        NBI2 = NB2 - NBF2

        NBF = max(NBF1, NBF2)
        NBI = max(NBI1, NBI2) + 1
        NB  = NBF + NBI

        # Zero padding (fracción)
        if NBF > NBF2:
            for _ in range(NBF - NBF2):
                y.append(0)
        if NBF > NBF1:
            for _ in range(NBF - NBF1):
                x.append(0)

        # Sign extension (entera)
        if NBI > NBI2:
            for _ in range(NBI - NBI2):
                y.insert(0, y[0] if signed else 0)
        if NBI > NBI1:
            for _ in range(NBI - NBI1):
                x.insert(0, x[0] if signed else 0)

        
        carry_in = 0
        suma = []
        for i in range(NB - 1, -1, -1):
            bit_x = x[i]
            bit_y = y[i]

            # Full adder
            bit_sum = bit_x ^ bit_y ^ carry_in
            carry_out = (bit_x & bit_y) | (carry_in & (bit_x ^ bit_y))
            carry_in = carry_out

            suma.insert(0, bit_sum)

        Fxp._assert_bits(suma)
        assert len(suma) == NB

        return suma, NB, NBF

    def __add__(self, sumando):
        if not isinstance(sumando, Fxp):
            raise TypeError("Sólo se puede sumar Fxp con Fxp")

        if self.signed != sumando.signed:
            raise ValueError("No se soporta (por ahora) sumar signed con unsigned mezclados")

        bits_sum, NB_res, NBF_res = Fxp._sum_bits(
            self.bits, self.NB, self.NBF,
            sumando.bits, sumando.NB, sumando.NBF,
            signed=self.signed
        )

        return Fxp(bits_sum, NB_res, NBF_res, signed=self.signed)

    def __radd__(self, sumando):
        # Para que funcione sum([fxp1, fxp2, ...]), Python llama 0 + fxp1, y utiliza el operador refelx add
        if sumando == 0:
            return self
        return self.__add__(sumando)
    
    def __sub__(self, restando: "Fxp") -> "Fxp":
        if not isinstance(restando, Fxp):
            return NotImplemented
        if self.signed != restando.signed:
            raise ValueError("No se soporta mezclar signed/unsigned")
        return self + (-restando)

    
    def __neg__(self):
        if not self.signed:
            raise NotImplementedError("Por ahora __neg__ solo para signed=True")

        # Caso especial: q_min
        if (self.bits[0] == 1 and all(b == 0 for b in self.bits[1:])):
            ext = [1] + self.bits[:] 
            neg_bits = Fxp.negate_2s_complement(ext) 
            return Fxp(neg_bits, self.NB + 1, self.NBF, signed=True)

        # Caso general
        neg_bits = Fxp.negate_2s_complement(self.bits)
        return Fxp(neg_bits, self.NB, self.NBF, signed=True)

    

    ################## Producto binario ##################
    @staticmethod
    def negate_2s_complement(bits):
        Fxp._assert_bits(bits)
        NB = len(bits)

        bits = bits[:]
        
        bits = [b ^ 1 for b in bits]
        one = [0] * (len(bits) - 1) + [1]

        neg, _, __ = Fxp._sum_bits(num1=bits, NB1=len(bits), NBF1=0,
                               num2=one, NB2=len(one), NBF2=0,
                               signed=False)
        
        out = neg[1:]
        assert len(out) == NB

        return neg[1:]
    
    @staticmethod
    def _mul_bits(num1, NB1, NBF1, num2, NB2, NBF2, signed=True):
        assert 0 <= NBF1 <= NB1 and 0 <= NBF2 <= NB2
        Fxp._assert_bits(num1, NB1)
        Fxp._assert_bits(num2, NB2)

        x = num1[:]
        y = num2[:]

        NB = NB1 + NB2
        NBF = NBF1 + NBF2

        x_neg, y_neg = False, False
        if x[0] == 1:
            x_neg = True
            x = Fxp.negate_2s_complement(x)
        if y[0] == 1:
            y_neg = True
            y = Fxp.negate_2s_complement(y)

        NB_acc = NB1
        acc = [0] * NB_acc

        for i in range(NB2 - 1, -1, -1):
            sumando = []
            for j in range(NB1):
                bit = y[i] & x[j]
                sumando.append(bit)

            shift = NB2 - 1 - i

            for _ in range(shift):
                sumando.append(0)

            acc, NB_acc, _ = Fxp._sum_bits(num1=sumando, NB1=NB1+shift, NBF1=0,
                                       num2=acc, NB2=NB_acc, NBF2=0, 
                                       signed=False)
            
        if signed and (x_neg ^ y_neg):
            acc = Fxp.negate_2s_complement(acc)

        Fxp._assert_bits(acc)
        assert len(acc) == NB_acc
        return acc, NB, NBF

        
    def __mul__(self, multiplicador):

        if not isinstance(multiplicador, Fxp):
            raise TypeError("Sólo se puede multiplicar Fxp con Fxp")

        if self.signed != multiplicador.signed:
            raise ValueError("No se soporta (por ahora) multiplicar signed con unsigned mezclados")

        bits_prod, NB_prod, NBF_prod = Fxp._mul_bits(
            self.bits, self.NB, self.NBF,
            multiplicador.bits, multiplicador.NB, multiplicador.NBF,
            signed=True
        )
        return Fxp(bits_prod, NB_prod, NBF_prod, signed=self.signed)

    ################## Casteo ##################
    def cast(self, NB_out, NBF_out, mode='round', overflow='saturate'):

        assert isinstance(NB_out, int) and NB_out > 0
        assert isinstance(NBF_out, int) and 0 <= NBF_out <= NB_out
        assert mode in ('round', 'trunc')
        assert overflow in ('saturate', 'wrap')
        assert self.signed is True, "por ahora solo signed=True"

        bits = self.bits[:]
        NB_in  = self.NB
        NBF_in = self.NBF

        if NB_out >= NB_in and NBF_out >= NBF_in:
            return self

        # A partir de acá, sí o sí hay alguna reducción
        if NBF_out > NBF_in:
            raise NotImplementedError("Por ahora solo NBF_out <= NBF_in")


        # 1) Truncar o redondear bits fraccionarios
        cut = NBF_in - NBF_out
        NB_mid = NB_in - cut

        if cut > 0:
            if mode == 'round':
                sign_before = bits[0]
                bits_rounded = Fxp._round(bits, NB_in, NB_mid)

                # Si era positivo y después del round quedó negativo => overflow por carry perdido
                if sign_before == 0 and bits_rounded[0] == 1:
                    bits = [0] + bits_rounded   # conservar carry
                else:
                    bits = bits_rounded
            else:
                bits = Fxp._trunc(bits, NB_in, NB_mid)

        # 2) Sign extend si hace falta
        if NB_out > len(bits):
            bits = [bits[0]] * (NB_out - len(bits)) + bits

        # 3) Overflow al achicar NB
        if overflow == 'saturate' and NB_out < len(bits):
            bits = Fxp._saturate(bits, len(bits), NBF_out, NB_out, NBF_out)

        bits = bits[-NB_out:]  # wrap around / trunc final

        Fxp._assert_bits(bits)
        assert len(bits) == NB_out
        return Fxp(bits, NB_out, NBF_out, signed=self.signed)



    @staticmethod
    def _trunc(bits, NB_in, NB_out):
        bits = bits[:]
        if NB_in <= NB_out:
            return bits
        
        NB_to_cut = NB_in - NB_out
        return bits[:-NB_to_cut]

    @staticmethod
    def _round(bits, NB_in, NB_out):
        bits = bits[:]

        if NB_in <= NB_out:
            return bits
        
        NB_to_cut = NB_in - NB_out

        bias = [0] * NB_out + [1] + [0] * (NB_to_cut - 1)

        rounded, _, __ = Fxp._sum_bits(num1=bits, NB1=NB_in, NBF1=0,
                                       num2=bias, NB2=NB_in, NBF2=0,
                                       signed=True)
        
        if len(rounded) == NB_in + 1:
            rounded = rounded[1:]

        cut = Fxp._trunc(rounded, NB_in, NB_out)

        return cut

    @staticmethod
    def _saturate(bits, NB_in, NBF_in, NB_out, NBF_out):
        bits = bits[:]

        assert NBF_in == NBF_out
        assert NB_out <= NB_in

        NBI_in = NB_in - NBF_in
        NBI_out = NB_out - NBF_out

        if NBI_in == NBI_out:
            return bits
        
        #Detección de overflow
        sign_bit = bits[0]
        overflow = 0
        for i in range(1, NBI_in - NBI_out + 1, 1):
            overflow = overflow | (sign_bit ^ bits[i])

        if overflow:
            if bits[0] == 1:
                bits = [1] + [0] * (NB_out - 1)
            else:
                bits = [0] + [1] * (NB_out - 1)

        return bits
    
    ################## Cuantización ##################
    @classmethod
    def quantize(cls, x, NB, NBF, mode='round', signed=True):
        scale = 2**NBF
        v = x*scale
        
        # 2) redondeo/truncado
        if mode == 'round':
            if v >= 0:
                q = int(v + 0.5)
            else:
                q = int(v - 0.5)
        else:
            q = math.floor(v)

        # 3) Saturación
        if signed:
            q_max = (2**(NB-1)) - 1
            q_min = -(2**(NB-1))
        else:
            q_max = (2**NB) - 1
            q_min = 0

        if q > q_max: q = q_max
        if q < q_min: q = q_min

        mask = (2**NB) - 1
        raw = q & mask

        bits = ""
        for i in range(NB):
            bits += "1" if (raw & (2**(NB - 1 - i))) else "0"

        NBI = NB - NBF
        if NBF > 0:
            bits = bits[:NBI] + "_" + bits[NBI:]


        return cls(bits, NB, NBF, signed)
    

    @staticmethod
    def _assert_bits(bits, NB=None):
        assert isinstance(bits, list), "bits debe ser list[int]"
        assert all(b in (0, 1) for b in bits), "bits debe tener solo 0/1"
        if NB is not None:
            assert len(bits) == NB, f"len(bits)={len(bits)} != NB={NB}"


    @staticmethod
    def _arith_shift_right_bits(bits, k: int):
        """
        Desplazamiento aritmético a la derecha sobre una lista de bits en
        complemento a 2. Mantiene la longitud, rellenando con el bit de signo.
        """
        Fxp._assert_bits(bits)
        if k <= 0:
            return bits[:]

        NB = len(bits)
        sign = bits[0]

        if k >= NB:
            return [sign] * NB

        # k bits de signo por delante, descartando los k LSB
        return [sign] * k + bits[:-k]
    
    def shift_right(self, k: int = 1) -> "Fxp":
        """
        Shift aritmético a la derecha k bits.
        Equivalente (en valor) a dividir por 2**k manteniendo NB y NBF.

        Ejemplo: Q1.15 -> sigue siendo Q1.15, pero con valor más chico.
        """
        if k == 0:
            return self

        new_bits = Fxp._arith_shift_right_bits(self.bits, k)
        # NB y NBF se mantienen
        return Fxp(new_bits, self.NB, self.NBF, signed=self.signed)

    def __rshift__(self, k: int) -> "Fxp":
        """
        Permite usar el operador '>>' directamente:
            y = x >> 1
        """
        if not isinstance(k, int):
            raise TypeError("shift amount debe ser int")
        return self.shift_right(k)



    

def show(tag, x: Fxp):
    print(f"{tag:<24} {x.get_bits_string():<14} val={x.get_val(): .6f}  (NB={x.NB},NBF={x.NBF})")

if __name__ == "__main__":


    a = Fxp("01011000", 8, 6)
    print(a)
    """ 
   # --- cuantización ---
    a = Fxp.quantize(0.75,     NB=8, NBF=6, mode='round', signed=True)
    b = Fxp.quantize(-0.5,     NB=8, NBF=6, mode='round', signed=True)

    # --- operaciones ---
    s = a + b
    p = a * b

    # --- cast: round vs trunc (diferencia clara) ---
    t  = Fxp.quantize(0.78125, NB=8, NBF=6, mode='round', signed=True)  # 0.78125 exacto en Q2.6
    tT = t.cast(NB_out=8, NBF_out=4, mode='trunc', overflow='saturate')
    tR = t.cast(NB_out=8, NBF_out=4, mode='round', overflow='saturate')

    # --- overflow: saturate vs wrap ---
    g  = Fxp.quantize(3.5, NB=8, NBF=4, mode='round', signed=True)      # Q4.4
    gS = g.cast(NB_out=6, NBF_out=4, mode='round', overflow='saturate') # Q2.4 (satura)
    gW = g.cast(NB_out=6, NBF_out=4, mode='round', overflow='wrap')     # Q2.4 (wrap)

    n  = Fxp.quantize(-3.5, NB=8, NBF=4, mode='round', signed=True)
    nS = n.cast(NB_out=6, NBF_out=4, mode='round', overflow='saturate')
    nW = n.cast(NB_out=6, NBF_out=4, mode='round', overflow='wrap')

    show("a = Q(0.75,8,6)", a)
    show("b = Q(-0.5,8,6)", b)
    show("a + b", s)
    show("a * b", p)

    show("t = Q(0.78125,8,6)", t)
    show("t cast trunc ->(8,4)", tT)
    show("t cast round ->(8,4)", tR)

    show("g = Q(3.5,8,4)", g)
    show("g cast sat ->(6,4)", gS)
    show("g cast wrap->(6,4)", gW)

    show("n = Q(-3.5,8,4)", n)
    show("n cast sat ->(6,4)", nS)
    show("n cast wrap->(6,4)", nW)
"""

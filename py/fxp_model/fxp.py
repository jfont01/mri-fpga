from apytypes import APyFixed, QuantizationMode, OverflowMode
import math

DEBUG_FXP_ASSERTS = False

FXP_STATS = {
    "fxp_add": 0,
    "fxp_mul": 0,
    "fxp_sub": 0,
    "sat": 0,
    "underflow": 0,
    }

class Fxp:
    def __init__(self, bits, NB, NBF, signed=True):
        self.NB = NB
        self.NBF = NBF
        self.NBI = NB - NBF
        self.signed = signed

        if isinstance(bits, list):
            bits_list = [int(b) for b in bits]
        else:
            s = str(bits).replace('_', '')
            bits_list = [int(b) for b in s]

        assert len(bits_list) == NB, f"len(bits)={len(bits_list)} != NB={NB}"
        self.bits = bits_list

        bits_str = ''.join(str(b) for b in bits_list)
        value_int = int(bits_str, 2)

        self._val = APyFixed(value_int,
                            bits=NB,
                            int_bits=self.NBI)
        

                
    @staticmethod
    def reset_fxp_stats():
        for k in FXP_STATS:
            FXP_STATS[k] = 0

    @staticmethod
    def get_fxp_stats():
        return dict(FXP_STATS)


    @classmethod
    def from_uint(cls, raw: int, NB: int, NBF: int, signed: bool = True) -> "Fxp":
        bits_str = format(int(raw), f"0{NB}b")
        return cls(bits_str, NB=NB, NBF=NBF, signed=signed)

    def to_uint(self) -> int:
        return int("".join(str(b) for b in self.bits), 2)
    
    def to_hex(self) -> str:
        raw = int("".join(str(b) for b in self.bits), 2)
        nhex = (self.NB + 3) // 4
        return format(raw, f"0{nhex}X")
    
    @classmethod
    def from_float(cls, x: float, NB: int, NBF: int, signed=True):
        NBI = NB - NBF
        val = APyFixed.from_float(x, int_bits=NBI, frac_bits=NBF)
        return cls.from_apyfixed(val, signed=signed)

    

    @classmethod
    def from_apyfixed(cls, val: APyFixed, signed=True):
        """
        Construye un Fxp directamente a partir de un APyFixed existente.
        """
        NB  = val.bits
        NBF = val.frac_bits

        bits_int = val.to_bits()
        bits_str = format(bits_int, f'0{NB}b')
        bits_list = [int(b) for b in bits_str]

        obj = cls.__new__(cls)
        obj.NB = NB
        obj.NBF = NBF
        obj.NBI = NB - NBF
        obj.signed = signed
        obj.bits = bits_list
        obj._val = val
        return obj


    def __repr__(self):
        s = ''.join(str(b) for b in self.bits)
        return f"Fxp(bits='{s}', NB={self.NB}, NBF={self.NBF}, signed={self.signed}, val={self.get_val():.16f})"

    def get_val(self):
        return float(self._val)

    
    def get_bits_string(self):
        bits = self.bits
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

        # Zero padding
        if NBF > NBF2:
            for _ in range(NBF - NBF2):
                y.append(0)
        if NBF > NBF1:
            for _ in range(NBF - NBF1):
                x.append(0)

        # Sign extension
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
            return NotImplemented

        if self.signed != sumando.signed:
            raise ValueError("No se soporta (por ahora) sumar signed con unsigned mezclados")

        res_val = self._val + sumando._val
        FXP_STATS["fxp_add"] += 1
        return Fxp.from_apyfixed(res_val, signed=self.signed)


    def __radd__(self, sumando):
        if sumando == 0:
            return self
        return self.__add__(sumando)

    
    def __sub__(self, restando: "Fxp") -> "Fxp":
        if not isinstance(restando, Fxp):
            return NotImplemented
        if self.signed != restando.signed:
            raise ValueError("No se soporta mezclar signed/unsigned")
        FXP_STATS["fxp_sub"] += 1
        res_val = self._val - restando._val
        return Fxp.from_apyfixed(res_val, signed=self.signed)

    
    def __neg__(self):
        res_val = -self._val
        return Fxp.from_apyfixed(res_val, signed=self.signed)



    

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
            return NotImplemented

        if self.signed != multiplicador.signed:
            raise ValueError("No se soporta (por ahora) multiplicar signed con unsigned mezclados")

        res_val = self._val * multiplicador._val
        FXP_STATS["fxp_mul"] += 1
        return Fxp.from_apyfixed(res_val, signed=self.signed)



    ################## Casteo ##################
    def cast(
        self,
        NB_out: int,
        NBF_out: int,
        mode: str = "round",
        overflow: str = "saturate"
    ) -> "Fxp":
        """
        Recuantiza/castea el valor actual al formato de salida S(NB_out, NBF_out).

        Cuenta:
        - FXP_STATS["sat"]       : si el valor original no entra en el rango del formato destino
        - FXP_STATS["underflow"] : si el valor original es no nulo, entra en rango,
                                pero el resultado cuantizado queda exactamente en 0
        """

        # formato actual
        NB_in = self.NB
        NBF_in = self.NBF
        NBI_in = NB_in - NBF_in

        # formato destino
        NBI_out = NB_out - NBF_out

        # si el formato destino preserva o amplia tanto parte entera como fraccional,
        # el valor es exactamente representable en el nuevo formato
        if NBF_out >= NBF_in and NBI_out >= NBI_in:
            return self

        # mapeo de modos
        qmode_map = {
            "trunc": QuantizationMode.TRN,
            "round": QuantizationMode.TIES_EVEN,
        }
        omode_map = {
            "saturate": OverflowMode.SAT,
            "wrap": OverflowMode.WRAP,
        }

        if mode not in qmode_map:
            raise ValueError(f"Modo de cuantización inválido: {mode}")
        if overflow not in omode_map:
            raise ValueError(f"Modo de overflow inválido: {overflow}")

        # valor real de entrada
        x_in = float(self.get_val())

        # rango representable del formato destino
        if self.signed:
            qmin = -(2 ** (NBI_out - 1))
            qmax = (2 ** (NBI_out - 1)) - (2 ** (-NBF_out))
        else:
            qmin = 0.0
            qmax = (2 ** NBI_out) - (2 ** (-NBF_out))

        # contar saturación si el valor está fuera de rango antes del cast
        saturated = (x_in < qmin) or (x_in > qmax)
        if saturated:
            FXP_STATS["sat"] += 1

        # cast real usando APyFixed
        res_val = self._val.cast(
            int_bits=NBI_out,
            frac_bits=NBF_out,
            quantization=qmode_map[mode],
            overflow=omode_map[overflow],
        )

        x_out = float(res_val)

        # contar underflow sólo si:
        # - el valor de entrada era no nulo
        # - no hubo saturación
        # - el resultado terminó exactamente en 0
        if (x_in != 0.0) and (not saturated) and (x_out == 0.0):
            FXP_STATS["underflow"] += 1

        return Fxp.from_apyfixed(res_val, signed=self.signed)




    
    ################## Cuantización ##################
    @classmethod
    def quantize(cls, x, NB, NBF, mode='round', signed=True):
        scale = 2**NBF
        v = x*scale
        
        # 2) redondeo/truncado
        if mode == 'round':
            if v >= 0:
                q = math.floor(v + 0.5)
            else:
                q = math.ceil(v - 0.5)
        else:
            q = math.floor(v)

        # 3) Saturación
        if signed:
            q_max = (2**(NB-1)) - 1
            q_min = -(2**(NB-1))
        else:
            q_max = (2**NB) - 1
            q_min = 0

        if q > q_max: 
            q = q_max
            FXP_STATS["sat"] += 1
        if q < q_min: 
            q = q_min
            FXP_STATS["sat"] += 1

        mask = (2**NB) - 1
        raw = q & mask

        bits_str = format(raw, f'0{NB}b')
        bits_list = [int(b) for b in bits_str]

        if x != 0.0 and q == 0:
            FXP_STATS["underflow"] += 1

        return cls(bits_list, NB, NBF, signed)

    

    @staticmethod
    def _assert_bits(bits, NB=None):
        if not DEBUG_FXP_ASSERTS:
            return
        assert isinstance(bits, list), "bits debe ser list[int]"
        assert all(b in (0, 1) for b in bits), "bits debe tener solo 0/1"
        if NB is not None:
            assert len(bits) == NB, f"len(bits)={len(bits)} != NB={NB}"



    @staticmethod
    def _arith_shift_right_bits(bits, k: int):
        """
        Desplazamiento aritmético a la derecha sobre una lista de bits en
        complemento a 2.
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
        Shift aritmético a la derecha k bits usando APyFixed.
        """
        if k == 0:
            return self
        if k < 0:
            raise ValueError("k debe ser >= 0")

        res_val = self._val >> k
        return Fxp.from_apyfixed(res_val, signed=self.signed)

    def __rshift__(self, k: int) -> "Fxp":
        if not isinstance(k, int):
            raise TypeError("shift amount debe ser int")
        return self.shift_right(k)



    

def show(tag, x: Fxp):
    print(f"{tag:<24} {x.get_bits_string():<14} val={x.get_val(): .6f}  (NB={x.NB},NBF={x.NBF})")

if __name__ == "__main__":


    a = Fxp("011", 3, 0)
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

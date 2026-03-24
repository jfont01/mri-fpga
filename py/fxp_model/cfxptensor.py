import numpy as np
from numpy.lib.npyio import NpzFile
from typing import Any, Iterable

from fxp import Fxp
from cfxp import CFxp


class CFxpTensor:
    """
    Tensor complejo fixed-point de n dimensiones.

    Internamente almacena:
        - shape
        - data: list[CFxp] en orden row-major (C-order)

    Esta primera versión prioriza claridad del modelo.
    """

    def __init__(
        self,
        data: list[CFxp],
        shape: tuple[int, ...],
    ):
        if len(shape) < 1:
            raise ValueError(f"shape inválido: {shape}")

        total = 1
        for d in shape:
            if d <= 0:
                raise ValueError(f"shape inválido: {shape}")
            total *= d

        if len(data) != total:
            raise ValueError(
                f"Cantidad de elementos inconsistente: len(data)={len(data)} "
                f"pero shape={shape} requiere {total}"
            )

        for i, z in enumerate(data):
            if not isinstance(z, CFxp):
                raise TypeError(f"data[{i}] no es CFxp, es {type(z)}")

        self._data = data
        self._shape = tuple(shape)

    # ------------------------------------------------------------
    # propiedades básicas
    # ------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        total = 1
        for d in self._shape:
            total *= d
        return total

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        return f"CFxpTensor(shape={self.shape}, size={self.size})"
    
    @property
    def NB(self) -> int:
        z0 = self._data[0]
        return z0.re.NB

    @property
    def NBF(self) -> int:
        z0 = self._data[0]
        return z0.re.NBF

    @property
    def signed(self) -> bool:
        z0 = self._data[0]
        return z0.re.signed

    def same_format_as(self, other: "CFxpTensor") -> bool:
        return (
            self.NB == other.NB and
            self.NBF == other.NBF and
            self.signed == other.signed
        )
    # ------------------------------------------------------------
    # indexado interno
    # ------------------------------------------------------------
    def _flat_index(self, idx: tuple[int, ...]) -> int:
        if len(idx) != self.ndim:
            raise IndexError(
                f"Se esperaban {self.ndim} índices, recibido {len(idx)}"
            )

        flat = 0
        stride = 1

        for axis in range(self.ndim - 1, -1, -1):
            i = idx[axis]
            dim = self._shape[axis]

            if not isinstance(i, int):
                raise TypeError("Esta primera versión sólo soporta enteros o slices")

            if i < 0:
                i += dim

            if not (0 <= i < dim):
                raise IndexError(
                    f"Índice fuera de rango en axis={axis}: i={i}, dim={dim}"
                )

            flat += i * stride
            stride *= dim

        return flat

    # ------------------------------------------------------------
    # acceso escalar
    # ------------------------------------------------------------
    def get(self, *idx: int) -> CFxp:
        return self._data[self._flat_index(tuple(idx))]

    def set(self, value: CFxp, *idx: int) -> None:
        if not isinstance(value, CFxp):
            raise TypeError(f"value debe ser CFxp, recibido {type(value)}")
        self._data[self._flat_index(tuple(idx))] = value

    # ------------------------------------------------------------
    # slicing básico
    # ------------------------------------------------------------
    def __getitem__(self, key: Any) -> "CFxpTensor | CFxp":
        if not isinstance(key, tuple):
            key = (key,)

        # completar con ":" implícitos
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        if len(key) != self.ndim:
            raise IndexError(
                f"Cantidad de índices inválida: esperado {self.ndim}, recibido {len(key)}"
            )

        # Caso escalar: todos enteros
        if all(isinstance(k, int) for k in key):
            return self.get(*key)

        # Caso slicing: soporta enteros y slices
        index_lists: list[list[int]] = []
        out_shape: list[int] = []

        for axis, k in enumerate(key):
            dim = self._shape[axis]

            if isinstance(k, int):
                i = k
                if i < 0:
                    i += dim
                if not (0 <= i < dim):
                    raise IndexError(
                        f"Índice fuera de rango en axis={axis}: i={i}, dim={dim}"
                    )
                index_lists.append([i])

            elif isinstance(k, slice):
                rng = list(range(*k.indices(dim)))
                index_lists.append(rng)
                out_shape.append(len(rng))

            else:
                raise TypeError(
                    "Esta primera versión sólo soporta enteros y slices"
                )

        # si todos eran enteros ya se habría ido por el caso escalar
        out_data: list[CFxp] = []

        def rec_build(prefix: list[int], axis: int) -> None:
            if axis == self.ndim:
                out_data.append(self.get(*prefix))
                return
            for i in index_lists[axis]:
                rec_build(prefix + [i], axis + 1)

        rec_build([], 0)

        return CFxpTensor(out_data, tuple(out_shape))

    def __setitem__(self, key: Any, value: CFxp) -> None:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.ndim:
            raise IndexError("Esta primera versión sólo soporta asignación escalar completa")
        if not all(isinstance(k, int) for k in key):
            raise TypeError("Esta primera versión sólo soporta asignación escalar")
        self.set(value, *key)

    # ------------------------------------------------------------
    # constructores
    # ------------------------------------------------------------
    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        NB: int,
        NBF: int,
        signed: bool = True,
    ) -> "CFxpTensor":
        total = 1
        for d in shape:
            total *= d

        z0 = CFxp.from_complex(0.0 + 0.0j, NB=NB, NBF=NBF, signed=signed)
        data = [z0 for _ in range(total)]
        return cls(data=data, shape=shape)

    @classmethod
    def from_npz(cls, src: str | NpzFile) -> "CFxpTensor":
        if isinstance(src, str):
            npz = np.load(src)
            close_after = True
        else:
            npz = src
            close_after = False

        try:
            re_raw = npz["re_raw"]
            im_raw = npz["im_raw"]
            NB = int(npz["NB"])
            NBF = int(npz["NBF"])
            signed = bool(int(npz["signed"]))

            if re_raw.shape != im_raw.shape:
                raise ValueError(
                    f"re_raw e im_raw deben tener el mismo shape: "
                    f"{re_raw.shape} vs {im_raw.shape}"
                )

            flat_data: list[CFxp] = []

            it = np.nditer(re_raw, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                z = CFxp.from_uint_pair(
                    int(re_raw[idx]),
                    int(im_raw[idx]),
                    NB=NB,
                    NBF=NBF,
                    signed=signed,
                )
                flat_data.append(z)

            return cls(flat_data, re_raw.shape)

        finally:
            if close_after:
                npz.close()

    @classmethod
    def from_complex_ndarray(
        cls,
        X: np.ndarray,
        NB: int,
        NBF: int,
        mode: str = "round",
        signed: bool = True,
    ) -> "CFxpTensor":
        X = np.asarray(X)

        if not np.iscomplexobj(X):
            raise ValueError("X debe ser un ndarray complejo")

        flat_data: list[CFxp] = []

        it = np.nditer(X, flags=["multi_index"])
        for z in it:
            flat_data.append(
                CFxp.from_complex(
                    complex(z),
                    NB=NB,
                    NBF=NBF,
                    mode=mode,
                    signed=signed,
                )
            )

        return cls(flat_data, X.shape)



    # ------------------------------------------------------------
    # conversiones
    # ------------------------------------------------------------
    def to_complex_ndarray(self) -> np.ndarray:
        X = np.zeros(self.shape, dtype=np.complex128)

        for flat, z in enumerate(self._data):
            idx = np.unravel_index(flat, self.shape)
            X[idx] = z.to_complex()

        return X

    def to_npz(self, out_path: str, mode: str = "round") -> None:
        # detecta formato desde el primer elemento
        if self.size == 0:
            raise ValueError("Tensor vacío")

        z0 = self._data[0]
        NB = z0.re.NB
        NBF = z0.re.NBF
        signed = z0.re.signed

        # elegir dtype raw
        if NB <= 8:
            raw_dtype = np.uint8
        elif NB <= 16:
            raw_dtype = np.uint16
        elif NB <= 32:
            raw_dtype = np.uint32
        elif NB <= 64:
            raw_dtype = np.uint64
        else:
            raise ValueError(f"NB={NB} no soportado para exportar raw")

        re_raw = np.zeros(self.shape, dtype=raw_dtype)
        im_raw = np.zeros(self.shape, dtype=raw_dtype)

        for flat, z in enumerate(self._data):
            idx = np.unravel_index(flat, self.shape)
            re_u, im_u = z.to_uint()
            re_raw[idx] = re_u
            im_raw[idx] = im_u

        np.savez(
            out_path,
            re_raw=re_raw,
            im_raw=im_raw,
            NB=np.int32(NB),
            NBF=np.int32(NBF),
            signed=np.int32(1 if signed else 0),
            mode=np.array(mode),
            layout=np.array("cfxptensor_flat_list"),
            shape=np.array(self.shape, dtype=np.int32),
        )

    # ------------------------------------------------------------
    # utilidades
    # ------------------------------------------------------------
    def flatten(self) -> list[CFxp]:
        return list(self._data)

    def copy(self) -> "CFxpTensor":
        return CFxpTensor(data=list(self._data), shape=self.shape)
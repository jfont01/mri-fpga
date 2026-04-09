import numpy as np


def uint_to_hex_str(x, nb: int) -> str:
    nhex = (nb + 3) // 4
    return format(int(x), f"0{nhex}x")


def load_npz_raw(npz_path: str):
    z = np.load(npz_path)
    required = ["re_raw", "im_raw", "shape", "NB", "NBF"]
    for k in required:
        if k not in z:
            raise ValueError(f"Falta '{k}' en {npz_path}")

    re_raw = z["re_raw"]
    im_raw = z["im_raw"]
    shape = tuple(int(v) for v in z["shape"])
    NB = int(z["NB"])
    NBF = int(z["NBF"])

    return re_raw, im_raw, shape, NB, NBF


def save_full_tensor_dat(npz_path: str, out_dat_path: str) -> None:
    re_raw, im_raw, shape, NB, NBF = load_npz_raw(npz_path)

    with open(out_dat_path, "w", encoding="utf-8") as f:


        if len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    re_hex = uint_to_hex_str(re_raw[i, j], NB)
                    im_hex = uint_to_hex_str(im_raw[i, j], NB)
                    f.write(f"{i} {j} {re_hex} {im_hex}\n")

        elif len(shape) == 3:
            for c in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        re_hex = uint_to_hex_str(re_raw[c, i, j], NB)
                        im_hex = uint_to_hex_str(im_raw[c, i, j], NB)
                        f.write(f"{c} {i} {j} {re_hex} {im_hex}\n")

        elif len(shape) == 4:
            for a in range(shape[0]):
                for b in range(shape[1]):
                    for i in range(shape[2]):
                        for j in range(shape[3]):
                            re_hex = uint_to_hex_str(re_raw[a, b, i, j], NB)
                            im_hex = uint_to_hex_str(im_raw[a, b, i, j], NB)
                            f.write(f"{a} {b} {i} {j} {re_hex} {im_hex}\n")

        else:
            raise ValueError(f"Solo se soportan tensores 2D, 3D o 4D. shape={shape}")
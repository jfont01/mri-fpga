import argparse
import os
import shutil


def norm_hex(x: str, nb: int) -> str:
    nhex = (nb + 3) // 4
    return format(int(x, 16), f"0{nhex}x")


def normalize_dat_file(
    input_path: str,
    output_path: str,
    nb: int,
) -> None:
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                fout.write(line + "\n")
                continue

            parts = line.split()

            if len(parts) < 4:
                raise ValueError(f"Línea inválida en {input_path}: {line}")

            idx_parts = parts[:-2]
            re_hex = norm_hex(parts[-2], nb)
            im_hex = norm_hex(parts[-1], nb)

            fout.write(" ".join(idx_parts + [re_hex, im_hex]) + "\n")


def safe_copy(src: str, dst: str, nb: int) -> bool:
    if not os.path.exists(src):
        print(f"[vm_runner.py]   {src} doesn't exist. Skipping...")
        return False

    print(f"[vm_runner.py]   Copying {src} to {dst}...")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

    tmp_path = dst + ".tmp"
    normalize_dat_file(dst, tmp_path, nb=nb)
    os.replace(tmp_path, dst)
    return True


def compare_files_exact(rtl_path: str, py_path: str) -> None:
    if not os.path.exists(rtl_path):
        print(f"[vm_runner.py]   {rtl_path} doesn't exist. Skipping VM...")
        return

    if not os.path.exists(py_path):
        print(f"[vm_runner.py]   {py_path} doesn't exist. Skipping VM...")
        return

    with open(rtl_path, "r", encoding="utf-8") as fa:
        rtl_lines = [line.strip() for line in fa if line.strip()]

    with open(py_path, "r", encoding="utf-8") as fb:
        py_lines = [line.strip() for line in fb if line.strip()]

    if rtl_lines == py_lines:
        print(f"[OK] Files match exactly: {os.path.basename(rtl_path)}")
        return

    print(f"[ERROR] Files differ: {os.path.basename(rtl_path)}")

    n = min(len(rtl_lines), len(py_lines))
    for i in range(n):
        if rtl_lines[i] != py_lines[i]:
            print(f"First mismatch at line {i+1}")
            print(f"  py : {py_lines[i]}")
            print(f"  rtl: {rtl_lines[i]}")
            return

    if len(rtl_lines) != len(py_lines):
        print(f"Different number of lines: py={len(py_lines)} rtl={len(rtl_lines)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=["A", "b", "D", "I", "L", "m_hat", "x", "z"],
    )
    parser.add_argument(
        "--NB",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rtl-src",
        type=str,
        required=True,
        help="Path del archivo rtl_<case>.dat generado por Vivado/xsim",
    )
    parser.add_argument(
        "--rtl-dst",
        type=str,
        required=True,
        help="Path destino donde se copiará y normalizará rtl_<case>.dat",
    )
    parser.add_argument(
        "--py-path",
        type=str,
        required=True,
        help="Path del archivo py_<case>.dat contra el que se hará vector matching",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    copied = safe_copy(args.rtl_src, args.rtl_dst, nb=args.NB)
    if not copied:
        return

    compare_files_exact(args.rtl_dst, args.py_path)


if __name__ == "__main__":
    main()
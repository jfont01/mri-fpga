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


def build_paths(vm_root: str, vivado_sim_dir: str, case: str) -> tuple[str, str]:
    rtl_src = os.path.join(vivado_sim_dir, f"rtl_{case}.dat")
    rtl_dst = os.path.join(vm_root, case, f"rtl_{case}.dat")
    py_path = os.path.join(vm_root, case, f"py_{case}.dat")
    return rtl_src, rtl_dst, py_path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vm_root = os.environ.get("VM_ROOT")
    if vm_root is None:
        raise RuntimeError("[ERROR] VM_ROOT not defined")

    vivado_sim_dir = os.environ.get("VIVADO_SIM_DIR")
    if vivado_sim_dir is None:
        raise RuntimeError("[ERROR] VIVADO_SIM_DIR not defined")

    rtl_src, rtl_dst, py_path = build_paths(vm_root, vivado_sim_dir, args.case)

    copied = safe_copy(rtl_src, rtl_dst, nb=args.NB)
    if not copied:
        return

    compare_files_exact(rtl_dst, py_path)


if __name__ == "__main__":
    main()
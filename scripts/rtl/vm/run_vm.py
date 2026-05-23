import argparse
from pathlib import Path
import sys


CASES = ["A", "b", "D", "I", "L", "m_hat", "x", "z"]


def norm_hex(x: str) -> str:
    return format(int(x, 16), "x")


def normalize_line(line: str) -> str:
    line = line.strip()

    if not line:
        return ""

    if line.startswith("#"):
        return line

    parts = line.split()

    if len(parts) < 2:
        raise ValueError(f"Línea inválida: {line}")

    prefix = parts[:-2]
    re_hex = norm_hex(parts[-2]).lower()
    im_hex = norm_hex(parts[-1]).lower()

    return " ".join(prefix + [re_hex, im_hex])


def read_normalized_lines(path: Path) -> list[str]:
    lines: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()

            if not raw:
                continue

            norm = normalize_line(raw)
            if norm:
                lines.append(norm)

    return lines


def compare_case(py_path: Path, rtl_path: Path) -> bool:
    if not py_path.exists():
        print(f"[vm_runner.py]   Missing py file : {py_path}")
        return False

    if not rtl_path.exists():
        print(f"[vm_runner.py]   Missing rtl file: {rtl_path}")
        return False

    py_lines = read_normalized_lines(py_path)
    rtl_lines = read_normalized_lines(rtl_path)

    if py_lines == rtl_lines:
        print(f"[OK] {py_path.stem} == {rtl_path.stem}")
        return True

    print(f"[ERROR] Files differ: {py_path.name} vs {rtl_path.name}")

    n = min(len(py_lines), len(rtl_lines))
    for i in range(n):
        if py_lines[i] != rtl_lines[i]:
            print(f"  First mismatch at line {i+1}")
            print(f"    py : {py_lines[i]}")
            print(f"    rtl: {rtl_lines[i]}")
            return False

    if len(py_lines) != len(rtl_lines):
        print(f"  Different number of lines: py={len(py_lines)} rtl={len(rtl_lines)}")

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track-dir",
        type=str,
        default=".",
        help="Track root. Default: current working directory",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=CASES,
        default=None,
        help="Run vector matching only for one case",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_dir = Path(args.track_dir).resolve()
    py_dir = track_dir / "vectors" / "py"
    rtl_dir = track_dir / "vectors" / "rtl"

    if not py_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {py_dir}")

    if not rtl_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {rtl_dir}")

    cases = [args.case] if args.case is not None else CASES

    all_ok = True

    print(f"[vm_runner.py] Track dir : {track_dir}")
    print(f"[vm_runner.py] PY dir    : {py_dir}")
    print(f"[vm_runner.py] RTL dir   : {rtl_dir}")

    for case in cases:
        py_path = py_dir / f"py_{case}.dat"
        rtl_path = rtl_dir / f"rtl_{case}.dat"

        ok = compare_case(py_path, rtl_path)
        all_ok = all_ok and ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
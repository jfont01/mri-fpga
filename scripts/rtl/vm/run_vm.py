#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class CaseConfig:
    name: str
    py_file: str
    rtl_file: str
    fmt: str


COMPLEX_INDEXED_CASES = {
    "A": CaseConfig(
        name="A",
        py_file="py_A.dat",
        rtl_file="rtl_A.dat",
        fmt="complex_indexed",
    ),
    "b": CaseConfig(
        name="b",
        py_file="py_b.dat",
        rtl_file="rtl_b.dat",
        fmt="complex_indexed",
    ),
    "D": CaseConfig(
        name="D",
        py_file="py_D.dat",
        rtl_file="rtl_D.dat",
        fmt="complex_indexed",
    ),
    "I": CaseConfig(
        name="I",
        py_file="py_I.dat",
        rtl_file="rtl_I.dat",
        fmt="complex_indexed",
    ),
    "L": CaseConfig(
        name="L",
        py_file="py_L.dat",
        rtl_file="rtl_L.dat",
        fmt="complex_indexed",
    ),
    "m_hat": CaseConfig(
        name="m_hat",
        py_file="py_m_hat.dat",
        rtl_file="rtl_m_hat.dat",
        fmt="complex_indexed",
    ),
    "x": CaseConfig(
        name="x",
        py_file="py_x.dat",
        rtl_file="rtl_x.dat",
        fmt="complex_indexed",
    ),
    "z": CaseConfig(
        name="z",
        py_file="py_z.dat",
        rtl_file="rtl_z.dat",
        fmt="complex_indexed",
    ),
}

RTL_READY_CASES = ["A", "b", "div_restoring"]
SUPPORTED_CASES = list(COMPLEX_INDEXED_CASES.keys()) + ["div_restoring", "all"]


def die(msg: str) -> None:
    print(f"[run_vm.py] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def norm_hex(x: str) -> str:
    x = x.strip().lower()

    if x.startswith("0x"):
        x = x[2:]

    if not x:
        raise ValueError("empty hex token")

    return format(int(x, 16), "x")


def normalize_complex_indexed_line(line: str) -> str:
    line = line.strip()

    if not line:
        return ""

    if line.startswith("#"):
        return ""

    parts = line.split()

    if len(parts) < 2:
        raise ValueError(f"invalid complex indexed line: {line}")

    prefix = parts[:-2]
    re_hex = norm_hex(parts[-2])
    im_hex = norm_hex(parts[-1])

    return " ".join(prefix + [re_hex, im_hex])


def normalize_hex_only_line(line: str) -> str:
    line = line.strip()

    if not line:
        return ""

    if line.startswith("#"):
        return ""

    parts = line.split()

    if len(parts) != 1:
        raise ValueError(f"invalid hex-only line: {line}")

    return norm_hex(parts[0])


def normalize_line(line: str, fmt: str) -> str:
    if fmt == "complex_indexed":
        return normalize_complex_indexed_line(line)

    if fmt == "hex_only":
        return normalize_hex_only_line(line)

    raise ValueError(f"unsupported format: {fmt}")


def read_normalized_lines(path: Path, fmt: str) -> list[str]:
    lines: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()

            if not raw:
                continue

            try:
                norm = normalize_line(raw, fmt)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_num}: {exc}") from exc

            if norm:
                lines.append(norm)

    return lines


def compare_case(config: CaseConfig, py_dir: Path, rtl_dir: Path) -> bool:
    py_path = py_dir / config.py_file
    rtl_path = rtl_dir / config.rtl_file

    print(f"[run_vm.py] Comparing case: {config.name}")
    print(f"[run_vm.py]   PY  : {py_path}")
    print(f"[run_vm.py]   RTL : {rtl_path}")
    print(f"[run_vm.py]   fmt : {config.fmt}")

    if not py_path.exists():
        print(f"[run_vm.py]   Missing py file : {py_path}")
        return False

    if not rtl_path.exists():
        print(f"[run_vm.py]   Missing rtl file: {rtl_path}")
        return False

    try:
        py_lines = read_normalized_lines(py_path, config.fmt)
        rtl_lines = read_normalized_lines(rtl_path, config.fmt)
    except ValueError as exc:
        print(f"[run_vm.py]   ERROR while parsing files: {exc}")
        return False

    if py_lines == rtl_lines:
        print(f"[OK] {py_path.name} == {rtl_path.name}")
        return True

    print(f"[ERROR] Files differ: {py_path.name} vs {rtl_path.name}")

    n = min(len(py_lines), len(rtl_lines))

    for i in range(n):
        if py_lines[i] != rtl_lines[i]:
            print(f"  First mismatch at line {i + 1}")
            print(f"    py : {py_lines[i]}")
            print(f"    rtl: {rtl_lines[i]}")
            return False

    if len(py_lines) != len(rtl_lines):
        print(f"  Different number of lines: py={len(py_lines)} rtl={len(rtl_lines)}")

    return False


def find_div_restoring_config(py_dir: Path, rtl_dir: Path, div_mode: str | None) -> CaseConfig:
    if div_mode is not None:
        py_file = f"py_div_restoring_{div_mode}.dat"
        rtl_file = f"rtl_div_restoring_{div_mode}.dat"

        return CaseConfig(
            name="div_restoring",
            py_file=py_file,
            rtl_file=rtl_file,
            fmt="hex_only",
        )

    py_matches = sorted(py_dir.glob("py_div_restoring_*.dat"))

    if not py_matches:
        die(
            "could not find py_div_restoring_*.dat in vectors/py. "
            "Run create_release with TRACK_ENABLE_DIV_RESTORING=1 first."
        )

    if len(py_matches) > 1:
        names = "\n".join(f"  - {p.name}" for p in py_matches)
        die(
            "multiple py_div_restoring_*.dat files found. "
            "Use --div-mode to disambiguate.\n"
            f"{names}"
        )

    py_file = py_matches[0].name

    mode = py_file
    mode = mode.removeprefix("py_div_restoring_")
    mode = mode.removesuffix(".dat")

    rtl_file = f"rtl_div_restoring_{mode}.dat"

    return CaseConfig(
        name="div_restoring",
        py_file=py_file,
        rtl_file=rtl_file,
        fmt="hex_only",
    )


def build_case_config(case: str, py_dir: Path, rtl_dir: Path, div_mode: str | None) -> CaseConfig:
    if case == "div_restoring":
        return find_div_restoring_config(py_dir, rtl_dir, div_mode)

    if case in COMPLEX_INDEXED_CASES:
        return COMPLEX_INDEXED_CASES[case]

    raise ValueError(f"unsupported case: {case}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track-dir",
        type=str,
        default=".",
        help="Track root. Default: current working directory.",
    )

    parser.add_argument(
        "--case",
        type=str,
        choices=SUPPORTED_CASES,
        default="all",
        help=(
            "Case to compare. Default: all. "
            "The all case compares only RTL-ready cases: A, b, div_restoring."
        ),
    )

    parser.add_argument(
        "--div-mode",
        type=str,
        default=None,
        help=(
            "Division mode suffix for div_restoring files. "
            "Example: trunc selects py_div_restoring_trunc.dat and "
            "rtl_div_restoring_trunc.dat. If omitted, it is auto-detected."
        ),
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

    if args.case == "all":
        cases = RTL_READY_CASES
    else:
        cases = [args.case]

    print(f"[run_vm.py] Track dir : {track_dir}")
    print(f"[run_vm.py] PY dir    : {py_dir}")
    print(f"[run_vm.py] RTL dir   : {rtl_dir}")
    print(f"[run_vm.py] Cases     : {', '.join(cases)}")

    all_ok = True

    for case in cases:
        config = build_case_config(
            case=case,
            py_dir=py_dir,
            rtl_dir=rtl_dir,
            div_mode=args.div_mode,
        )

        ok = compare_case(config, py_dir, rtl_dir)
        all_ok = all_ok and ok

    if not all_ok:
        sys.exit(1)

    print("[run_vm.py] All selected vector matching checks passed.")


if __name__ == "__main__":
    main()
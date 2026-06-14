#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".Xil",
    "xsim.dir",
    "simv.daidir",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename a shell variable in all .sh files of a project."
    )

    p.add_argument(
        "old",
        type=str,
        help="Old variable name, e.g. PY_GEN_SCRIPT",
    )

    p.add_argument(
        "new",
        type=str,
        help="New variable name, e.g. RUN_GEN_SH",
    )

    p.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root. Default: current directory.",
    )

    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this flag, only prints what would change.",
    )

    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files when using --apply.",
    )

    p.add_argument(
        "--show-lines",
        action="store_true",
        help="Show matching lines before replacement.",
    )

    return p.parse_args()


def validate_var_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid shell variable name: {name}")


def iter_sh_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in DEFAULT_EXCLUDE_DIRS
        ]

        for fname in filenames:
            if fname.endswith(".sh"):
                yield Path(dirpath) / fname


def replace_var_token(text: str, old: str, new: str) -> tuple[str, int]:
    """
    Token-aware textual replacement.

    Replaces OLD only when it is not part of a larger shell identifier.
    Examples replaced:
      OLD=...
      export OLD=...
      "$OLD"
      "${OLD}"
      "${OLD:-default}"
      source "$OLD"

    Examples not replaced:
      OLD_EXTRA
      EXTRA_OLD
      MY_OLD_VAR
    """
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])"
    )
    new_text, n = pattern.subn(new, text)
    return new_text, n


def main() -> int:
    args = parse_args()

    old = args.old
    new = args.new
    root = Path(args.root).resolve()

    validate_var_name(old)
    validate_var_name(new)

    if not root.is_dir():
        print(f"[rename_sh_var.py] ERROR: root directory not found: {root}")
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[rename_sh_var.py] Mode      : {mode}")
    print(f"[rename_sh_var.py] Root      : {root}")
    print(f"[rename_sh_var.py] Rename    : {old} -> {new}")
    print("")

    total_files = 0
    changed_files = 0
    total_replacements = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in iter_sh_files(root):
        total_files += 1

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[rename_sh_var.py] SKIP non-utf8: {path}")
            continue

        new_text, n = replace_var_token(text, old, new)

        if n == 0:
            continue

        changed_files += 1
        total_replacements += n

        rel = path.relative_to(root)
        print(f"[rename_sh_var.py] {rel}: {n} replacement(s)")

        if args.show_lines:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", line):
                    print(f"  L{lineno}: {line}")

        if args.apply:
            if not args.no_backup:
                backup = path.with_name(path.name + f".bak.{timestamp}")
                shutil.copy2(path, backup)

            path.write_text(new_text, encoding="utf-8")

    print("")
    print(f"[rename_sh_var.py] Scanned files       : {total_files}")
    print(f"[rename_sh_var.py] Changed files       : {changed_files}")
    print(f"[rename_sh_var.py] Total replacements  : {total_replacements}")

    if not args.apply:
        print("")
        print("[rename_sh_var.py] Dry-run only. Re-run with --apply to modify files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
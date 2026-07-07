#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def set_use_color(enabled: bool) -> None:
    global USE_COLOR
    USE_COLOR = bool(enabled)


def color(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"{code}{text}{C.RESET}"


def banner(title: str) -> None:
    line = "=" * 88
    print("")
    print(color(line, C.CYAN))
    print(color(f" {title}", C.BOLD + C.CYAN))
    print(color(line, C.CYAN))


def section(title: str) -> None:
    print("")
    print(color(f"▶ {title}", C.BOLD + C.BLUE))
    print(color("-" * 88, C.GRAY))


def subsection(title: str) -> None:
    print("")
    print(color(f"  • {title}", C.BOLD))


def kv(key: str, value: Any) -> None:
    key_txt = f"{key}:"
    print(f"  {color(f'{key_txt:<24}', C.GRAY)} {value}")


def step_msg(msg: str) -> None:
    print(f"  {color('→', C.CYAN)} {msg}")


def pass_msg(msg: str) -> None:
    print(f"  {color('PASS', C.BOLD + C.GREEN)}  {msg}")


def fail_msg(msg: str) -> None:
    print(f"  {color('FAIL', C.BOLD + C.RED)}  {msg}")


def warn_msg(msg: str) -> None:
    print(f"  {color('WARN', C.BOLD + C.YELLOW)}  {msg}")


def error_line(msg: str) -> None:
    print(f"  {color('ERROR', C.BOLD + C.RED)} {msg}")


def cmd_line(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m {rem:.1f}s"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(json_safe(data), f, indent=2)
        f.write("\n")


def require_env(name: str) -> str:
    value = os.environ.get(name)

    if value is None or value == "":
        raise RuntimeError(f"[ERROR] environment variable not defined: {name}")

    return value


def module_prefix(module_name: str) -> str:
    return module_name.upper()


def detect_current_module() -> tuple[str, Path, str]:
    modules_root = Path(require_env("MODULES_ROOT")).resolve()
    cwd = Path.cwd().resolve()

    try:
        rel = cwd.relative_to(modules_root)
    except ValueError as exc:
        raise RuntimeError(
            f"[ERROR] current directory is not inside MODULES_ROOT\n"
            f"        cwd={cwd}\n"
            f"        MODULES_ROOT={modules_root}"
        ) from exc

    if len(rel.parts) < 1:
        raise RuntimeError("[ERROR] could not detect module name")

    module_name = rel.parts[0]
    module_dir = modules_root / module_name

    if not module_dir.is_dir():
        raise RuntimeError(f"[ERROR] module directory not found: {module_dir}")

    return module_name, module_dir, module_prefix(module_name)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (base_dir / path).resolve()


def load_json(path: Path, label: str = "regression JSON") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] {label} not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"[ERROR] {label} root must be an object")

    return data


def is_define_param_key(key: str) -> bool:
    return re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None


def normalize_dict(obj: Any, label: str) -> dict[str, Any]:
    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise ValueError(f"[ERROR] field '{label}' must be an object")

    return {str(k): v for k, v in obj.items()}


def extract_params(obj: dict[str, Any], meta_keys: set[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}

    nested = obj.get("params", {})
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("[ERROR] field 'params' must be an object")
        params.update(nested)

    for key, value in obj.items():
        if key in meta_keys:
            continue
        params[key] = value

    return params


def split_flow_params_from_params(
    params: dict[str, Any],
    flow_param_keys: set[str],
) -> dict[str, Any]:
    flow_params: dict[str, Any] = {}

    for key in list(params.keys()):
        if key in flow_param_keys:
            flow_params[key] = params.pop(key)

    return flow_params


def cli_key(key: str) -> str:
    if key.upper() == key:
        return "--" + key
    return "--" + key.replace("_", "-")


def run_command(
    cmd: list[str],
    cwd: Path,
    label: str,
    verbose: bool = False,
) -> None:
    step_msg(label)

    if verbose:
        print(color("    command:", C.GRAY))
        print(color(f"    {cmd_line(cmd)}", C.DIM))

    t0 = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        fail_msg(f"{label} ({format_seconds(elapsed)})")

        if result.stdout:
            print("")
            print(color("---- command output ----", C.RED))
            print(result.stdout.rstrip())
            print(color("------------------------", C.RED))

        raise RuntimeError(f"[ERROR] command failed: {label}")

    if verbose and result.stdout:
        print(color("    output:", C.GRAY))

        for line in result.stdout.rstrip().splitlines():
            print(color(f"    {line}", C.DIM))

    step_msg(f"{label} done ({format_seconds(elapsed)})")


def parse_numeric(value: str) -> float | None:
    value = value.strip()

    if value.upper() == "NA":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_key_value_report(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if not path.exists():
        return metrics

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        number = parse_numeric(value)
        metrics[key] = number if number is not None else value

    return metrics


def parse_vivado_used_from_table_line(line: str) -> float | None:
    fields = [x.strip().replace(",", "") for x in line.strip().strip("|").split("|")]

    if len(fields) < 2:
        return None

    used = fields[1]

    try:
        return float(used)
    except ValueError:
        return None


def parse_utilization_report(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "LUT": None,
        "FF": None,
        "DSP": None,
        "BRAM": None,
    }

    if not path.exists():
        return metrics

    bram36: float | None = None
    bram18: float | None = None

    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        if (
            re.search(r"\|\s*Slice LUTs\s*\|", stripped)
            or re.search(r"\|\s*CLB LUTs\s*\|", stripped)
            or re.search(r"\|\s*LUT as Logic\s*\|", stripped)
        ):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                metrics["LUT"] = int(value)

        elif (
            re.search(r"\|\s*Slice Registers\s*\|", stripped)
            or re.search(r"\|\s*CLB Registers\s*\|", stripped)
            or re.search(r"\|\s*Register as Flip Flop\s*\|", stripped)
            or re.search(r"\|\s*FDRE\s*\|", stripped)
        ):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                metrics["FF"] = int(value)

        elif (
            re.search(r"\|\s*DSPs\s*\|", stripped)
            or re.search(r"\|\s*DSP\s*\|", stripped)
            or re.search(r"\|\s*DSP48E2\s*\|", stripped)
        ):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                metrics["DSP"] = int(value)

        elif re.search(r"\|\s*Block RAM Tile\s*\|", stripped):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                metrics["BRAM"] = value

        elif re.search(r"\|\s*RAMB36/FIFO\s*\|", stripped):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                bram36 = value

        elif re.search(r"\|\s*RAMB18\s*\|", stripped):
            value = parse_vivado_used_from_table_line(stripped)
            if value is not None:
                bram18 = value

    if metrics["BRAM"] is None:
        if bram36 is not None or bram18 is not None:
            metrics["BRAM"] = float(bram36 or 0.0) + 0.5 * float(bram18 or 0.0)

    return metrics


def as_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value))
    except ValueError:
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    try:
        return int(float(str(value)))
    except ValueError:
        return None


def resolve_tcl_from_env(
    env_name: str,
    project_root: Path,
    fallback_candidates: list[str],
) -> Path:
    env_value = os.environ.get(env_name)

    if env_value:
        return Path(env_value).resolve()

    for candidate in fallback_candidates:
        path = (project_root / candidate).resolve()
        if path.exists():
            return path

    return (project_root / fallback_candidates[0]).resolve()
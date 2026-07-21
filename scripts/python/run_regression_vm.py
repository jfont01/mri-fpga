#!/usr/bin/env python3

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil
import subprocess

from regression_common import (
    C,
    banner,
    color,
    detect_current_module,
    fail_msg,
    format_seconds,
    is_define_param_key,
    kv,
    load_json,
    normalize_dict,
    now_iso,
    pass_msg,
    require_env,
    resolve_path,
    resolve_tcl_from_env,
    run_command,
    section,
    set_use_color,
    step_msg,
    subsection,
    warn_msg,
    write_json,
)


META_KEYS = {
    "name",
    "enabled",
    "params",
    "defines",
    "rtl_defines",
    "xsim_defines",
    "vm_defines",
    "auto_defines",
    "stimuli_csv",
    "expected_csv",
    "actual_csv",
    "expected_dat_dir",
    "actual_dat_dir",
    "expected_file",   # legacy, ignored
    "actual_file",     # legacy, ignored
}


RUNTIME_PARAM_KEYS = {
    "N_CYCLES",
    "n_cycles",
}


@dataclass(frozen=True)
class RegressionCase:
    name: str
    case_dir: Path
    params: dict[str, Any]
    xsim_defines: dict[str, Any]
    n_cycles: int

    stimuli_csv: Path
    expected_csv: Path
    actual_csv: Path

    expected_dat_dir: Path
    actual_dat_dir: Path


@dataclass(frozen=True)
class CaseResult:
    name: str
    status: str
    vectors: int
    errors: int
    elapsed_s: float
    message: str = ""
    manifest_file: Path | None = None
    report_file: Path | None = None

def _to_native_path(path: str) -> str:

    if os.environ.get("SIM_BACKEND") != "iverilog":
        return path
    if shutil.which("cygpath") is None:
        return path
    try:
        result = subprocess.run(
            ["cygpath", "-m", path],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return path
    
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


def count_csv_vectors(path: Path) -> int:
    if not path.exists():
        return 0

    lines = [line for line in path.read_text().splitlines() if line.strip()]

    if not lines:
        return 0

    return max(0, len(lines) - 1)


def read_text_vector_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] file not found: {path}")

    values: list[str] = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        values.append(line)

    return values


def build_cases_from_json(
    module_name: str,
    module_dir: Path,
    prefix: str,
    regression_json: Path,
) -> tuple[Path, list[RegressionCase]]:
    data = load_json(regression_json, "VM regression JSON")

    regression_dir = resolve_path(
        data.get("regression_dir", "build"),
        module_dir,
    )

    defaults = normalize_dict(data.get("defaults", {}), "defaults")

    default_params = extract_params(defaults, META_KEYS)

    default_xsim_defines: dict[str, Any] = {}
    default_xsim_defines.update(normalize_dict(defaults.get("defines", {}), "defaults.defines"))
    default_xsim_defines.update(normalize_dict(defaults.get("rtl_defines", {}), "defaults.rtl_defines"))
    default_xsim_defines.update(normalize_dict(defaults.get("xsim_defines", {}), "defaults.xsim_defines"))
    default_xsim_defines.update(normalize_dict(defaults.get("vm_defines", {}), "defaults.vm_defines"))

    default_auto_defines = bool(defaults.get("auto_defines", True))

    raw_cases = data.get("cases")
    if raw_cases is None:
        raise ValueError("[ERROR] VM regression JSON missing required field: cases")
    if not isinstance(raw_cases, list):
        raise ValueError("[ERROR] field 'cases' must be a list")

    cases: list[RegressionCase] = []
    seen: set[str] = set()

    for idx, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"[ERROR] case #{idx} must be an object")

        if not raw_case.get("enabled", True):
            continue

        case_name = raw_case.get("name")
        if case_name is None or str(case_name).strip() == "":
            raise ValueError(f"[ERROR] case #{idx} has no name")

        case_name = str(case_name)

        if case_name in seen:
            raise ValueError(f"[ERROR] duplicated VM regression case name: {case_name}")
        seen.add(case_name)

        params = dict(default_params)
        params.update(extract_params(raw_case, META_KEYS))

        auto_defines = bool(raw_case.get("auto_defines", default_auto_defines))

        xsim_defines = dict(default_xsim_defines)
        xsim_defines.update(normalize_dict(raw_case.get("defines", {}), f"{case_name}.defines"))
        xsim_defines.update(normalize_dict(raw_case.get("rtl_defines", {}), f"{case_name}.rtl_defines"))
        xsim_defines.update(normalize_dict(raw_case.get("xsim_defines", {}), f"{case_name}.xsim_defines"))
        xsim_defines.update(normalize_dict(raw_case.get("vm_defines", {}), f"{case_name}.vm_defines"))

        if auto_defines:
            for key, value in params.items():
                key = str(key)

                if key in RUNTIME_PARAM_KEYS:
                    continue

                if is_define_param_key(key):
                    xsim_defines.setdefault(f"{prefix}_{key}", value)

        n_cycles_value = (
            raw_case.get("N_CYCLES",
            raw_case.get("n_cycles",
            defaults.get("N_CYCLES",
            defaults.get("n_cycles", 16))))
        )

        try:
            n_cycles = int(n_cycles_value)
        except Exception as exc:
            raise ValueError(
                f"[ERROR] case {case_name}: N_CYCLES must be an integer"
            ) from exc

        if n_cycles < 0:
            raise ValueError(f"[ERROR] case {case_name}: N_CYCLES must be non-negative")

        case_dir = regression_dir / case_name

        stimuli_csv = resolve_path(
            raw_case.get(
                "stimuli_csv",
                defaults.get("stimuli_csv", "simulation/vectors/stimuli/in_ports.csv"),
            ),
            case_dir,
        )

        expected_csv = resolve_path(
            raw_case.get(
                "expected_csv",
                defaults.get("expected_csv", "simulation/vectors/expected/out_ports.csv"),
            ),
            case_dir,
        )

        actual_csv = resolve_path(
            raw_case.get(
                "actual_csv",
                defaults.get("actual_csv", "simulation/vectors/actual/out_ports.csv"),
            ),
            case_dir,
        )

        expected_dat_dir = resolve_path(
            raw_case.get(
                "expected_dat_dir",
                defaults.get("expected_dat_dir", "simulation/vectors/expected/out_ports"),
            ),
            case_dir,
        )

        actual_dat_dir = resolve_path(
            raw_case.get(
                "actual_dat_dir",
                defaults.get("actual_dat_dir", "simulation/vectors/actual/out_ports"),
            ),
            case_dir,
        )

        cases.append(
            RegressionCase(
                name=case_name,
                case_dir=case_dir,
                params=params,
                xsim_defines=xsim_defines,
                n_cycles=n_cycles,
                stimuli_csv=stimuli_csv,
                expected_csv=expected_csv,
                actual_csv=actual_csv,
                expected_dat_dir=expected_dat_dir,
                actual_dat_dir=actual_dat_dir,
            )
        )

    if not cases:
        raise RuntimeError("[ERROR] no enabled VM regression cases found")

    return regression_dir, cases


def validate_xsim_inputs(case: RegressionCase) -> None:
    missing: list[Path] = []

    if not case.stimuli_csv.exists():
        missing.append(case.stimuli_csv)

    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "[ERROR] missing XSIM input artifacts. "
            "Run run_regression_sim for this case first:\n" + msg
        )


def validate_compare_inputs(case: RegressionCase) -> None:
    missing: list[Path] = []

    if not case.expected_csv.exists():
        missing.append(case.expected_csv)

    if not case.actual_csv.exists():
        missing.append(case.actual_csv)

    if not case.expected_dat_dir.is_dir():
        missing.append(case.expected_dat_dir)

    if not case.actual_dat_dir.is_dir():
        missing.append(case.actual_dat_dir)

    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "[ERROR] missing comparison artifacts:\n" + msg
        )

def _xsim_define_args(defines: dict[str, Any]) -> list[str]:
    """Defines en formato XSIM: -d KEY=VALUE"""
    args: list[str] = []
    for key, value in defines.items():
        args.append("-d")
        args.append(f"{key}={value}")
    return args


def _iverilog_args_from_flist(tb_flist: Path) -> list[str]:
    args: list[str] = []
    for raw in tb_flist.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-i "):
            incdir = line[len("-i "):].strip()
            args.append(f"-I{_to_native_path(incdir)}")
        else:
            args.append(_to_native_path(line))
    return args


def _run_xsim_backend(
    case: RegressionCase,
    module_dir: Path,
    project_root: Path,
    verbose: bool,
    waves: bool = False,
) -> None:
    run_xsim_tcl = resolve_tcl_from_env(
        "RUN_XSIM_TCL",
        project_root,
        ["scripts/tcl/run_xsim.tcl", "scripts/run_xsim.tcl"],
    )
    if not run_xsim_tcl.exists():
        raise FileNotFoundError(f"[ERROR] run_xsim.tcl not found: {run_xsim_tcl}")

    define_args = [f"{key}={value}" for key, value in case.xsim_defines.items()]
    runtime_args = [f"N_CYCLES={case.n_cycles}"]
    if waves:
        runtime_args.append("WAVES=1")

    vivado_log_dir = case.case_dir / "simulation" / "vivado"
    vivado_log_dir.mkdir(parents=True, exist_ok=True)
    vivado_journal = vivado_log_dir / "vivado.jou"
    vivado_log = vivado_log_dir / "vivado.log"

    cmd = [
        "vivado", "-mode", "batch", "-notrace",
        "-journal", str(vivado_journal),
        "-log", str(vivado_log),
        "-source", str(run_xsim_tcl),
        "-tclargs",
        str(case.case_dir),
        str(module_dir),
        *define_args,
        *runtime_args,
    ]
    run_command(cmd=cmd, cwd=module_dir, label="run xsim", verbose=verbose)


def _run_iverilog_backend(
    case: RegressionCase,
    module_dir: Path,
    prefix: str,
    verbose: bool,
    waves: bool = False,
) -> None:
    # flist del testbench (lo genera update_flist)
    tb_flist_env = os.environ.get(f"{prefix}_TB_FLIST")
    if tb_flist_env:
        tb_flist = Path(tb_flist_env).resolve()
    else:
        tb_flist = (module_dir / "flist" / f"{module_dir.name}_tb.flist").resolve()

    if not tb_flist.exists():
        raise FileNotFoundError(
            f"[ERROR] tb flist not found: {tb_flist}. Run update_flist first."
        )

    # El testbench escribe en actual/out_ports (Icarus no tiene $system).
    actual_dir = case.case_dir / "simulation" / "vectors" / "actual" / "out_ports"
    actual_dir.mkdir(parents=True, exist_ok=True)

    xsim_dir = case.case_dir / "simulation" / "xsim"
    xsim_dir.mkdir(parents=True, exist_ok=True)
    vvp_out = xsim_dir / "sim.vvp"

    # defines en formato iverilog: -D KEY=VALUE
    define_args: list[str] = []
    for key, value in case.xsim_defines.items():
        define_args.append("-D")
        define_args.append(f"{key}={value}")

    flist_args = _iverilog_args_from_flist(tb_flist)

    compile_cmd = [
            "iverilog", "-g2012",
            "-o", _to_native_path(str(vvp_out)),
            *define_args, *flist_args,
        ]
    run_command(cmd=compile_cmd, cwd=module_dir, label="compile (iverilog)", verbose=verbose)

    # plusargs: mismos nombres que XSIM (CASE_DIR / N_CYCLES)
    run_cmd = [
            "vvp", _to_native_path(str(vvp_out)),
            f"+CASE_DIR={_to_native_path(str(case.case_dir))}",
            f"+N_CYCLES={case.n_cycles}",
        ]
    if waves:
        run_cmd.append("+WAVES")

    run_command(cmd=run_cmd, cwd=module_dir, label="run (vvp)", verbose=verbose)

    if waves:
        step_msg(f"waves: {case.case_dir / 'simulation' / 'waves.vcd'}")


def run_sim_backend(
    case: RegressionCase,
    module_dir: Path,
    project_root: Path,
    prefix: str,
    verbose: bool = False,
    waves: bool = False,
) -> None:
    """
    Ejecuta la simulación RTL con el backend indicado por SIM_BACKEND
    (exportado por set_env.sh): 'xsim' en Linux, 'iverilog' en msys.
    """
    backend = os.environ.get("SIM_BACKEND", "").strip().lower()

    if backend == "xsim":
        _run_xsim_backend(case, module_dir, project_root, verbose, waves)
    elif backend == "iverilog":
        _run_iverilog_backend(case, module_dir, prefix, verbose, waves)
    else:
        raise RuntimeError(
            f"[ERROR] SIM_BACKEND inválido o no definido: '{backend}'. "
            f"Esperado 'xsim' o 'iverilog'. ¿Sourceaste set_env.sh?"
        )

def compare_dat_files(
    case: RegressionCase,
    max_errors: int = 10,
) -> tuple[int, int, Path]:
    step_msg("compare expected vs RTL actual DAT files")

    validate_compare_inputs(case)

    t0 = time.perf_counter()

    expected_files = sorted(case.expected_dat_dir.glob("*.dat"))

    if not expected_files:
        raise FileNotFoundError(f"[ERROR] no expected DAT files found in: {case.expected_dat_dir}")

    errors = 0
    total_vectors = 0
    signal_reports: list[dict[str, Any]] = []

    for expected_file in expected_files:
        signal_name = expected_file.stem
        actual_file = case.actual_dat_dir / expected_file.name

        signal_errors = 0
        mismatches: list[tuple[int, str, str]] = []

        if not actual_file.exists():
            errors += 1
            signal_reports.append({
                "signal": signal_name,
                "expected_file": expected_file,
                "actual_file": actual_file,
                "expected_count": 0,
                "actual_count": 0,
                "compared_count": 0,
                "line_count_match": False,
                "errors": 1,
                "message": "actual file missing",
                "mismatches": [],
            })
            continue

        expected = read_text_vector_file(expected_file)
        actual = read_text_vector_file(actual_file)

        line_count_match = len(expected) == len(actual)

        if not line_count_match:
            signal_errors += 1

        n = min(len(expected), len(actual))
        total_vectors = max(total_vectors, len(expected))

        for idx in range(n):
            if expected[idx] != actual[idx]:
                if len(mismatches) < max_errors:
                    mismatches.append((idx, expected[idx], actual[idx]))
                signal_errors += 1

        errors += signal_errors

        signal_reports.append({
            "signal": signal_name,
            "expected_file": expected_file,
            "actual_file": actual_file,
            "expected_count": len(expected),
            "actual_count": len(actual),
            "compared_count": n,
            "line_count_match": line_count_match,
            "errors": signal_errors,
            "message": "",
            "mismatches": mismatches,
        })

    elapsed = time.perf_counter() - t0

    rpt_file = case.case_dir / "reports" / "vector_match.rpt"
    rpt_file.parent.mkdir(parents=True, exist_ok=True)

    with rpt_file.open("w") as f:
        f.write("VECTOR MATCH REPORT\n")
        f.write("===================\n\n")

        f.write(f"generated_at : {now_iso()}\n")
        f.write(f"case         : {case.name}\n")
        f.write(f"case_dir     : {case.case_dir}\n")
        f.write(f"expected_dir : {case.expected_dat_dir}\n")
        f.write(f"actual_dir   : {case.actual_dat_dir}\n")
        f.write(f"errors       : {errors}\n")
        f.write(f"elapsed      : {format_seconds(elapsed)}\n\n")

        f.write("PARAMETERS\n")
        f.write("----------\n")
        if case.params:
            for key, value in case.params.items():
                f.write(f"{key:<24}: {value}\n")
        else:
            f.write("No parameters.\n")

        f.write("\nXSIM DEFINES\n")
        f.write("------------\n")
        if case.xsim_defines:
            for key, value in case.xsim_defines.items():
                f.write(f"{key:<24}: {value}\n")
        else:
            f.write("No XSIM defines.\n")

        f.write("\n")

        for item in signal_reports:
            f.write(f"SIGNAL {item['signal']}\n")
            f.write("-" * (7 + len(item["signal"])) + "\n")
            f.write(f"expected_file    : {item['expected_file']}\n")
            f.write(f"actual_file      : {item['actual_file']}\n")
            f.write(f"expected_count   : {item['expected_count']}\n")
            f.write(f"actual_count     : {item['actual_count']}\n")
            f.write(f"compared_count   : {item['compared_count']}\n")
            f.write(f"line_count_match : {item['line_count_match']}\n")
            f.write(f"errors           : {item['errors']}\n")

            if item["message"]:
                f.write(f"message          : {item['message']}\n")

            if item["mismatches"]:
                f.write("\n")
                f.write(f"{'idx':>10} {'expected':>24} {'actual':>24}\n")
                f.write(f"{'-' * 10} {'-' * 24} {'-' * 24}\n")

                for idx, exp, act in item["mismatches"]:
                    f.write(f"{idx:10d} {exp:>24} {act:>24}\n")

            f.write("\n")

    if errors == 0:
        pass_msg(f"{case.name}: DAT files match")
    else:
        fail_msg(f"{case.name}: errors={errors}")

    step_msg(f"vector match report: {rpt_file}")

    return errors, total_vectors, rpt_file


def write_case_manifest(
    case: RegressionCase,
    module_name: str,
    module_dir: Path,
    regression_json: Path,
    regression_dir: Path,
    status: str,
    vectors: int,
    errors: int,
    elapsed_s: float,
    message: str = "",
    report_file: Path | None = None,
) -> Path:
    manifest_file = case.case_dir / "manifest.json"

    data = {
        "generated_at": now_iso(),
        "module": {
            "name": module_name,
            "dir": module_dir,
        },
        "regression": {
            "json": regression_json,
            "dir": regression_dir,
            "type": "vm",
            "engine": "vivado_xsim",
        },
        "case": {
            "name": case.name,
            "dir": case.case_dir,
            "params": case.params,
            "xsim_defines": case.xsim_defines,
            "n_cycles": case.n_cycles,
        },
        "artifacts": {
            "manifest_file": manifest_file,
            "vector_match_report": report_file,
            "vivado_log": case.case_dir / "simulation" / "vivado" / "vivado.log",
            "vivado_journal": case.case_dir / "simulation" / "vivado" / "vivado.jou",
            "xvlog_log": case.case_dir / "simulation" / "xsim" / "xvlog.log",
            "xelab_log": case.case_dir / "simulation" / "xsim" / "xelab.log",
            "xsim_log": case.case_dir / "simulation" / "xsim" / "xsim.log",
            "stimuli_csv": case.stimuli_csv,
            "expected_csv": case.expected_csv,
            "actual_csv": case.actual_csv,
            "expected_dat_dir": case.expected_dat_dir,
            "actual_dat_dir": case.actual_dat_dir,
        },
        "result": {
            "status": status,
            "vectors": vectors,
            "errors": errors,
            "elapsed_s": elapsed_s,
            "elapsed": format_seconds(elapsed_s),
            "message": message,
        },
    }

    write_json(manifest_file, data)
    return manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_regression_vm",
        description="Run RTL/XSIM regressions for the current module using vectors generated by run_regression_sim.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--case", action="append", default=[], metavar="CASE_NAME")
    parser.add_argument("--list-cases", action="store_true")

    parser.add_argument("--skip-gen", action="store_true",
                        help="Deprecated/no-op. Stimuli are generated by run_regression_sim.")
    parser.add_argument("--skip-xsim", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--keep-going", action="store_true")

    parser.add_argument("--waves", action="store_true",
                        help="Genera ondas VCD en <case_dir>/simulation/waves.vcd "
                             "(el testbench debe soportar el plusarg +WAVES).")

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def print_case_info(case: RegressionCase) -> None:
    section(f"CASE {case.name}")

    kv("case_dir", case.case_dir)
    kv("n_cycles", case.n_cycles)
    kv("stimuli_csv", case.stimuli_csv)
    kv("expected_csv", case.expected_csv)
    kv("actual_csv", case.actual_csv)
    kv("expected_dat_dir", case.expected_dat_dir)
    kv("actual_dat_dir", case.actual_dat_dir)

    if case.params:
        subsection("parameters")
        for key, value in case.params.items():
            kv(str(key), value)

    if case.xsim_defines:
        subsection("xsim defines")
        for key, value in case.xsim_defines.items():
            kv(str(key), value)


def print_results_summary(
    results: list[CaseResult],
    total_elapsed_s: float,
    selected_cases: int,
) -> None:
    banner("VM REGRESSION SUMMARY")

    cases_run = len(results)
    cases_not_run = selected_cases - cases_run

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")

    total_vectors = sum(r.vectors for r in results)
    total_errors = sum(r.errors for r in results)

    kv("cases_selected", selected_cases)
    kv("cases_run", cases_run)
    kv("cases_not_run", cases_not_run)
    kv("cases_passed", passed)
    kv("cases_failed", failed)
    kv("total_vectors", total_vectors)
    kv("total_errors", total_errors)
    kv("elapsed", format_seconds(total_elapsed_s))

    print("")
    print(color("  Per-case results", C.BOLD))
    print(color("  " + "-" * 104, C.GRAY))
    print(color(f"  {'status':<8} {'case':<48} {'vectors':>10} {'errors':>10} {'time':>10}", C.GRAY))
    print(color("  " + "-" * 104, C.GRAY))

    for result in results:
        status = (
            color(f"{'PASS':<8}", C.BOLD + C.GREEN)
            if result.status == "PASS"
            else color(f"{'FAIL':<8}", C.BOLD + C.RED)
        )

        print(
            f"  {status} "
            f"{result.name:<48} "
            f"{result.vectors:>10} "
            f"{result.errors:>10} "
            f"{format_seconds(result.elapsed_s):>10}"
        )

        if result.message:
            print(color(f"           {result.message}", C.GRAY))

    print(color("  " + "-" * 104, C.GRAY))

    if failed == 0:
        pass_msg("VM regression completed successfully")
    else:
        fail_msg("VM regression failed")


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    project_root = Path(require_env("PROJECT_ROOT")).resolve()
    module_name, module_dir, prefix = detect_current_module()

    regression_json_env = (
        os.environ.get(f"{prefix}_TB_REGRESSION_JSON")
        or os.environ.get(f"{prefix}_REGRESSION_JSON")
    )

    if regression_json_env:
        regression_json = resolve_path(regression_json_env, module_dir)
    else:
        regression_json = module_dir / "testbench" / f"{module_name}_tb_regression.json"

    regression_dir, cases = build_cases_from_json(
        module_name=module_name,
        module_dir=module_dir,
        prefix=prefix,
        regression_json=regression_json,
    )

    if args.case:
        selected = set(args.case)
        available = {case.name for case in cases}
        missing = selected - available

        if missing:
            raise ValueError(f"[ERROR] unknown regression case(s): {sorted(missing)}")

        cases = [case for case in cases if case.name in selected]

    if args.list_cases:
        banner("AVAILABLE VM REGRESSION CASES")
        kv("module", module_name)
        kv("regression_json", regression_json)
        kv("regression_dir", regression_dir)
        print("")
        for case in cases:
            print(f"  - {case.name}")
        return 0

    if not cases:
        raise RuntimeError("[ERROR] no regression cases selected")

    banner("VM REGRESSION")

    kv("module", module_name)
    kv("module_dir", module_dir)
    kv("regression_json", regression_json)
    kv("regression_dir", regression_dir)
    kv("selected_cases", len(cases))

    if args.skip_gen:
        warn_msg("--skip-gen is deprecated and ignored. Use run_regression_sim to generate vectors.")
    if args.skip_xsim:
        warn_msg("XSIM disabled (--skip-xsim)")
    if args.skip_compare:
        warn_msg("vector comparison disabled (--skip-compare)")

    t_regression_start = time.perf_counter()
    results: list[CaseResult] = []

    for case in cases:
        print_case_info(case)

        case_t0 = time.perf_counter()
        status = "FAIL"
        message = ""
        vectors = 0
        errors = 0
        report_file: Path | None = None
        manifest_file: Path | None = None

        try:
            if not args.skip_xsim:
                validate_xsim_inputs(case)

                run_sim_backend(
                    case=case,
                    module_dir=module_dir,
                    project_root=project_root,
                    prefix=prefix,
                    verbose=args.verbose,
                    waves=args.waves,
                )

            if not args.skip_compare:
                errors, vectors, report_file = compare_dat_files(case)
                status = "PASS" if errors == 0 else "FAIL"
                message = "" if errors == 0 else f"errors={errors}"
            else:
                status = "PASS"
                message = "comparison skipped"
                vectors = count_csv_vectors(case.actual_csv)

            case_elapsed = time.perf_counter() - case_t0

            manifest_file = write_case_manifest(
                case=case,
                module_name=module_name,
                module_dir=module_dir,
                regression_json=regression_json,
                regression_dir=regression_dir,
                status=status,
                vectors=vectors,
                errors=errors,
                elapsed_s=case_elapsed,
                message=message,
                report_file=report_file,
            )

            step_msg(f"manifest: {manifest_file}")

            if status == "PASS":
                pass_msg(case.name)
            else:
                fail_msg(f"{case.name}: {message}")

            results.append(
                CaseResult(
                    name=case.name,
                    status=status,
                    vectors=vectors,
                    errors=errors,
                    elapsed_s=case_elapsed,
                    message=message,
                    manifest_file=manifest_file,
                    report_file=report_file,
                )
            )

            if status == "FAIL" and not args.keep_going:
                break

        except Exception as exc:
            case_elapsed = time.perf_counter() - case_t0
            message = str(exc)

            fail_msg(f"{case.name}: {message}")

            try:
                manifest_file = write_case_manifest(
                    case=case,
                    module_name=module_name,
                    module_dir=module_dir,
                    regression_json=regression_json,
                    regression_dir=regression_dir,
                    status="FAIL",
                    vectors=vectors,
                    errors=errors,
                    elapsed_s=case_elapsed,
                    message=message,
                    report_file=report_file,
                )
                step_msg(f"manifest: {manifest_file}")
            except Exception as report_exc:
                warn_msg(f"could not write manifest for {case.name}: {report_exc}")

            results.append(
                CaseResult(
                    name=case.name,
                    status="FAIL",
                    vectors=vectors,
                    errors=errors,
                    elapsed_s=case_elapsed,
                    message=message,
                    manifest_file=manifest_file,
                    report_file=report_file,
                )
            )

            if not args.keep_going:
                break

    elapsed = time.perf_counter() - t_regression_start

    print_results_summary(
        results=results,
        total_elapsed_s=elapsed,
        selected_cases=len(cases),
    )

    failed_cases = sum(1 for r in results if r.status == "FAIL")
    return 0 if failed_cases == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
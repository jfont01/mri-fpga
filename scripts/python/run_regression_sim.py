#!/usr/bin/env python3

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regression_common import (
    C,
    banner,
    cli_key,
    color,
    detect_current_module,
    error_line,
    extract_params,
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
    "cpp_defines",
    "sim_defines",
    "auto_defines",
    "binary_file",
}

RUNTIME_PARAM_KEYS = {
    "N_CYCLES",
    "n_cycles",
}


@dataclass(frozen=True)
class SimCase:
    name: str
    case_dir: Path
    params: dict[str, Any]
    cpp_defines: dict[str, Any]
    n_cycles: int
    binary_file: Path
    in_ports_csv: Path
    out_ports_csv: Path


@dataclass(frozen=True)
class SimResult:
    name: str
    status: str
    vectors: int
    elapsed_s: float
    message: str = ""
    manifest_file: Path | None = None
    summary_report: Path | None = None
    binary_file: Path | None = None
    in_ports_csv: Path | None = None
    out_ports_csv: Path | None = None


def env_path(name: str) -> Path:
    return Path(require_env(name)).resolve()


def value_to_define(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)

    if text == "":
        return '""'

    # Numeric strings or symbolic C/C++ values, e.g. AP_RND, AP_SAT.
    if text.replace(".", "", 1).replace("-", "", 1).isdigit():
        return text

    if text.replace("_", "").isalnum() and not text[0].isdigit():
        return text

    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def make_defines_arg(defines: dict[str, Any]) -> str:
    parts: list[str] = []

    for key, value in defines.items():
        parts.append(f"-D{key}={value_to_define(value)}")

    return " ".join(parts)


def count_csv_vectors(path: Path) -> int:
    if not path.exists():
        return 0

    lines = [line for line in path.read_text().splitlines() if line.strip()]

    if not lines:
        return 0

    # Header line is not a vector.
    return max(0, len(lines) - 1)


def build_cases_from_json(
    module_name: str,
    module_dir: Path,
    prefix: str,
    regression_json: Path,
) -> tuple[Path, list[SimCase]]:
    data = load_json(regression_json, "simulation regression JSON")

    # Default layout:
    #
    #   modules/<module>/build/<CASE>/
    #     binary/
    #     simulation/vectors/stimuli/
    #     simulation/vectors/expected/
    #     reports/
    regression_dir = resolve_path(
        data.get("regression_dir", "build"),
        module_dir,
    )
    regression_dir = resolve_path(
        data.get("regression_dir", "build"),
        module_dir,
    )

    defaults = normalize_dict(data.get("defaults", {}), "defaults")

    default_params = extract_params(defaults, META_KEYS)

    default_cpp_defines: dict[str, Any] = {}
    default_cpp_defines.update(normalize_dict(defaults.get("defines", {}), "defaults.defines"))
    default_cpp_defines.update(normalize_dict(defaults.get("cpp_defines", {}), "defaults.cpp_defines"))
    default_cpp_defines.update(normalize_dict(defaults.get("sim_defines", {}), "defaults.sim_defines"))

    default_auto_defines = bool(defaults.get("auto_defines", True))

    raw_cases = data.get("cases")
    if raw_cases is None:
        raise ValueError("[ERROR] simulation regression JSON missing required field: cases")
    if not isinstance(raw_cases, list):
        raise ValueError("[ERROR] field 'cases' must be a list")

    cases: list[SimCase] = []
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
            raise ValueError(f"[ERROR] duplicated regression case name: {case_name}")
        seen.add(case_name)

        params = dict(default_params)
        params.update(extract_params(raw_case, META_KEYS))

        auto_defines = bool(raw_case.get("auto_defines", default_auto_defines))

        cpp_defines = dict(default_cpp_defines)
        cpp_defines.update(normalize_dict(raw_case.get("defines", {}), f"{case_name}.defines"))
        cpp_defines.update(normalize_dict(raw_case.get("cpp_defines", {}), f"{case_name}.cpp_defines"))
        cpp_defines.update(normalize_dict(raw_case.get("sim_defines", {}), f"{case_name}.sim_defines"))

        if auto_defines:
            for key, value in params.items():
                key = str(key)

                if key in RUNTIME_PARAM_KEYS:
                    continue

                if is_define_param_key(key):
                    cpp_defines.setdefault(f"{prefix}_{key}", value)

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

        binary_file = resolve_path(
            raw_case.get(
                "binary_file",
                defaults.get("binary_file", f"binary/{module_name}_tb"),
            ),
            case_dir,
        )

        in_ports_csv = case_dir / "simulation" / "vectors" / "stimuli" / "in_ports.csv"
        out_ports_csv = case_dir / "simulation" / "vectors" / "expected" / "out_ports.csv"

        cases.append(
            SimCase(
                name=case_name,
                case_dir=case_dir,
                params=params,
                cpp_defines=cpp_defines,
                n_cycles=n_cycles,
                binary_file=binary_file,
                in_ports_csv=in_ports_csv,
                out_ports_csv=out_ports_csv,
            )
        )

    if not cases:
        raise RuntimeError("[ERROR] no enabled simulation regression cases found")

    return regression_dir, cases


def compile_sim_case(
    case: SimCase,
    module_name: str,
    module_dir: Path,
    prefix: str,
    makefile: Path,
    verbose: bool = False,
) -> None:
    tb_cpp = env_path(f"{prefix}_TB_CPP")
    tb_hpp = env_path(f"{prefix}_TB_HPP")
    model_cpp = env_path(f"{prefix}_CPP_CPP")
    model_hpp = env_path(f"{prefix}_CPP_HPP")

    if not makefile.exists():
        raise FileNotFoundError(f"[ERROR] Makefile not found: {makefile}")

    for label, path in {
        f"{prefix}_TB_CPP": tb_cpp,
        f"{prefix}_TB_HPP": tb_hpp,
        f"{prefix}_CPP_CPP": model_cpp,
        f"{prefix}_CPP_HPP": model_hpp,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] {label} not found: {path}")

    rtlsim_root = Path(
        os.environ.get("RTLSIM_ROOT", Path(require_env("PROJECT_ROOT")) / "rtlsim")
    ).resolve()

    cmd = [
        "make",
        "-f",
        str(makefile),
        f"RTLSIM_ROOT={rtlsim_root}",
        f"SRC={tb_cpp}",
        f"EXTRA_SRCS={model_cpp}",
        f"EXTRA_HEADERS={tb_hpp} {model_hpp}",
        f"TARGET={case.binary_file}",
        f"DEFINES={make_defines_arg(case.cpp_defines)}",
    ]

    run_command(
        cmd=cmd,
        cwd=module_dir,
        label="compile C++ simulation model",
        verbose=verbose,
    )

    if not case.binary_file.exists():
        raise FileNotFoundError(f"[ERROR] binary was not created: {case.binary_file}")


def run_sim_case(
    case: SimCase,
    module_dir: Path,
    verbose: bool = False,
) -> None:
    if not case.binary_file.exists():
        raise FileNotFoundError(f"[ERROR] binary not found: {case.binary_file}")

    cmd = [
        str(case.binary_file),
        "--case-dir",
        str(case.case_dir),
        "--n-cycles",
        str(case.n_cycles),
    ]

    run_command(
        cmd=cmd,
        cwd=module_dir,
        label="run C++ simulation model",
        verbose=verbose,
    )


def validate_case_outputs(case: SimCase) -> tuple[str, str, int]:
    missing: list[Path] = []

    if not case.in_ports_csv.exists():
        missing.append(case.in_ports_csv)

    if not case.out_ports_csv.exists():
        missing.append(case.out_ports_csv)

    if missing:
        msg = "missing simulation output files: " + ", ".join(str(p) for p in missing)
        return "FAIL", msg, 0

    in_vectors = count_csv_vectors(case.in_ports_csv)
    out_vectors = count_csv_vectors(case.out_ports_csv)

    if in_vectors == 0:
        return "FAIL", "in_ports.csv has no vectors", 0

    if out_vectors == 0:
        return "FAIL", "out_ports.csv has no vectors", 0

    return "PASS", "", out_vectors


def write_sim_summary_report(
    case: SimCase,
    status: str,
    message: str,
    vectors: int,
    elapsed_s: float,
) -> Path:
    rpt_file = case.case_dir / "reports" / "sim_summary.rpt"
    rpt_file.parent.mkdir(parents=True, exist_ok=True)

    with rpt_file.open("w") as f:
        f.write("SIMULATION SUMMARY\n")
        f.write("==================\n\n")

        f.write(f"case        : {case.name}\n")
        f.write(f"status      : {status}\n")
        f.write(f"message     : {message}\n")
        f.write(f"vectors     : {vectors}\n")
        f.write(f"elapsed_s   : {elapsed_s:.6f}\n")
        f.write(f"elapsed     : {format_seconds(elapsed_s)}\n")
        f.write("\n")

        f.write("PATHS\n")
        f.write("-----\n")
        f.write(f"case_dir    : {case.case_dir}\n")
        f.write(f"binary      : {case.binary_file}\n")
        f.write(f"in_ports    : {case.in_ports_csv}\n")
        f.write(f"out_ports   : {case.out_ports_csv}\n")
        f.write("\n")

        f.write("PARAMETERS\n")
        f.write("----------\n")
        if case.params:
            for key, value in case.params.items():
                f.write(f"{key:<24}: {value}\n")
        else:
            f.write("No parameters.\n")
        f.write("\n")

        f.write("CPP DEFINES\n")
        f.write("-----------\n")
        if case.cpp_defines:
            for key, value in case.cpp_defines.items():
                f.write(f"{key:<24}: {value}\n")
        else:
            f.write("No C++ defines.\n")

    return rpt_file


def write_case_manifest(
    case: SimCase,
    module_name: str,
    module_dir: Path,
    regression_json: Path,
    regression_dir: Path,
    makefile: Path,
    status: str,
    message: str,
    vectors: int,
    elapsed_s: float,
    summary_report: Path | None,
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
            "type": "simulation",
            "engine": "cpp_rtlsim",
        },
        "case": {
            "name": case.name,
            "dir": case.case_dir,
            "params": case.params,
            "cpp_defines": case.cpp_defines,
            "n_cycles": case.n_cycles,
        },
        "scripts": {
            "makefile": makefile,
        },
        "artifacts": {
            "manifest_file": manifest_file,
            "summary_report": summary_report,
            "binary_file": case.binary_file,
            "in_ports_csv": case.in_ports_csv,
            "out_ports_csv": case.out_ports_csv,
        },
        "result": {
            "status": status,
            "vectors": vectors,
            "elapsed_s": elapsed_s,
            "elapsed": format_seconds(elapsed_s),
            "message": message,
        },
    }

    write_json(manifest_file, data)
    return manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_regression_sim",
        description="Run C++ rtlsim regressions for the current module.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--case", action="append", default=[], metavar="CASE_NAME")
    parser.add_argument("--list-cases", action="store_true")

    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def print_case_info(case: SimCase) -> None:
    section(f"CASE {case.name}")

    kv("case_dir", case.case_dir)
    kv("binary", case.binary_file)
    kv("in_ports_csv", case.in_ports_csv)
    kv("out_ports_csv", case.out_ports_csv)
    kv("n_cycles", case.n_cycles)

    if case.params:
        subsection("parameters")
        for key, value in case.params.items():
            kv(str(key), value)

    if case.cpp_defines:
        subsection("C++ defines")
        for key, value in case.cpp_defines.items():
            kv(str(key), value)


def print_results_summary(
    results: list[SimResult],
    total_elapsed_s: float,
    selected_cases: int,
) -> None:
    banner("C++ SIMULATION REGRESSION SUMMARY")

    cases_run = len(results)
    cases_not_run = selected_cases - cases_run

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    total_vectors = sum(r.vectors for r in results)

    kv("cases_selected", selected_cases)
    kv("cases_run", cases_run)
    kv("cases_not_run", cases_not_run)
    kv("cases_passed", passed)
    kv("cases_failed", failed)
    kv("total_vectors", total_vectors)
    kv("elapsed", format_seconds(total_elapsed_s))

    print("")
    print(color("  Per-case results", C.BOLD))
    print(color("  " + "-" * 96, C.GRAY))
    print(color(f"  {'status':<8} {'case':<48} {'vectors':>10} {'time':>10}", C.GRAY))
    print(color("  " + "-" * 96, C.GRAY))

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
            f"{format_seconds(result.elapsed_s):>10}"
        )

        if result.message:
            print(color(f"           {result.message}", C.GRAY))

    print(color("  " + "-" * 96, C.GRAY))

    if failed == 0:
        pass_msg("C++ simulation regression completed successfully")
    else:
        fail_msg("C++ simulation regression failed")


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    module_name, module_dir, prefix = detect_current_module()

    project_root = Path(require_env("PROJECT_ROOT")).resolve()
    makefile = Path(require_env("MAKEFILE")).resolve()

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
        known = {case.name for case in cases}
        unknown = selected - known

        if unknown:
            raise RuntimeError(f"[ERROR] unknown regression case(s): {', '.join(sorted(unknown))}")

        cases = [case for case in cases if case.name in selected]

    if args.list_cases:
        banner("AVAILABLE C++ SIMULATION REGRESSION CASES")
        kv("module", module_name)
        kv("regression_json", regression_json)
        kv("regression_dir", regression_dir)
        print("")
        for case in cases:
            print(f"  - {case.name}")
        return 0

    if not cases:
        raise RuntimeError("[ERROR] no simulation regression cases selected")

    banner("C++ SIMULATION REGRESSION")

    kv("module", module_name)
    kv("module_dir", module_dir)
    kv("project_root", project_root)
    kv("makefile", makefile)
    kv("regression_json", regression_json)
    kv("regression_dir", regression_dir)
    kv("selected_cases", len(cases))

    if args.skip_compile:
        warn_msg("C++ compilation disabled (--skip-compile)")

    if args.skip_run:
        warn_msg("C++ execution disabled (--skip-run)")

    t_regression_start = time.perf_counter()
    results: list[SimResult] = []

    for case in cases:
        print_case_info(case)

        case_t0 = time.perf_counter()
        status = "FAIL"
        message = ""
        vectors = 0
        summary_report: Path | None = None
        manifest_file: Path | None = None

        try:
            if not args.skip_compile:
                compile_sim_case(
                    case=case,
                    module_name=module_name,
                    module_dir=module_dir,
                    prefix=prefix,
                    makefile=makefile,
                    verbose=args.verbose,
                )
            else:
                if not case.binary_file.exists():
                    raise FileNotFoundError(
                        f"[ERROR] --skip-compile was used but binary does not exist: "
                        f"{case.binary_file}"
                    )

            if not args.skip_run:
                run_sim_case(
                    case=case,
                    module_dir=module_dir,
                    verbose=args.verbose,
                )

                status, message, vectors = validate_case_outputs(case)
            else:
                status = "PASS"
                message = "run skipped"
                vectors = 0

            case_elapsed = time.perf_counter() - case_t0

            summary_report = write_sim_summary_report(
                case=case,
                status=status,
                message=message,
                vectors=vectors,
                elapsed_s=case_elapsed,
            )

            manifest_file = write_case_manifest(
                case=case,
                module_name=module_name,
                module_dir=module_dir,
                regression_json=regression_json,
                regression_dir=regression_dir,
                makefile=makefile,
                status=status,
                message=message,
                vectors=vectors,
                elapsed_s=case_elapsed,
                summary_report=summary_report,
            )

            step_msg(f"simulation summary: {summary_report}")
            step_msg(f"manifest: {manifest_file}")

            if status == "PASS":
                pass_msg(case.name)
            else:
                fail_msg(f"{case.name}: {message}")

            results.append(
                SimResult(
                    name=case.name,
                    status=status,
                    vectors=vectors,
                    elapsed_s=case_elapsed,
                    message=message,
                    manifest_file=manifest_file,
                    summary_report=summary_report,
                    binary_file=case.binary_file,
                    in_ports_csv=case.in_ports_csv,
                    out_ports_csv=case.out_ports_csv,
                )
            )

            if status == "FAIL" and not args.keep_going:
                break

        except Exception as exc:
            case_elapsed = time.perf_counter() - case_t0
            message = str(exc)

            fail_msg(f"{case.name}: {message}")

            try:
                summary_report = write_sim_summary_report(
                    case=case,
                    status="FAIL",
                    message=message,
                    vectors=0,
                    elapsed_s=case_elapsed,
                )

                manifest_file = write_case_manifest(
                    case=case,
                    module_name=module_name,
                    module_dir=module_dir,
                    regression_json=regression_json,
                    regression_dir=regression_dir,
                    makefile=makefile,
                    status="FAIL",
                    message=message,
                    vectors=0,
                    elapsed_s=case_elapsed,
                    summary_report=summary_report,
                )

                step_msg(f"simulation summary: {summary_report}")
                step_msg(f"manifest: {manifest_file}")

            except Exception as report_exc:
                warn_msg(f"could not write simulation reports for {case.name}: {report_exc}")

            results.append(
                SimResult(
                    name=case.name,
                    status="FAIL",
                    vectors=0,
                    elapsed_s=case_elapsed,
                    message=message,
                    manifest_file=manifest_file,
                    summary_report=summary_report,
                    binary_file=case.binary_file,
                    in_ports_csv=case.in_ports_csv,
                    out_ports_csv=case.out_ports_csv,
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
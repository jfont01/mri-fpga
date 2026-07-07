#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from regression_common import (
    C,
    as_float,
    as_int,
    banner,
    cli_key,
    color,
    detect_current_module,
    error_line,
    extract_params,
    fail_msg,
    format_seconds,
    is_define_param_key,
    json_safe,
    kv,
    load_json,
    module_prefix,
    normalize_dict,
    now_iso,
    parse_key_value_report,
    parse_utilization_report,
    pass_msg,
    require_env,
    resolve_path,
    resolve_tcl_from_env,
    run_command,
    section,
    split_flow_params_from_params,
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
    "rtl_defines",
    "synth_defines",
    "defines",
    "auto_defines",
    "limits",
    "constraints",
}

FLOW_PARAM_KEYS = {
    "PART_NAME",
    "CLOCK_PERIOD_NS",
    "CLOCK_FREQ_MHZ",
    "CLOCK_UNCERTAINTY_NS",
}

LIMIT_KEYS = {
    "MIN_WNS_NS",
    "MAX_LUT",
    "MAX_FF",
    "MAX_DSP",
    "MAX_BRAM",
}































@dataclass(frozen=True)
class SynthCase:
    name: str
    case_dir: Path
    params: dict[str, Any]
    flow_params: dict[str, Any]
    rtl_defines: dict[str, Any]
    limits: dict[str, Any]


@dataclass(frozen=True)
class SynthResult:
    name: str
    status: str
    wns_ns: float | None
    whs_ns: float | None
    est_fmax_mhz: float | None
    lut: int | None
    ff: int | None
    dsp: int | None
    bram: float | None
    elapsed_s: float
    message: str = ""
    manifest_file: Path | None = None
    summary_report: Path | None = None




















def build_cases_from_json(
    module_name: str,
    module_dir: Path,
    prefix: str,
    regression_json: Path,
) -> tuple[Path, list[SynthCase]]:
    data = load_json(regression_json, "synthesis regression JSON")

    regression_dir = resolve_path(
        data.get("regression_dir", "build"),
        module_dir,
    )

    defaults = normalize_dict(data.get("defaults", {}), "defaults")

    default_params = extract_params(defaults, META_KEYS)
    default_flow_params = normalize_dict(defaults.get("constraints", {}), "defaults.constraints")
    default_flow_params.update(split_flow_params_from_params(default_params, FLOW_PARAM_KEYS))

    default_limits = normalize_dict(defaults.get("limits", {}), "defaults.limits")

    default_rtl_defines: dict[str, Any] = {}
    default_rtl_defines.update(normalize_dict(defaults.get("defines", {}), "defaults.defines"))
    default_rtl_defines.update(normalize_dict(defaults.get("rtl_defines", {}), "defaults.rtl_defines"))
    default_rtl_defines.update(normalize_dict(defaults.get("synth_defines", {}), "defaults.synth_defines"))

    default_auto_defines = bool(defaults.get("auto_defines", True))

    raw_cases = data.get("cases")
    if raw_cases is None:
        raise ValueError("[ERROR] synthesis regression JSON missing required field: cases")
    if not isinstance(raw_cases, list):
        raise ValueError("[ERROR] field 'cases' must be a list")

    cases: list[SynthCase] = []
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
            raise ValueError(f"[ERROR] duplicated synthesis regression case name: {case_name}")
        seen.add(case_name)

        params = dict(default_params)
        params.update(extract_params(raw_case, META_KEYS))

        flow_params = dict(default_flow_params)
        flow_params.update(normalize_dict(raw_case.get("constraints", {}), f"{case_name}.constraints"))
        flow_params.update(split_flow_params_from_params(params, FLOW_PARAM_KEYS))

        limits = dict(default_limits)
        limits.update(normalize_dict(raw_case.get("limits", {}), f"{case_name}.limits"))

        for key in LIMIT_KEYS:
            if key in params:
                limits[key] = params.pop(key)

        rtl_defines = dict(default_rtl_defines)

        rtl_defines.update(normalize_dict(raw_case.get("defines", {}), f"{case_name}.defines"))
        rtl_defines.update(normalize_dict(raw_case.get("rtl_defines", {}), f"{case_name}.rtl_defines"))
        rtl_defines.update(normalize_dict(raw_case.get("synth_defines", {}), f"{case_name}.synth_defines"))

        auto_defines = bool(raw_case.get("auto_defines", default_auto_defines))

        if auto_defines:
            for key, value in params.items():
                key = str(key)
                if is_define_param_key(key):
                    rtl_defines.setdefault(f"{prefix}_{key}", value)

        case_dir = regression_dir / case_name / "synthesis"

        cases.append(
            SynthCase(
                name=case_name,
                case_dir=case_dir,
                params=params,
                flow_params=flow_params,
                rtl_defines=rtl_defines,
                limits=limits,
            )
        )

    if not cases:
        raise RuntimeError("[ERROR] no enabled synthesis regression cases found")

    return regression_dir, cases






def run_synth_case(
    case: SynthCase,
    module_dir: Path,
    project_root: Path,
    verbose: bool = False,
) -> None:
    project_root = Path(project_root).resolve()
    module_dir = Path(module_dir).resolve()

    run_synth_tcl = resolve_tcl_from_env(
        "RUN_SYNTH_TCL",
        project_root,
        ["scripts/tcl/run_synth.tcl", "scripts/run_synth.tcl"],
    )

    if not run_synth_tcl.exists():
        raise FileNotFoundError(f"[ERROR] run_synth.tcl not found: {run_synth_tcl}")

    case.case_dir.mkdir(parents=True, exist_ok=True)

    vivado_log_dir = case.case_dir / "vivado"
    vivado_log_dir.mkdir(parents=True, exist_ok=True)

    vivado_journal = vivado_log_dir / "vivado.jou"
    vivado_log = vivado_log_dir / "vivado.log"

    flow_args = [
        f"{key}={value}"
        for key, value in case.flow_params.items()
    ]

    define_args = [
        f"{key}={value}"
        for key, value in case.rtl_defines.items()
    ]

    cmd = [
        "vivado",
        "-mode",
        "batch",
        "-notrace",
        "-journal",
        str(vivado_journal),
        "-log",
        str(vivado_log),
        "-source",
        str(run_synth_tcl),
        "-tclargs",
        str(case.case_dir),
        str(module_dir),
        *flow_args,
        *define_args,
    ]

    run_command(
        cmd=cmd,
        cwd=module_dir,
        label="run synth",
        verbose=verbose,
    )







def collect_synth_metrics(case: SynthCase) -> dict[str, Any]:
    reports_dir = case.case_dir / "reports"

    metrics_rpt = reports_dir / "synth_metrics.rpt"
    utilization_rpt = reports_dir / "utilization_synth.rpt"
    timing_summary_rpt = reports_dir / "timing_summary_synth.rpt"

    metrics: dict[str, Any] = {}

    metrics.update(parse_key_value_report(metrics_rpt))
    metrics.update(parse_utilization_report(utilization_rpt))

    metrics["metrics_report"] = metrics_rpt
    metrics["utilization_report"] = utilization_rpt
    metrics["timing_summary_report"] = timing_summary_rpt
    metrics["checkpoint"] = case.case_dir / "checkpoints" / "synth.dcp"

    return metrics






def evaluate_synth_result(metrics: dict[str, Any], limits: dict[str, Any]) -> tuple[str, str]:
    messages: list[str] = []

    min_wns = as_float(limits.get("MIN_WNS_NS", 0.0))
    wns = as_float(metrics.get("WNS_NS"))

    if wns is None:
        messages.append("missing WNS")
    elif min_wns is not None and wns < min_wns:
        messages.append(f"WNS {wns:.3f} ns < required {min_wns:.3f} ns")

    max_lut = as_int(limits.get("MAX_LUT"))
    max_ff = as_int(limits.get("MAX_FF"))
    max_dsp = as_int(limits.get("MAX_DSP"))
    max_bram = as_float(limits.get("MAX_BRAM"))

    lut = as_int(metrics.get("LUT"))
    ff = as_int(metrics.get("FF"))
    dsp = as_int(metrics.get("DSP"))
    bram = as_float(metrics.get("BRAM"))

    if max_lut is not None and lut is not None and lut > max_lut:
        messages.append(f"LUT {lut} > MAX_LUT {max_lut}")

    if max_ff is not None and ff is not None and ff > max_ff:
        messages.append(f"FF {ff} > MAX_FF {max_ff}")

    if max_dsp is not None and dsp is not None and dsp > max_dsp:
        messages.append(f"DSP {dsp} > MAX_DSP {max_dsp}")

    if max_bram is not None and bram is not None and bram > max_bram:
        messages.append(f"BRAM {bram} > MAX_BRAM {max_bram}")

    if messages:
        return "FAIL", "; ".join(messages)

    return "PASS", ""


def write_synth_summary_report(
    case: SynthCase,
    metrics: dict[str, Any],
    status: str,
    message: str,
    elapsed_s: float,
) -> Path:
    report_file = case.case_dir / "reports" / "synth_summary.rpt"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with report_file.open("w") as f:
        f.write("SYNTHESIS REGRESSION REPORT\n")
        f.write("===========================\n\n")

        f.write(f"generated_at      : {now_iso()}\n")
        f.write(f"case              : {case.name}\n")
        f.write(f"case_dir          : {case.case_dir}\n")
        f.write(f"status            : {status}\n")
        f.write(f"message           : {message}\n")
        f.write(f"elapsed           : {format_seconds(elapsed_s)}\n\n")

        f.write("CONSTRAINTS / FLOW PARAMETERS\n")
        f.write("-----------------------------\n")
        for key, value in case.flow_params.items():
            f.write(f"{key:<24}: {value}\n")

        f.write("\nRTL DEFINES\n")
        f.write("-----------\n")
        for key, value in case.rtl_defines.items():
            f.write(f"{key:<24}: {value}\n")

        f.write("\nLIMITS\n")
        f.write("------\n")
        if case.limits:
            for key, value in case.limits.items():
                f.write(f"{key:<24}: {value}\n")
        else:
            f.write("No explicit limits.\n")

        f.write("\nTIMING\n")
        f.write("------\n")
        f.write(f"{'CLOCK_PERIOD_NS':<24}: {metrics.get('CLOCK_PERIOD_NS')}\n")
        f.write(f"{'CLOCK_FREQ_MHZ':<24}: {metrics.get('CLOCK_FREQ_MHZ')}\n")
        f.write(f"{'CLOCK_UNCERTAINTY_NS':<24}: {metrics.get('CLOCK_UNCERTAINTY_NS')}\n")
        f.write(f"{'WNS_NS':<24}: {metrics.get('WNS_NS')}\n")
        f.write(f"{'WHS_NS':<24}: {metrics.get('WHS_NS')}\n")
        f.write(f"{'EST_FMAX_MHZ':<24}: {metrics.get('EST_FMAX_MHZ')}\n")

        f.write("\nUTILIZATION\n")
        f.write("-----------\n")
        f.write(f"{'LUT':<24}: {metrics.get('LUT')}\n")
        f.write(f"{'FF':<24}: {metrics.get('FF')}\n")
        f.write(f"{'DSP':<24}: {metrics.get('DSP')}\n")
        f.write(f"{'BRAM':<24}: {metrics.get('BRAM')}\n")

        f.write("\nARTIFACTS\n")
        f.write("---------\n")
        f.write(f"{'utilization_report':<24}: {metrics.get('utilization_report')}\n")
        f.write(f"{'timing_summary_report':<24}: {metrics.get('timing_summary_report')}\n")
        f.write(f"{'metrics_report':<24}: {metrics.get('metrics_report')}\n")
        f.write(f"{'checkpoint':<24}: {metrics.get('checkpoint')}\n")

    return report_file


def write_case_manifest(
    case: SynthCase,
    module_name: str,
    module_dir: Path,
    regression_json: Path,
    regression_dir: Path,
    status: str,
    message: str,
    elapsed_s: float,
    metrics: dict[str, Any],
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
            "type": "synthesis",
        },
        "case": {
            "name": case.name,
            "dir": case.case_dir,
            "params": case.params,
            "constraints": case.flow_params,
            "rtl_defines": case.rtl_defines,
            "limits": case.limits,
        },
        "artifacts": {
            "manifest_file": manifest_file,
            "summary_report": summary_report,
            "utilization_report": metrics.get("utilization_report"),
            "timing_summary_report": metrics.get("timing_summary_report"),
            "metrics_report": metrics.get("metrics_report"),
            "checkpoint": metrics.get("checkpoint"),
            "vivado_log": case.case_dir / "vivado" / "vivado.log",
            "vivado_journal": case.case_dir / "vivado" / "vivado.jou",
        },
        "metrics": metrics,
        "result": {
            "status": status,
            "message": message,
            "elapsed_s": elapsed_s,
            "elapsed": format_seconds(elapsed_s),
        },
    }

    write_json(manifest_file, data)
    return manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_regression_synth",
        description="Run synthesis regressions for the current module.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--case", action="append", default=[], metavar="CASE_NAME")
    parser.add_argument("--list-cases", action="store_true")

    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--keep-going", action="store_true")

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def print_case_info(case: SynthCase) -> None:
    section(f"SYNTH CASE {case.name}")

    kv("case_dir", case.case_dir)

    if case.flow_params:
        subsection("constraints / flow parameters")
        for key, value in case.flow_params.items():
            kv(str(key), value)

    if case.rtl_defines:
        subsection("rtl defines")
        for key, value in case.rtl_defines.items():
            kv(str(key), value)

    if case.limits:
        subsection("limits")
        for key, value in case.limits.items():
            kv(str(key), value)


def print_results_summary(
    results: list[SynthResult],
    total_elapsed_s: float,
    selected_cases: int,
) -> None:
    banner("SYNTHESIS REGRESSION SUMMARY")

    cases_run = len(results)
    cases_not_run = selected_cases - cases_run
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")

    kv("cases_selected", selected_cases)
    kv("cases_run", cases_run)
    kv("cases_not_run", cases_not_run)
    kv("cases_passed", passed)
    kv("cases_failed", failed)
    kv("elapsed", format_seconds(total_elapsed_s))

    print("")
    print(color("  Per-case synthesis results", C.BOLD))
    print(color("  " + "-" * 124, C.GRAY))

    header = (
        f"  {'status':<8} "
        f"{'case':<44} "
        f"{'WNS(ns)':>10} "
        f"{'Fmax(MHz)':>10} "
        f"{'LUT':>8} "
        f"{'FF':>8} "
        f"{'DSP':>6} "
        f"{'BRAM':>8} "
        f"{'time':>10}"
    )

    print(color(header, C.GRAY))
    print(color("  " + "-" * 124, C.GRAY))

    for result in results:
        status = (
            color(f"{'PASS':<8}", C.BOLD + C.GREEN)
            if result.status == "PASS"
            else color(f"{'FAIL':<8}", C.BOLD + C.RED)
        )

        def fmt_float(value: float | None, digits: int = 3) -> str:
            return "NA" if value is None else f"{value:.{digits}f}"

        def fmt_any(value: Any) -> str:
            return "NA" if value is None else str(value)

        print(
            f"  {status} "
            f"{result.name:<44} "
            f"{fmt_float(result.wns_ns):>10} "
            f"{fmt_float(result.est_fmax_mhz):>10} "
            f"{fmt_any(result.lut):>8} "
            f"{fmt_any(result.ff):>8} "
            f"{fmt_any(result.dsp):>6} "
            f"{fmt_any(result.bram):>8} "
            f"{format_seconds(result.elapsed_s):>10}"
        )

        if result.message:
            print(color(f"           {result.message}", C.GRAY))

    print(color("  " + "-" * 124, C.GRAY))

    if failed == 0:
        pass_msg("synthesis regression completed successfully")
    else:
        fail_msg("synthesis regression failed")


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    project_root = Path(require_env("PROJECT_ROOT")).resolve()
    module_name, module_dir, prefix = detect_current_module()

    regression_json_env = os.environ.get(f"{prefix}_SYNTH_REGRESSION_JSON")
    if regression_json_env:
        regression_json = resolve_path(regression_json_env, module_dir)
    else:
        regression_json = module_dir / "synthesis" / f"{module_name}_synth_regression.json"

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
            raise ValueError(f"[ERROR] unknown synthesis regression case(s): {sorted(missing)}")

        cases = [case for case in cases if case.name in selected]

    if args.list_cases:
        banner("AVAILABLE SYNTHESIS REGRESSION CASES")
        kv("module", module_name)
        kv("regression_json", regression_json)
        kv("regression_dir", regression_dir)
        print("")
        for case in cases:
            print(f"  - {case.name}")
        return 0

    if not cases:
        raise RuntimeError("[ERROR] no synthesis regression cases selected")

    banner("SYNTHESIS REGRESSION")

    kv("module", module_name)
    kv("module_dir", module_dir)
    kv("regression_json", regression_json)
    kv("regression_dir", regression_dir)
    kv("selected_cases", len(cases))

    if args.skip_synth:
        warn_msg("Vivado synthesis disabled (--skip-synth)")

    t_regression_start = time.perf_counter()
    results: list[SynthResult] = []

    for case in cases:
        print_case_info(case)

        case_t0 = time.perf_counter()
        metrics: dict[str, Any] = {}
        status = "FAIL"
        message = ""
        summary_report: Path | None = None
        manifest_file: Path | None = None

        try:
            if not args.skip_synth:
                run_synth_case(
                    case=case,
                    module_dir=module_dir,
                    project_root=project_root,
                    verbose=args.verbose,
                )

            metrics = collect_synth_metrics(case)
            status, message = evaluate_synth_result(metrics, case.limits)

            case_elapsed = time.perf_counter() - case_t0

            summary_report = write_synth_summary_report(
                case=case,
                metrics=metrics,
                status=status,
                message=message,
                elapsed_s=case_elapsed,
            )

            manifest_file = write_case_manifest(
                case=case,
                module_name=module_name,
                module_dir=module_dir,
                regression_json=regression_json,
                regression_dir=regression_dir,
                status=status,
                message=message,
                elapsed_s=case_elapsed,
                metrics=metrics,
                summary_report=summary_report,
            )

            step_msg(f"synth summary: {summary_report}")
            step_msg(f"manifest: {manifest_file}")

            if status == "PASS":
                pass_msg(case.name)
            else:
                fail_msg(f"{case.name}: {message}")

            result = SynthResult(
                name=case.name,
                status=status,
                wns_ns=as_float(metrics.get("WNS_NS")),
                whs_ns=as_float(metrics.get("WHS_NS")),
                est_fmax_mhz=as_float(metrics.get("EST_FMAX_MHZ")),
                lut=as_int(metrics.get("LUT")),
                ff=as_int(metrics.get("FF")),
                dsp=as_int(metrics.get("DSP")),
                bram=as_float(metrics.get("BRAM")),
                elapsed_s=case_elapsed,
                message=message,
                manifest_file=manifest_file,
                summary_report=summary_report,
            )

            results.append(result)

            if status == "FAIL" and not args.keep_going:
                break

        except Exception as exc:
            case_elapsed = time.perf_counter() - case_t0
            message = str(exc)

            fail_msg(f"{case.name}: {message}")

            try:
                metrics = collect_synth_metrics(case)

                summary_report = write_synth_summary_report(
                    case=case,
                    metrics=metrics,
                    status="FAIL",
                    message=message,
                    elapsed_s=case_elapsed,
                )

                manifest_file = write_case_manifest(
                    case=case,
                    module_name=module_name,
                    module_dir=module_dir,
                    regression_json=regression_json,
                    regression_dir=regression_dir,
                    status="FAIL",
                    message=message,
                    elapsed_s=case_elapsed,
                    metrics=metrics,
                    summary_report=summary_report,
                )

                step_msg(f"synth summary: {summary_report}")
                step_msg(f"manifest: {manifest_file}")

            except Exception as report_exc:
                warn_msg(f"could not write synthesis reports for {case.name}: {report_exc}")

            results.append(
                SynthResult(
                    name=case.name,
                    status="FAIL",
                    wns_ns=as_float(metrics.get("WNS_NS")),
                    whs_ns=as_float(metrics.get("WHS_NS")),
                    est_fmax_mhz=as_float(metrics.get("EST_FMAX_MHZ")),
                    lut=as_int(metrics.get("LUT")),
                    ff=as_int(metrics.get("FF")),
                    dsp=as_int(metrics.get("DSP")),
                    bram=as_float(metrics.get("BRAM")),
                    elapsed_s=case_elapsed,
                    message=message,
                    manifest_file=manifest_file,
                    summary_report=summary_report,
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
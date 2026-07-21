#!/usr/bin/env python3
"""
run_gtest.py

Corre los tests unitarios (Google Test) del MODELO C++ del módulo actual, en
dos modos de compilación:

  fixed  : el modelo con ap_fixed (punto fijo, con AP_RND/AP_SAT)
  double : el mismo modelo compilado con -DDOUBLE, o sea con double en todos
           los Wire/Reg/Port del datapath

Por qué los dos modos
---------------------
Es una bisección de la causa de un fallo:

  double FALLA            -> el bug es ESTRUCTURAL (fórmula, signo, índice).
                             La cuantización no tiene nada que ver.
  double PASA, fixed FALLA -> el bug es NUMÉRICO (formato, redondeo,
                             saturación, crecimiento de bits).

Sin el modo double, un error de signo y un error de redondeo se ven igual:
"los números no dan".

Dónde encaja en el flujo
------------------------
  run_gtest  : modelo C++  ~=  matemática de referencia   (tolerancia acotada)
  run_regression_sim : genera vectores desde el modelo
  run_regression_vm  : RTL  ==  modelo C++                (bit a bit)

El vm compara el RTL contra el modelo. Si el modelo estuviera mal pero fuera
consistente con el RTL, el vm pasaría en verde igual: run_gtest es la única
etapa que valida el modelo en sí.

JSON
----
  testbench/<module>_gtest_regression.json

  {
    "regression_dir": "build",
    "defaults": {
      "modes": ["fixed", "double"],
      "auto_defines": true
    },
    "cases": [
      { "name": "q2_14", "NB_IN": 16, "NBF_IN": 14, "NB_OUT": 16, "NBF_OUT": 14 }
    ]
  }

Cada caso se expande a un caso por modo: q2_14[fixed], q2_14[double].
Con auto_defines, las claves en MAYÚSCULAS se pasan como
-D<PREFIX>_<KEY>=<valor>. El modo double agrega -DDOUBLE (sin prefijo, igual
que en los .hpp).

Variables de entorno esperadas (las define <module>_vars.sh):
  <PREFIX>_GTEST_CPP  : fuente del test (testbench/<module>_gtest.cpp)
  <PREFIX>_CPP_CPP    : implementación del modelo
  <PREFIX>_CPP_HPP    : header del modelo

Google Test se busca en flags estándar (-lgtest -lgtest_main -pthread). Si está
en una ruta no estándar, definir GTEST_INCLUDE_DIR y/o GTEST_LIB_DIR.

Salidas por caso:
  build/<case>[_<mode>]/reports/gtest.rpt   (resumen legible)
  build/<case>[_<mode>]/reports/gtest.xml   (JUnit, para CI)
"""

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regression_common import (
    banner,
    detect_current_module,
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
    section,
    set_use_color,
    step_msg,
    subsection,
    warn_msg,
    write_json,
)


MODES = ("fixed", "double")

# Claves que NO son parámetros del diseño. Incluye las de run_regression_sim y
# run_regression_vm, para poder compartir el MISMO JSON de testbench: las que
# no aplican a gtest simplemente se ignoran.
META_KEYS = {
    # comunes
    "name",
    "enabled",
    "params",
    "defines",
    "auto_defines",
    # específicas de gtest
    "gtest",
    "modes",
    "gtest_defines",
    "gtest_filter",
    # de run_regression_sim (ignoradas acá)
    "cpp_defines",
    "sim_defines",
    "binary_file",
    # de run_regression_vm (ignoradas acá)
    "rtl_defines",
    "xsim_defines",
    "vm_defines",
    "stimuli_csv",
    "expected_csv",
    "actual_csv",
    "expected_dat_dir",
    "actual_dat_dir",
    "expected_file",
    "actual_file",
}

# Parámetros de flujo, no del diseño: no se convierten en -D.
# (mismo criterio que RUNTIME_PARAM_KEYS de run_regression_sim)
RUNTIME_PARAM_KEYS = {
    "N_CYCLES",
    "n_cycles",
}


# ---------------------------------------------------------------------------
# Modelo de caso
# ---------------------------------------------------------------------------


@dataclass
class GtestCase:
    name: str
    mode: str
    case_dir: Path
    binary_file: Path
    params: dict[str, Any] = field(default_factory=dict)
    defines: dict[str, Any] = field(default_factory=dict)
    gtest_filter: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name}[{self.mode}]"


@dataclass
class GtestResult:
    case: GtestCase
    ran: bool = False
    total: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    failed_names: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.ran and self.failures == 0 and self.errors == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def env_path(name: str) -> Path:
    return Path(require_env(name)).resolve()


def resolve_executable(binary_path: Path) -> Path:
    """En MSYS/Windows el binario sale con .exe."""
    candidates = []

    if binary_path.suffix != ".exe":
        candidates.append(binary_path.with_suffix(".exe"))

    candidates.append(binary_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"[ERROR] no se encontró ejecutable para '{binary_path}'."
    )


def make_defines_arg(defines: dict[str, Any]) -> str:
    parts: list[str] = []

    for key, value in defines.items():
        if value is None or value is True:
            parts.append(f"-D{key}")
        elif value is False:
            continue
        else:
            parts.append(f"-D{key}={value}")

    return " ".join(parts)


def gtest_compile_flags() -> tuple[str, str]:
    """
    Devuelve (extra_cppflags, extra_ldflags) para Google Test.
    Se pueden forzar rutas con GTEST_INCLUDE_DIR / GTEST_LIB_DIR.
    """
    cpp_parts: list[str] = []
    ld_parts: list[str] = []

    include_dir = os.environ.get("GTEST_INCLUDE_DIR")
    if include_dir:
        cpp_parts.append(f"-isystem {include_dir}")

    lib_dir = os.environ.get("GTEST_LIB_DIR")
    if lib_dir:
        ld_parts.append(f"-L{lib_dir}")

    ld_parts.extend(["-lgtest", "-lgtest_main", "-pthread"])

    return " ".join(cpp_parts), " ".join(ld_parts)


# ---------------------------------------------------------------------------
# Carga del JSON
# ---------------------------------------------------------------------------


def build_cases(
    data: dict[str, Any],
    module_name: str,
    module_dir: Path,
    prefix: str,
    mode_filter: str | None,
) -> tuple[list[GtestCase], Path]:
    regression_dir = resolve_path(
        data.get("regression_dir", "build"),
        module_dir,
    )

    defaults = normalize_dict(data.get("defaults"), "defaults")
    raw_cases = data.get("cases", [])

    if not isinstance(raw_cases, list):
        raise ValueError("[ERROR] 'cases' debe ser una lista")

    default_gtest = normalize_dict(defaults.get("gtest"), "defaults.gtest")

    default_modes = default_gtest.get("modes", defaults.get("modes", list(MODES)))
    default_filter = default_gtest.get("gtest_filter", defaults.get("gtest_filter"))
    default_auto = bool(defaults.get("auto_defines", True))
    default_params = extract_params(defaults, META_KEYS)

    cases: list[GtestCase] = []

    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"[ERROR] case #{index} debe ser un objeto")

        name = str(raw.get("name", f"case_{index}"))

        if not bool(raw.get("enabled", True)):
            continue

        params = dict(default_params)
        params.update(extract_params(raw, META_KEYS))

        # N_CYCLES y compañía son del flujo de simulación, no del diseño:
        # no deben terminar como -D<PREFIX>_N_CYCLES.
        for key in RUNTIME_PARAM_KEYS:
            params.pop(key, None)

        auto_defines = bool(raw.get("auto_defines", default_auto))

        defines: dict[str, Any] = {}
        if auto_defines:
            for key, value in params.items():
                if is_define_param_key(key):
                    defines.setdefault(f"{prefix}_{key}", value)

        defines.update(normalize_dict(raw.get("defines"), "defines"))
        defines.update(normalize_dict(raw.get("gtest_defines"), "gtest_defines"))

        case_gtest = normalize_dict(raw.get("gtest"), f"case '{name}'.gtest")

        modes = case_gtest.get("modes", raw.get("modes", default_modes))
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            if mode not in MODES:
                raise ValueError(
                    f"[ERROR] modo inválido '{mode}' en case '{name}'. "
                    f"Esperado uno de {MODES}"
                )

            if mode_filter and mode != mode_filter:
                continue

            # MISMO case_dir que run_regression_sim / run_regression_vm: los
            # artefactos de gtest se distinguen por el sufijo de modo, así todo
            # lo del caso queda junto en build/<case>/.
            case_dir = regression_dir / name

            case_defines = dict(defines)
            if mode == "double":
                # sin prefijo: los .hpp usan #ifdef DOUBLE
                case_defines["DOUBLE"] = True

            cases.append(
                GtestCase(
                    name=name,
                    mode=mode,
                    case_dir=case_dir,
                    binary_file=case_dir / "binary" / f"{module_name}_gtest_{mode}",
                    params=dict(params),
                    defines=case_defines,
                    gtest_filter=case_gtest.get(
                        "gtest_filter", raw.get("gtest_filter", default_filter)
                    ),
                )
            )

    return cases, regression_dir


# ---------------------------------------------------------------------------
# Compilar y correr
# ---------------------------------------------------------------------------


def compile_case(
    case: GtestCase,
    module_dir: Path,
    prefix: str,
    makefile: Path,
    verbose: bool,
) -> None:
    gtest_cpp = env_path(f"{prefix}_GTEST_CPP")
    model_cpp = env_path(f"{prefix}_CPP_CPP")
    model_hpp = env_path(f"{prefix}_CPP_HPP")

    if not makefile.exists():
        raise FileNotFoundError(f"[ERROR] Makefile no encontrado: {makefile}")

    for label, path in {
        f"{prefix}_GTEST_CPP": gtest_cpp,
        f"{prefix}_CPP_CPP": model_cpp,
        f"{prefix}_CPP_HPP": model_hpp,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] {label} no encontrado: {path}")

    rtlsim_root = Path(
        os.environ.get("RTLSIM_ROOT", Path(require_env("PROJECT_ROOT")) / "rtlsim")
    ).resolve()

    extra_cppflags, extra_ldflags = gtest_compile_flags()

    cmd = [
        "make",
        "-f", str(makefile),
        f"RTLSIM_ROOT={rtlsim_root}",
        f"SRC={gtest_cpp}",
        f"EXTRA_SRCS={model_cpp}",
        f"EXTRA_HEADERS={model_hpp}",
        f"TARGET={case.binary_file}",
        f"DEFINES={make_defines_arg(case.defines)}",
        f"EXTRA_CPPFLAGS={extra_cppflags}",
        f"EXTRA_LDFLAGS={extra_ldflags}",
    ]

    if verbose:
        step_msg("comando: " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=module_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.rstrip())
        raise RuntimeError(f"[ERROR] falló la compilación de {case.label}")

    if verbose and result.stdout:
        print(result.stdout.rstrip())

    resolve_executable(case.binary_file)


def parse_gtest_xml(xml_path: Path) -> dict[str, Any]:
    """Parsea el XML JUnit que emite gtest con --gtest_output=xml:"""
    info: dict[str, Any] = {
        "total": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
        "failed_names": [],
    }

    if not xml_path.exists():
        return info

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return info

    info["total"] = int(root.get("tests", 0))
    info["failures"] = int(root.get("failures", 0))
    info["errors"] = int(root.get("errors", 0))
    info["time"] = float(root.get("time", 0.0))

    # NOTA: el <testsuites> raiz NO lleva atributo 'skipped' (solo 'disabled').
    # Los GTEST_SKIP() aparecen como testcase con result="skipped", asi que hay
    # que contarlos recorriendo los casos.
    skipped = int(root.get("disabled", 0))

    for testcase in root.iter("testcase"):
        if testcase.find("failure") is not None:
            suite = testcase.get("classname", "")
            name = testcase.get("name", "")
            info["failed_names"].append(f"{suite}.{name}")

        if testcase.get("result") == "skipped" or testcase.find("skipped") is not None:
            skipped += 1

    info["skipped"] = skipped

    return info


def run_case(
    case: GtestCase,
    module_dir: Path,
    verbose: bool,
) -> GtestResult:
    result = GtestResult(case=case)

    reports_dir = case.case_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    xml_path = reports_dir / f"gtest_{case.mode}.xml"
    rpt_path = reports_dir / f"gtest_{case.mode}.rpt"

    if xml_path.exists():
        xml_path.unlink()

    exe = resolve_executable(case.binary_file)

    cmd = [str(exe), f"--gtest_output=xml:{xml_path}"]

    if case.gtest_filter:
        cmd.append(f"--gtest_filter={case.gtest_filter}")

    t0 = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=module_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    result.elapsed_s = time.perf_counter() - t0
    result.ran = True

    info = parse_gtest_xml(xml_path)

    result.total = info["total"]
    result.failures = info["failures"]
    result.errors = info["errors"]
    result.skipped = info["skipped"]
    result.failed_names = info["failed_names"]

    # si el XML no se pudo leer, caer al return code
    if result.total == 0 and proc.returncode != 0:
        result.failures = 1
        result.message = "gtest falló y no se pudo parsear el XML"

    output = proc.stdout or ""

    with rpt_path.open("w") as f:
        f.write("GTEST REPORT\n")
        f.write("============\n\n")
        f.write(f"generated_at : {now_iso()}\n")
        f.write(f"case         : {case.name}\n")
        f.write(f"mode         : {case.mode}\n")
        f.write(f"case_dir     : {case.case_dir}\n")
        f.write(f"binary       : {exe}\n")
        f.write(f"status       : {'PASS' if result.passed else 'FAIL'}\n")
        f.write(f"tests        : {result.total}\n")
        f.write(f"failures     : {result.failures}\n")
        f.write(f"errors       : {result.errors}\n")
        f.write(f"skipped      : {result.skipped}\n")
        f.write(f"elapsed      : {format_seconds(result.elapsed_s)}\n")
        f.write("\n")

        f.write("PARAMETERS\n")
        f.write("----------\n")
        for key, value in case.params.items():
            f.write(f"{key:<24}: {value}\n")
        f.write("\n")

        f.write("DEFINES\n")
        f.write("-------\n")
        for key, value in case.defines.items():
            f.write(f"{key:<24}: {value}\n")
        f.write("\n")

        if result.failed_names:
            f.write("FAILED TESTS\n")
            f.write("------------\n")
            for name in result.failed_names:
                f.write(f"  {name}\n")
            f.write("\n")

        f.write("OUTPUT\n")
        f.write("------\n")
        f.write(output.rstrip())
        f.write("\n")

    if verbose or not result.passed:
        print(output.rstrip())

    step_msg(f"reporte: {rpt_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_gtest",
        description="Corre los tests de Google Test del modelo C++ del módulo actual.",
    )

    parser.add_argument("--json", default=None, metavar="PATH",
                        help="JSON de regresión. Default: testbench/<module>_gtest_regression.json "
                             "si existe, si no testbench/<module>_tb_regression.json (compartido "
                             "con sim y vm).")
    parser.add_argument("--case", action="append", default=None, metavar="NAME",
                        help="Correr solo el/los casos indicados (repetible).")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="Correr solo un modo (fixed o double). Default: los dos.")
    parser.add_argument("--gtest-filter", default=None, metavar="PATTERN",
                        help="Se pasa tal cual a --gtest_filter.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="Seguir aunque un caso falle.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    module_name, module_dir, prefix = detect_current_module()

    if args.json:
        json_path = resolve_path(args.json, module_dir)
    else:
        # Por defecto se comparte el JSON del testbench (el mismo que usan
        # run_regression_sim y run_regression_vm). Si existe uno específico de
        # gtest, tiene prioridad.
        gtest_json = module_dir / "testbench" / f"{module_name}_gtest_regression.json"
        tb_json = module_dir / "testbench" / f"{module_name}_tb_regression.json"
        json_path = gtest_json if gtest_json.exists() else tb_json

    makefile = Path(os.environ.get("MAKEFILE", "")).resolve() if os.environ.get("MAKEFILE") \
        else (Path(require_env("PROJECT_ROOT")) / "scripts" / "Makefile").resolve()

    banner("GTEST REGRESSION")
    kv("module", module_name)
    kv("module_dir", module_dir)
    kv("regression_json", json_path)

    try:
        data = load_json(json_path, "gtest regression JSON")
        cases, regression_dir = build_cases(
            data=data,
            module_name=module_name,
            module_dir=module_dir,
            prefix=prefix,
            mode_filter=args.mode,
        )
    except (FileNotFoundError, ValueError) as exc:
        fail_msg(str(exc))
        return 1

    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]

    if args.gtest_filter:
        for case in cases:
            case.gtest_filter = args.gtest_filter

    kv("regression_dir", regression_dir)
    kv("selected_cases", len(cases))

    if args.list_cases:
        section("cases")
        for case in cases:
            print(f"  {case.label}")
        return 0

    if not cases:
        warn_msg("no hay casos para correr")
        return 1

    results: list[GtestResult] = []
    failed = False

    for case in cases:
        section(f"CASE {case.label}")
        kv("case_dir", case.case_dir)
        kv("mode", case.mode)

        if case.defines:
            subsection("defines")
            for key, value in case.defines.items():
                kv(key, value)

        try:
            step_msg("compilando")
            compile_case(
                case=case,
                module_dir=module_dir,
                prefix=prefix,
                makefile=makefile,
                verbose=args.verbose,
            )

            step_msg("corriendo gtest")
            result = run_case(case=case, module_dir=module_dir, verbose=args.verbose)

        except Exception as exc:
            result = GtestResult(case=case, ran=False, message=str(exc))
            fail_msg(f"{case.label}: {exc}")

        results.append(result)

        if result.passed:
            pass_msg(
                f"{case.label}: {result.total - result.skipped} tests OK"
                + (f", {result.skipped} skipped" if result.skipped else "")
                + f" ({format_seconds(result.elapsed_s)})"
            )
        else:
            failed = True
            fail_msg(f"{case.label}: {result.failures} fallas de {result.total}")

            if not args.keep_going:
                break

    # -----------------------------------------------------------------------
    # Resumen
    # -----------------------------------------------------------------------

    banner("GTEST SUMMARY")

    total_tests = sum(r.total for r in results)
    total_fail = sum(r.failures + r.errors for r in results)
    total_skip = sum(r.skipped for r in results)

    kv("cases_run", len(results))
    kv("cases_passed", sum(1 for r in results if r.passed))
    kv("cases_failed", sum(1 for r in results if not r.passed))
    kv("total_tests", total_tests)
    kv("total_failures", total_fail)
    kv("total_skipped", total_skip)

    print("")
    header = f"  {'status':<8} {'case':<32} {'mode':<8} {'tests':>6} {'fail':>6} {'skip':>6} {'time':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {status:<8} {r.case.name:<32} {r.case.mode:<8} "
            f"{r.total:>6} {r.failures + r.errors:>6} {r.skipped:>6} "
            f"{format_seconds(r.elapsed_s):>9}"
        )
        if r.failed_names:
            for name in r.failed_names:
                print(f"           {name}")
        if r.message:
            print(f"           {r.message}")

    print("")

    manifest = regression_dir / "gtest_manifest.json"  # no pisa el manifest.json del vm
    write_json(
        manifest,
        {
            "generated_at": now_iso(),
            "module": module_name,
            "cases": [
                {
                    "name": r.case.name,
                    "mode": r.case.mode,
                    "status": "PASS" if r.passed else "FAIL",
                    "tests": r.total,
                    "failures": r.failures + r.errors,
                    "skipped": r.skipped,
                    "elapsed_s": round(r.elapsed_s, 3),
                    "failed_tests": r.failed_names,
                    "case_dir": r.case.case_dir,
                }
                for r in results
            ],
        },
    )
    step_msg(f"manifest: {manifest}")

    if failed:
        fail_msg("GTEST regression failed")
        return 1

    pass_msg("GTEST regression passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
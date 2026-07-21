#!/usr/bin/env python3
"""
run_lint.py

Lint estático del RTL puro del módulo actual usando Verilator (--lint-only).
No simula, no necesita vectores ni golden model: analiza el .v directamente.

Verilator corre igual en msys y linux, así que este runner NO ramifica por
SIM_BACKEND: es uniforme en ambas plataformas.

El módulo se detecta desde el directorio actual (igual que los otros runners).
El top-level se asume igual al nombre del módulo.

WAIVERS:
  Los warnings que NO queremos tratar como error se listan en un archivo de
  waivers (uno por línea, admite comentarios con '#'). Cada código se traduce
  a un flag -Wno-<CODE> de Verilator. Ejemplo de archivo:

      # cosmético, no es un bug
      EOFNEWLINE
      # DECLFILENAME   (comentado -> NO se silencia)

  El archivo por defecto se toma de la variable de entorno LINT_WAIVERS, o de
  scripts/lint_waivers.txt. Se puede override con --waivers <path>.

REPORTE:
  Se guarda en <module>/build/reports/lint.rpt
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from regression_common import (
    banner,
    collect_include_dirs,
    detect_current_module,
    fail_msg,
    format_seconds,
    kv,
    now_iso,
    pass_msg,
    section,
    set_use_color,
    step_msg,
    warn_msg,
)


def to_native_path(path: str) -> str:
    """
    Convierte un path MSYS (/c/Users/...) a formato Windows (C:/Users/...)
    con cygpath, para pasárselo al binario nativo de Verilator en Windows.
    En Linux (sin cygpath) devuelve el path sin cambios.
    """
    if shutil.which("cygpath") is None:
        return path
    try:
        result = subprocess.run(
            ["cygpath", "-m", path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return path


def resolve_rtl_top_file(module_name: str, module_dir: Path, prefix: str) -> Path:
    """
    Resuelve el archivo RTL top. Prioriza ${PREFIX}_V / ${PREFIX}_SV (mismas
    variables que flist_utils.sh); si no, cae al layout por convención
    modules/<module>/rtl/<module>.(v|sv).
    """
    for suffix in ("_V", "_SV"):
        var = f"{prefix}{suffix}"
        value = os.environ.get(var)
        if value:
            path = Path(value).resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"[ERROR] {var} apunta a un archivo inexistente: {path}")

    for ext in (".v", ".sv"):
        candidate = (module_dir / "rtl" / f"{module_name}{ext}").resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"[ERROR] no se encontró el RTL top de '{module_name}'. "
        f"Definí {prefix}_V/{prefix}_SV o poné el archivo en "
        f"{module_dir / 'rtl' / (module_name + '.v')}"
    )


def resolve_waivers_file(cli_waivers: str | None) -> Path | None:
    """
    Resuelve el archivo de waivers. Prioridad:
      1. --waivers <path> (CLI)
      2. $LINT_WAIVERS
      3. $PROJECT_ROOT/scripts/lint_waivers.txt
    Devuelve None si no hay ninguno (lint sin waivers).
    """
    if cli_waivers:
        path = Path(cli_waivers).resolve()
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] archivo de waivers no encontrado: {path}")
        return path

    env_waivers = os.environ.get("LINT_WAIVERS")
    if env_waivers:
        path = Path(env_waivers).resolve()
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] LINT_WAIVERS apunta a un archivo inexistente: {path}")
        return path

    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        default = Path(project_root) / "scripts" / "lint_waivers.txt"
        if default.exists():
            return default.resolve()

    return None


def parse_waivers(waivers_file: Path | None) -> list[str]:
    """
    Lee el archivo de waivers y devuelve la lista de códigos de warning a
    silenciar. Formato: un código por línea, '#' inicia comentario. Se admite
    comentario al final de la línea (EOFNEWLINE  # cosmético).
    """
    if waivers_file is None:
        return []

    codes: list[str] = []
    for raw in waivers_file.read_text().splitlines():
        # quitar comentario de fin de línea
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        codes.append(line)
    return codes


def run_verilator_lint(
    rtl_file: Path,
    top_name: str,
    waiver_codes: list[str],
    verbose: bool,
) -> tuple[int, str]:
    """
    Corre 'verilator --lint-only' sobre el RTL. Devuelve (returncode, output).
    No usa run_command porque queremos capturar la salida SIEMPRE (pase o
    falle) para volcarla al reporte, en vez de abortar.

    En MSYS2/MINGW64 el ejecutable 'verilator' es un script Perl que
    shutil.which() no detecta en Windows (no está en PATHEXT), así que se
    prioriza el binario nativo 'verilator_bin'. Ese binario crudo tampoco
    autocalcula VERILATOR_ROOT (lo hace el wrapper Perl), así que se deriva
    de la ubicación del .exe y se pasa en formato Windows.
    """
    verilator_exe = shutil.which("verilator_bin") or shutil.which("verilator")
    if verilator_exe is None:
        raise RuntimeError("[ERROR] verilator no está en el PATH.")

    env = os.environ.copy()
    if "VERILATOR_ROOT" not in env:
        # layout estándar: <root>/bin/verilator_bin.exe -> <root>/share/verilator
        vroot = Path(verilator_exe).resolve().parent.parent / "share" / "verilator"
        if vroot.exists():
            env["VERILATOR_ROOT"] = to_native_path(str(vroot))

    rtl_native = to_native_path(str(rtl_file))

    # -I por cada directorio de los `include, resueltos recursivamente
    # (el RTL puede depender de otros módulos: modules/<modulo>/rtl).
    missing: list[str] = []
    incdirs = collect_include_dirs(rtl_file, missing=missing)

    if missing:
        raise RuntimeError(
            "[ERROR] no se pudieron resolver estos `include: " + ", ".join(missing)
        )

    cmd = [
        verilator_exe,
        "--lint-only",
        "-Wall",
        "--top-module", top_name,
    ]

    for incdir in incdirs:
        cmd.append(f"-I{to_native_path(str(incdir))}")

    for code in waiver_codes:
        cmd.append(f"-Wno-{code}")
    cmd.append(rtl_native)

    if verbose:
        step_msg("command: " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(rtl_file.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return result.returncode, result.stdout


def write_lint_report(
    report_file: Path,
    module_name: str,
    rtl_file: Path,
    top_name: str,
    waivers_file: Path | None,
    waiver_codes: list[str],
    returncode: int,
    output: str,
    elapsed_s: float,
) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if returncode == 0 else "FAIL"

    with report_file.open("w") as f:
        f.write("LINT REPORT (Verilator)\n")
        f.write("=======================\n\n")
        f.write(f"generated_at : {now_iso()}\n")
        f.write(f"module       : {module_name}\n")
        f.write(f"rtl_top_file : {rtl_file}\n")
        f.write(f"top_module   : {top_name}\n")
        f.write(f"status       : {status}\n")
        f.write(f"returncode   : {returncode}\n")
        f.write(f"elapsed      : {format_seconds(elapsed_s)}\n")
        f.write(f"waivers_file : {waivers_file if waivers_file else '<none>'}\n")
        f.write("\n")
        f.write("WAIVED WARNINGS (-Wno-)\n")
        f.write("-----------------------\n")
        if waiver_codes:
            for code in waiver_codes:
                f.write(f"  {code}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")
        f.write("VERILATOR OUTPUT\n")
        f.write("----------------\n")
        if output.strip():
            f.write(output.rstrip())
            f.write("\n")
        else:
            f.write("(no warnings or errors)\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_lint",
        description="Lint estático del RTL del módulo actual con Verilator.",
    )
    parser.add_argument("--waivers", default=None, metavar="PATH",
                        help="Archivo de waivers (códigos -Wno-). "
                             "Default: $LINT_WAIVERS o scripts/lint_waivers.txt")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    module_name, module_dir, prefix = detect_current_module()
    top_name = module_name  # el top se llama igual que el módulo

    banner("RTL LINT (Verilator)")
    kv("module", module_name)
    kv("module_dir", module_dir)

    try:
        rtl_file = resolve_rtl_top_file(module_name, module_dir, prefix)
        waivers_file = resolve_waivers_file(args.waivers)
        waiver_codes = parse_waivers(waivers_file)
    except FileNotFoundError as exc:
        fail_msg(str(exc))
        return 1

    kv("rtl_top_file", rtl_file)
    kv("top_module", top_name)
    kv("waivers_file", waivers_file if waivers_file else "<none>")
    if waiver_codes:
        kv("waived", ", ".join(waiver_codes))

    section("lint")

    report_file = module_dir / "build" / "reports" / "lint.rpt"

    t0 = time.perf_counter()
    try:
        returncode, output = run_verilator_lint(
            rtl_file=rtl_file,
            top_name=top_name,
            waiver_codes=waiver_codes,
            verbose=args.verbose,
        )
    except Exception as exc:
        fail_msg(f"{module_name}: {exc}")
        return 1
    elapsed = time.perf_counter() - t0

    write_lint_report(
        report_file=report_file,
        module_name=module_name,
        rtl_file=rtl_file,
        top_name=top_name,
        waivers_file=waivers_file,
        waiver_codes=waiver_codes,
        returncode=returncode,
        output=output,
        elapsed_s=elapsed,
    )

    # Mostrar la salida de Verilator en consola (si hubo algo)
    if output.strip():
        print(output.rstrip())

    step_msg(f"lint report: {report_file}")

    if returncode == 0:
        pass_msg(f"{module_name}: lint OK ({format_seconds(elapsed)})")
        return 0
    else:
        fail_msg(f"{module_name}: lint encontró problemas (ver reporte)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
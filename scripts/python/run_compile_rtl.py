#!/usr/bin/env python3
"""
run_compile_rtl.py

Check rápido de COMPILACIÓN + ELABORACIÓN del RTL puro del módulo actual,
SIN testbench y SIN simulación. Sirve para iterar sobre el diseño y cazar
errores de sintaxis y de elaboración (parámetros, generates, anchos de bus)
antes de montar vectores.

Backend según SIM_BACKEND (exportado por set_env.sh):
  - iverilog (msys)  : iverilog -g2012 -s <top> -o <tmp>  (compila y elabora
                       en un paso; con -s indica el top a elaborar)
  - xsim (linux)     : xvlog -sv <rtl>   (compila / analiza)
                       xelab <top>       (elabora la jerarquía)

El módulo se detecta desde el directorio actual, igual que run_regression_sim
y run_regression_vm. El top-level se asume igual al nombre del módulo.

Elabora con los DEFAULTS del RTL (sin defines): verifica que el diseño
compile y elabore con los valores de parámetro por defecto.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from regression_common import (
    banner,
    collect_include_dirs,
    detect_current_module,
    fail_msg,
    kv,
    pass_msg,
    require_env,
    run_command,
    section,
    set_use_color,
    step_msg,
    warn_msg,
)


def resolve_tb_file(module_name: str, module_dir: Path, prefix: str) -> Path:
    """
    Resuelve el archivo del testbench. Prioriza ${PREFIX}_TB_SV / ${PREFIX}_TB_V
    (las mismas variables que usa flist_utils.sh); si no, cae al layout por
    convencion modules/<module>/testbench/<module>_tb.(sv|v).
    """
    for suffix in ("_TB_SV", "_TB_V"):
        var = f"{prefix}{suffix}"
        value = os.environ.get(var)
        if value:
            path = Path(value).resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"[ERROR] {var} apunta a un archivo inexistente: {path}")

    for ext in (".sv", ".v"):
        candidate = (module_dir / "testbench" / f"{module_name}_tb{ext}").resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"[ERROR] no se encontro el testbench de '{module_name}'. "
        f"Defini {prefix}_TB_SV/{prefix}_TB_V o pone el archivo en "
        f"{module_dir / 'testbench' / (module_name + '_tb.sv')}"
    )


def resolve_incdirs(*sources: Path) -> list[Path]:
    """
    Directorios -I necesarios para preprocesar el RTL, siguiendo los `include
    de forma RECURSIVA (fft1d_r2.v -> cmul.v -> cast.v). Los módulos de los que
    se depende viven en $MODULES_ROOT/<modulo>/rtl, igual que en flist_utils.sh.
    """
    missing: list[str] = []
    dirs: list[Path] = []

    for source in sources:
        for d in collect_include_dirs(source, missing=missing):
            if d not in dirs:
                dirs.append(d)

    if missing:
        raise FileNotFoundError(
            "[ERROR] no se pudieron resolver estos `include: "
            + ", ".join(missing)
            + ".\n        Se buscó junto al archivo y en "
            "$MODULES_ROOT/<modulo>/rtl/<archivo>."
        )

    return dirs


def to_native_path(path: str) -> str:
    """
    Convierte un path estilo MSYS (/c/Users/...) a formato Windows
    (C:/Users/...) con cygpath, para pasárselo a binarios nativos de
    Windows (iverilog.exe). En Linux (sin cygpath) devuelve el path igual.
    Solo actúa cuando el backend es iverilog.
    """
    if os.environ.get("SIM_BACKEND") != "iverilog":
        return path
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
    Resuelve el path del archivo RTL top.
    Prioriza las variables de entorno ${PREFIX}_V / ${PREFIX}_SV (las mismas
    que usa flist_utils.sh). Si no están, cae al layout por convención:
    modules/<module>/rtl/<module>.v (o .sv).
    """
    for suffix in ("_V", "_SV"):
        var = f"{prefix}{suffix}"
        value = os.environ.get(var)
        if value:
            path = Path(value).resolve()
            if path.exists():
                return path
            raise FileNotFoundError(
                f"[ERROR] {var} apunta a un archivo inexistente: {path}"
            )

    # Fallback por convención
    for ext in (".v", ".sv"):
        candidate = (module_dir / "rtl" / f"{module_name}{ext}").resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"[ERROR] no se encontró el RTL top de '{module_name}'. "
        f"Definí {prefix}_V/{prefix}_SV o poné el archivo en "
        f"{module_dir / 'rtl' / (module_name + '.v')}"
    )


def compile_iverilog(top_name: str, sources: list[Path], incdirs: list[Path], verbose: bool) -> None:
    """
    Compila + elabora el RTL con Icarus. -s fija el top a elaborar, así no
    necesita testbench. La salida va a un temporal que se descarta: solo
    nos interesa el return code (¿compiló y elaboró sin errores?).
    """
    if shutil.which("iverilog") is None:
        raise RuntimeError("[ERROR] iverilog no está en el PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = to_native_path(str(Path(tmpdir) / "rtl_check.out"))

        cmd = [
            "iverilog",
            "-g2012",
            "-s", top_name,          # top a elaborar
            "-o", out_file,
        ]

        for incdir in incdirs:
            cmd.append(f"-I{to_native_path(str(incdir))}")

        for source in sources:
            cmd.append(to_native_path(str(source)))

        run_command(
            cmd=cmd,
            cwd=sources[0].parent,
            label=f"compile+elaborate (iverilog, top={top_name})",
            verbose=verbose,
        )


def compile_xsim(top_name: str, sources: list[Path], module_dir: Path, incdirs: list[Path], verbose: bool) -> None:
    """
    Compila (xvlog) + elabora (xelab) el RTL con el flujo de Vivado.
    Los logs quedan en un temporal; solo interesa el return code.
    """
    if shutil.which("xvlog") is None or shutil.which("xelab") is None:
        raise RuntimeError(
            "[ERROR] xvlog/xelab no están en el PATH. ¿Sourceaste las settings de Vivado?"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # xvlog: análisis / compilación
        xvlog_cmd = ["xvlog", "-sv"]

        for incdir in incdirs:
            xvlog_cmd.extend(["-i", str(incdir)])

        for source in sources:
            xvlog_cmd.append(str(source))

        run_command(
            cmd=xvlog_cmd,
            cwd=tmpdir,
            label="compile (xvlog)",
            verbose=verbose,
        )

        # xelab: elaboración de la jerarquía desde el top
        xelab_cmd = [
            "xelab",
            top_name,
        ]
        run_command(
            cmd=xelab_cmd,
            cwd=tmpdir,
            label=f"elaborate (xelab, top={top_name})",
            verbose=verbose,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_compile_rtl",
        description="Compila y elabora el RTL del módulo actual. Con --testbench, "
                    "incluye además el testbench.",
    )
    parser.add_argument("--testbench", "-t", action="store_true",
                        help="Compilar y elaborar TAMBIEN el testbench "
                             "(top = <module>_tb). Sin esta flag solo se "
                             "chequea el RTL.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_color:
        set_use_color(False)

    module_name, module_dir, prefix = detect_current_module()

    backend = os.environ.get("SIM_BACKEND", "").strip().lower()

    banner("RTL COMPILE CHECK")
    kv("module", module_name)
    kv("module_dir", module_dir)
    kv("backend", backend or "<undefined>")

    try:
        rtl_file = resolve_rtl_top_file(module_name, module_dir, prefix)
    except FileNotFoundError as exc:
        fail_msg(str(exc))
        return 1

    sources = [rtl_file]

    if args.testbench:
        # el top pasa a ser el testbench, que instancia al RTL
        top_name = f"{module_name}_tb"

        try:
            tb_file = resolve_tb_file(module_name, module_dir, prefix)
        except FileNotFoundError as exc:
            fail_msg(str(exc))
            return 1

        sources.append(tb_file)
        kv("tb_file", tb_file)
    else:
        # por decisión: el top se llama igual que el módulo
        top_name = module_name

    kv("rtl_top_file", rtl_file)
    kv("top_module", top_name)

    try:
        incdirs = resolve_incdirs(*sources)
    except FileNotFoundError as exc:
        fail_msg(str(exc))
        return 1

    kv("incdirs", ", ".join(str(d) for d in incdirs))

    section("compile + elaborate")

    try:
        if backend == "iverilog":
            compile_iverilog(top_name, sources, incdirs, args.verbose)
        elif backend == "xsim":
            compile_xsim(top_name, sources, module_dir, incdirs, args.verbose)
        else:
            fail_msg(
                f"SIM_BACKEND inválido o no definido: '{backend}'. "
                f"Esperado 'xsim' o 'iverilog'. ¿Sourceaste set_env.sh?"
            )
            return 1
    except Exception as exc:
        fail_msg(f"{module_name}: {exc}")
        return 1

    if args.testbench:
        pass_msg(f"{module_name}: RTL + testbench compilan y elaboran correctamente")
    else:
        pass_msg(f"{module_name}.v : RTL compiles and elaborate correctly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
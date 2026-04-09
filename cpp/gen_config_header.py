from pathlib import Path

CONF_PATH = Path("params.conf")
OUT_PATH = Path("inc/config.h")

REQUIRED_KEYS = [
    "NB_S", "NBF_S",
    "NB_Y", "NBF_Y",
    "NB_A", "NBF_A",
]

def parse_conf(path: Path) -> dict[str, int]:
    vals: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(f"Línea inválida {lineno}: {line}")

            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()

            try:
                vals[k] = int(v)
            except ValueError as e:
                raise ValueError(f"Valor no entero en línea {lineno}: {line}") from e

    missing = [k for k in REQUIRED_KEYS if k not in vals]
    if missing:
        raise ValueError(f"Faltan claves en {path}: {missing}")

    return vals

def write_header(vals: dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("#ifndef _CONFIG_H\n")
        f.write("#define _CONFIG_H\n\n")

        for key in REQUIRED_KEYS:
            f.write(f"constexpr int {key} = {vals[key]};\n")

        f.write("\n")
        f.write("constexpr int NBI_S = NB_S - NBF_S;\n")
        f.write("constexpr int NBI_Y = NB_Y - NBF_Y;\n")
        f.write("constexpr int NBI_A = NB_A - NBF_A;\n\n")

        f.write("#endif\n")

def main() -> None:
    vals = parse_conf(CONF_PATH)
    write_header(vals, OUT_PATH)
    print(f"[gen_config_header.py] Generated: {OUT_PATH}")

if __name__ == "__main__":
    main()
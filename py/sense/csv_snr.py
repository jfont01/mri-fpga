import re
import csv
import argparse
from pathlib import Path


def extract_nbf(folder_name: str, sweep_var: str) -> int | None:
    pattern = rf"NB_{sweep_var}\d+_NBF_{sweep_var}(\d+)"
    m = re.search(pattern, folder_name)
    return int(m.group(1)) if m else None


def extract_last_snr(report_path: Path) -> float | None:
    snr_values = []
    pattern = re.compile(r"snr_db\s*:\s*([+-]?\d+(?:\.\d+)?)")

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                snr_values.append(float(m.group(1)))

    if not snr_values:
        return None

    return snr_values[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sweep-var",
        type=str,
        required=True,
        choices=["S", "Y", "A", "B"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.in_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    sweep_var = args.sweep_var.upper()

    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {root}")

    nbf_col = f"NBF_{sweep_var}"
    rows = []

    for report_path in root.rglob("global_compare_report.rpt"):
        parent_folder = report_path.parent.name
        nbf_val = extract_nbf(parent_folder, sweep_var)
        snr_db = extract_last_snr(report_path)

        if nbf_val is None:
            print(f"[WARN] No pude extraer {nbf_col} de: {parent_folder}")
            continue

        if snr_db is None:
            print(f"[WARN] No encontré snr_db en: {report_path}")
            continue

        rows.append({
            nbf_col: nbf_val,
            "snr_db": snr_db,
        })

    rows.sort(key=lambda x: x[nbf_col])

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[nbf_col, "snr_db"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV generado: {out_csv}")
    print(f"[OK] Filas escritas: {len(rows)}")
    print(f"[OK] Variable barrida: {nbf_col}")


if __name__ == "__main__":
    main()
import re
import csv
import argparse
from pathlib import Path


def extract_nbf_y(folder_name: str) -> int | None:
    m = re.search(r"NB_Y\d+_NBF_Y(\d+)", folder_name)
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
        required=True,
        help="Directorio raíz donde buscar global_compare_report.rpt",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Path del CSV de salida",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.in_dir).resolve()
    out_csv = Path(args.out_csv).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {root}")

    rows = []

    for report_path in root.rglob("global_compare_report.rpt"):
        parent_folder = report_path.parent.name
        nbf_y = extract_nbf_y(parent_folder)
        snr_db = extract_last_snr(report_path)

        if nbf_y is None:
            print(f"[WARN] No pude extraer NBF_Y de: {parent_folder}")
            continue

        if snr_db is None:
            print(f"[WARN] No encontré snr_db en: {report_path}")
            continue

        rows.append({
            "NBF_Y": nbf_y,
            "snr_db": snr_db,
        })

    rows.sort(key=lambda x: x["NBF_Y"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["NBF_Y", "snr_db"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV generado: {out_csv}")
    print(f"[OK] Filas escritas: {len(rows)}")


if __name__ == "__main__":
    main()
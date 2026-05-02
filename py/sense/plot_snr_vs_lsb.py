import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(csv_path: Path, sweep_var: str):
    rows = []
    nbf_col = f"NBF_{sweep_var}"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {nbf_col, "snr_db"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"El CSV debe contener las columnas {required}. "
                f"Columnas encontradas: {reader.fieldnames}"
            )

        for row in reader:
            try:
                nbf_val = int(row[nbf_col])
                snr_db = float(row["snr_db"])
            except ValueError:
                continue

            lsb = 2.0 ** (-nbf_val)

            rows.append({
                nbf_col: nbf_val,
                "LSB": lsb,
                "snr_db": snr_db,
            })

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-png",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sweep-var",
        type=str,
        required=True,
        choices=["Y", "A", "B", "S"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.in_csv).resolve()
    out_png = Path(args.out_png).resolve()
    sweep_var = args.sweep_var.upper()
    nbf_col = f"NBF_{sweep_var}"

    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el CSV de entrada: {csv_path}")

    rows = read_csv(csv_path, sweep_var)

    if not rows:
        raise RuntimeError("No se encontraron datos válidos en el CSV.")

    rows.sort(key=lambda x: x["LSB"])

    x = [r["LSB"] for r in rows]
    y = [r["snr_db"] for r in rows]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xscale("log", base=2)
    plt.gca().invert_xaxis()
    plt.xlabel(f"LSB = 2^(-{nbf_col})")
    plt.ylabel("SNR [dB]")
    plt.title(f"SNR vs LSB ({nbf_col})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] PNG generado: {out_png}")
    print(f"[OK] Variable barrida: {nbf_col}")


if __name__ == "__main__":
    main()
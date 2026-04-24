import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(csv_path: Path):
    rows = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"NBF_Y", "snr_db"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"El CSV debe contener las columnas {required}. "
                f"Columnas encontradas: {reader.fieldnames}"
            )

        for row in reader:
            try:
                nbf_y = int(row["NBF_Y"])
                snr_db = float(row["snr_db"])
            except ValueError:
                continue

            lsb = 2.0 ** (-nbf_y)

            rows.append({
                "NBF_Y": nbf_y,
                "LSB": lsb,
                "snr_db": snr_db,
            })

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        type=str,
        required=True,
        help="Path del CSV de entrada",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        required=True,
        help="Path del PNG de salida",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.in_csv).resolve()
    out_png = Path(args.out_png).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el CSV de entrada: {csv_path}")

    rows = read_csv(csv_path)

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
    plt.xlabel("LSB = 2^{-NBF_Y}")
    plt.ylabel("SNR [dB]")
    plt.title("SNR vs LSB")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] PNG generado: {out_png}")


if __name__ == "__main__":
    main()
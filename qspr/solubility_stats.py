from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "AqSolDB_v1.0_min.csv"


def load_solubility(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "Solubility" not in df.columns:
        raise ValueError("Column 'Solubility' not found in the dataset.")
    return df["Solubility"].dropna().astype(float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Basic stats for AqSolDB solubility values."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to AqSolDB CSV (default: AqSolDB_v1.0_min.csv in this folder)",
    )
    args = parser.parse_args()

    values = load_solubility(args.path)
    if values.empty:
        raise ValueError("No solubility values after dropping NaNs.")

    mean = values.mean()
    median = values.median()
    std = values.std(ddof=1)
    min_val = values.min()
    max_val = values.max()
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    print(f"Dataset: {args.path}")
    print(f"Pocet hodnot: {len(values)}")
    print(f"Prumer: {mean:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Smerodatna odchylka (sample): {std:.6f}")
    print(f"Minimum: {min_val:.6f}")
    print(f"25% kvartil: {q25:.6f}")
    print(f"75% kvartil: {q75:.6f}")
    print(f"Maximum: {max_val:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

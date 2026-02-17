from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "random-forest"
MODEL_NAME = "Random Forest"


def _resolve_qspr_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "qspr_config.py").exists() and (candidate / "qspr_common.py").exists():
            return candidate.resolve()
    raise RuntimeError("Could not locate qspr root directory.")


def _sanitize_label(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not cleaned:
        cleaned = "compound"
    return cleaned[:60]


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Highlight top Random Forest Morgan-bit motifs for a SMILES string."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="SMILES string to visualize. If omitted, --dataset-index is used.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Row index from the filtered AqSolDB dataset.",
    )
    parser.add_argument(
        "--top-n-bits",
        type=int,
        default=6,
        help="Number of important active bits to render for the selected molecule.",
    )
    parser.add_argument(
        "--mols-per-row",
        type=int,
        default=3,
        help="How many highlighted motif panels to place in each row.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output filename prefix (without extension).",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    qspr_root = _resolve_qspr_root()

    if str(qspr_root) not in sys.path:
        sys.path.insert(0, str(qspr_root))

    from qspr_config import (
        DATA_PATH,
        ECFP_N_BITS,
        ECFP_RADIUS,
        N_ESTIMATORS,
        N_JOBS,
        OUTPUT_DIRNAME,
        RANDOM_SEED,
    )
    from qspr_common import (
        build_feature_matrix_with_metadata,
        draw_morgan_bit_grid,
        draw_morgan_bit_overlay,
        fingerprint_mol_with_bit_info,
        load_dataset,
        make_binary_target,
    )

    RDLogger.DisableLog("rdApp.warning")

    df = load_dataset(DATA_PATH)
    df, x, mols, bit_info_maps = build_feature_matrix_with_metadata(
        df,
        radius=ECFP_RADIUS,
        n_bits=ECFP_N_BITS,
    )
    y, cutoff = make_binary_target(df["Solubility"].to_numpy())

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )
    rf.fit(x, y)
    bit_scores = rf.feature_importances_

    if args.smiles:
        target_smiles = args.smiles
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise SystemExit(f"Invalid SMILES: {target_smiles}")

        target_vector, target_bit_info = fingerprint_mol_with_bit_info(
            target_mol,
            radius=ECFP_RADIUS,
            n_bits=ECFP_N_BITS,
        )
        target_index = None
        target_id = "custom"
    else:
        if args.dataset_index is None:
            probabilities = rf.predict_proba(x)[:, 1]
            confidence = np.abs(probabilities - 0.5)
            target_index = int(np.argmax(confidence))
        else:
            target_index = int(args.dataset_index)

        if target_index < 0 or target_index >= len(df):
            raise SystemExit(f"--dataset-index must be in [0, {len(df) - 1}]")

        row = df.iloc[target_index]
        target_id = row["ID"] if "ID" in row else target_index
        target_smiles = row["SMILES"]
        target_mol = mols[target_index]
        target_vector = x[target_index]
        target_bit_info = bit_info_maps[target_index]

    probability = float(rf.predict_proba(target_vector.reshape(1, -1))[0, 1])

    try:
        grid_image, bit_summaries = draw_morgan_bit_grid(
            mol=target_mol,
            bit_info_map=target_bit_info,
            bit_scores=bit_scores,
            top_n=args.top_n_bits,
            mols_per_row=args.mols_per_row,
        )
        overlay_image, _ = draw_morgan_bit_overlay(
            mol=target_mol,
            bit_info_map=target_bit_info,
            bit_scores=bit_scores,
            top_n=args.top_n_bits,
            mols_per_row=args.mols_per_row,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output_prefix:
        output_prefix = args.output_prefix
    elif target_index is None:
        output_prefix = f"smiles_{_sanitize_label(target_smiles)}"
    else:
        output_prefix = f"dataset_idx_{target_index}_id_{_sanitize_label(str(target_id))}"

    out_dir = qspr_root / MODEL_DIR / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_image_path = out_dir / f"{output_prefix}_motif_grid.png"
    overlay_image_path = out_dir / f"{output_prefix}_motif_overlay.png"
    csv_path = out_dir / f"{output_prefix}_motif_bits.csv"

    grid_image.save(grid_image_path)
    overlay_image.save(overlay_image_path)

    summary_df = pd.DataFrame(bit_summaries)
    summary_df.insert(0, "smiles", target_smiles)
    summary_df.insert(1, "rf_predicted_p_soluble", probability)
    summary_df.to_csv(csv_path, index=False)

    print(f"Trained on {len(df)} molecules. Binary cutoff (median solubility): {cutoff:.5f}")
    print(f"Selected molecule ID: {target_id}")
    print(f"Selected SMILES: {target_smiles}")
    print(f"RF predicted P(soluble): {probability:.4f}")
    print(f"Saved grid image: {grid_image_path}")
    print(f"Saved overlay image: {overlay_image_path}")
    print(f"Saved bit summary: {csv_path}")
    columns_to_print = [
        "rank",
        "bit",
        "importance",
        "importance_norm",
        "highlight_alpha",
        "occurrences",
        "fragments",
    ]
    available_columns = [col for col in columns_to_print if col in summary_df.columns]
    print(summary_df[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()

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
    parser.add_argument(
        "--boruta-trials",
        type=int,
        default=None,
        help="Number of BorutaShap trials. Defaults to qspr_config.BORUTA_N_TRIALS.",
    )
    parser.add_argument(
        "--boruta-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable BorutaShap sampling for speed.",
    )
    parser.add_argument(
        "--boruta-normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable BorutaShap normalization.",
    )
    return parser


def _parse_bit_index(feature_name):
    feature_name = str(feature_name)
    if not feature_name.startswith("bit_"):
        return None
    try:
        return int(feature_name.split("_", 1)[1])
    except ValueError:
        return None


def _patch_borutashap_for_current_dependencies():
    import scipy.stats as stats
    from scipy.stats import binomtest

    # Compatibility shims for newer numpy/scipy releases used by BorutaShap internals.
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if not hasattr(stats, "binom_test"):
        def _binom_test(k, n=None, p=0.5, alternative="two-sided"):
            k_clean = int(k)
            n_clean = int(n) if n is not None else None
            return binomtest(k=k_clean, n=n_clean, p=p, alternative=alternative).pvalue

        stats.binom_test = _binom_test

    import shap
    from BorutaShap import BorutaShap

    # Patch BorutaShap.explain for SHAP outputs shaped as (n_samples, n_features, n_outputs).
    def _patched_explain(self):
        explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation="tree_path_dependent",
            approximate=True,
        )
        data = self.find_sample() if self.sample else self.X_boruta
        values = explainer.shap_values(data)
        arr = np.array(values)

        if isinstance(values, list):
            arr = np.array(values)
            if arr.ndim == 3:
                arr = np.abs(arr).sum(axis=0).mean(0)
            else:
                arr = np.abs(arr).mean(0)
        elif arr.ndim == 3:
            if arr.shape[0] == data.shape[0] and arr.shape[1] == data.shape[1]:
                arr = np.abs(arr).mean(axis=0)
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
            else:
                arr = np.abs(arr).sum(axis=0).mean(0)
        else:
            arr = np.abs(arr).mean(0)

        self.shap_values = arr

    BorutaShap.explain = _patched_explain
    return BorutaShap


def _compute_borutashap_bit_scores(
    x,
    y,
    n_bits,
    random_seed,
    n_jobs,
    boruta_n_trials,
    boruta_sample,
    boruta_normalize,
    boruta_train_or_test,
    boruta_rf_n_estimators,
    boruta_rf_max_depth,
):
    BorutaShap = _patch_borutashap_for_current_dependencies()

    feature_names = [f"bit_{i}" for i in range(n_bits)]
    x_df = pd.DataFrame(x, columns=feature_names)

    selector = BorutaShap(
        model=RandomForestClassifier(
            n_estimators=boruta_rf_n_estimators,
            max_depth=boruta_rf_max_depth,
            n_jobs=n_jobs,
            random_state=random_seed,
            class_weight="balanced_subsample",
        ),
        importance_measure="shap",
        classification=True,
    )

    selector.fit(
        X=x_df,
        y=y,
        n_trials=boruta_n_trials,
        sample=boruta_sample,
        train_or_test=boruta_train_or_test,
        normalize=boruta_normalize,
        verbose=False,
    )

    raw_scores = np.zeros((n_bits,), dtype=np.float32)
    history_means = selector.history_x.mean(axis=0)
    for feature_name, score in history_means.items():
        bit_idx = _parse_bit_index(feature_name)
        if bit_idx is None or bit_idx < 0 or bit_idx >= n_bits:
            continue
        raw_scores[bit_idx] = float(score)

    accepted = [str(name) for name in getattr(selector, "accepted", [])]
    rejected = [str(name) for name in getattr(selector, "rejected", [])]
    tentative = [str(name) for name in getattr(selector, "tentative", [])]

    accepted_mask = np.zeros((n_bits,), dtype=bool)
    for feature_name in accepted:
        bit_idx = _parse_bit_index(feature_name)
        if bit_idx is None or bit_idx < 0 or bit_idx >= n_bits:
            continue
        accepted_mask[bit_idx] = True

    if accepted_mask.any():
        bit_scores = np.where(accepted_mask, np.maximum(raw_scores, 0.0), 0.0).astype(np.float32)
        scope = "accepted"
    else:
        bit_scores = np.maximum(raw_scores, 0.0).astype(np.float32)
        scope = "all_positive"

    if not np.any(bit_scores > 0):
        bit_scores = np.abs(raw_scores).astype(np.float32)
        scope = f"{scope}_abs_fallback"

    metadata = {
        "scope": scope,
        "score_source": "history_x_mean",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "tentative_count": len(tentative),
        "positive_count": int(np.count_nonzero(bit_scores > 0)),
    }
    return bit_scores, metadata


def main():
    args = _build_arg_parser().parse_args()
    qspr_root = _resolve_qspr_root()

    if str(qspr_root) not in sys.path:
        sys.path.insert(0, str(qspr_root))

    from qspr_config import (
        BORUTA_NORMALIZE,
        BORUTA_N_TRIALS,
        BORUTA_RF_MAX_DEPTH,
        BORUTA_RF_N_ESTIMATORS,
        BORUTA_SAMPLE,
        BORUTA_TRAIN_OR_TEST,
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

    boruta_n_trials = BORUTA_N_TRIALS if args.boruta_trials is None else args.boruta_trials
    boruta_sample = BORUTA_SAMPLE if args.boruta_sample is None else args.boruta_sample
    boruta_normalize = BORUTA_NORMALIZE if args.boruta_normalize is None else args.boruta_normalize

    bit_scores, boruta_meta = _compute_borutashap_bit_scores(
        x=x,
        y=y,
        n_bits=ECFP_N_BITS,
        random_seed=RANDOM_SEED,
        n_jobs=N_JOBS,
        boruta_n_trials=boruta_n_trials,
        boruta_sample=boruta_sample,
        boruta_normalize=boruta_normalize,
        boruta_train_or_test=BORUTA_TRAIN_OR_TEST,
        boruta_rf_n_estimators=BORUTA_RF_N_ESTIMATORS,
        boruta_rf_max_depth=BORUTA_RF_MAX_DEPTH,
    )

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
    print(
        "Importance source: BorutaShap "
        f"(scores={boruta_meta['score_source']}, scope={boruta_meta['scope']}, "
        f"trials={boruta_n_trials}, sample={boruta_sample}, normalize={boruta_normalize})"
    )
    print(
        "BorutaShap feature counts: "
        f"accepted={boruta_meta['accepted_count']}, "
        f"tentative={boruta_meta['tentative_count']}, "
        f"rejected={boruta_meta['rejected_count']}, "
        f"positive_scores={boruta_meta['positive_count']}"
    )
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

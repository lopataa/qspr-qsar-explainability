import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from qsar_config import (
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
    TOP_N_BITS,
)


def _resolve_repo_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "qsar" / "nr_ic_merged.csv").exists() and (candidate / "qspr" / "qspr_common.py").exists():
            return candidate.resolve()
    raise RuntimeError("Could not locate repository root.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate QSAR motif maps/overlays per activity type using Random Forest."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Optional custom SMILES to visualize for every target.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to QSAR CSV (defaults to qsar/nr_ic_merged.csv).",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=ECFP_RADIUS,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=ECFP_N_BITS,
        help="Morgan fingerprint size.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="RandomForest n_estimators for each one-vs-rest classifier.",
    )
    parser.add_argument(
        "--top-n-bits",
        type=int,
        default=TOP_N_BITS,
        help="Top active important bits to visualize.",
    )
    parser.add_argument(
        "--mols-per-row",
        type=int,
        default=3,
        help="Panels per row in motif map image.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="CPU workers for RandomForest.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to qsar/outputs/random-forest-motifs).",
    )
    parser.add_argument(
        "--boruta-trials",
        type=int,
        default=None,
        help="Number of BorutaShap trials. Defaults to qsar_config.BORUTA_N_TRIALS.",
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
    return parser.parse_args()


def _build_dataset(data_path, radius, n_bits):
    df = pd.read_csv(data_path)

    def smiles_to_mol(s):
        if pd.isna(s) or str(s).lower() == "nan":
            return None
        try:
            return Chem.MolFromSmiles(str(s))
        except Exception:
            return None

    df["mol"] = df["Smiles"].map(smiles_to_mol)
    df = df[df["mol"].notna()].copy()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def mol_to_fp(mol):
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    df["fp"] = df["mol"].map(mol_to_fp)

    try:
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(sparse=False)
    target_encoded = encoder.fit_transform(df[["Target"]])
    target_names = [str(name) for name in encoder.categories_[0]]
    df["target_encoded"] = target_encoded.tolist()

    df["fp_tuple"] = df["fp"].apply(lambda arr: tuple(arr))
    grouped_targets = df.groupby("fp_tuple")["target_encoded"].apply(
        lambda rows: np.any(np.vstack(rows.values), axis=0).astype(np.int8)
    )
    grouped_smiles = df.groupby("fp_tuple")["Smiles"].first()
    grouped_mols = df.groupby("fp_tuple")["mol"].first()

    df_agg = grouped_targets.reset_index(name="target")
    df_agg["smiles"] = df_agg["fp_tuple"].map(grouped_smiles)
    df_agg["mol"] = df_agg["fp_tuple"].map(grouped_mols)
    df_agg["fp"] = df_agg["fp_tuple"].apply(lambda x: np.array(x, dtype=np.int8))
    df_agg = df_agg.drop(columns=["fp_tuple"])

    x = np.vstack(df_agg["fp"].values).astype(np.float32)
    y = np.vstack(df_agg["target"].values).astype(np.int8)
    return df_agg, x, y, target_names, generator


def _fingerprint_array(mol, generator, n_bits):
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


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


def _compute_borutashap_bit_scores_for_target(
    x_df,
    y_target,
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
        y=pd.Series(y_target),
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


def _ensure_probability_matrix(proba, n_samples, n_targets):
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape == (n_samples, n_targets):
        return arr
    if isinstance(proba, list):
        stacked = np.column_stack(proba)
        if stacked.shape[0] == n_samples:
            return stacked
        if stacked.shape[1] == n_samples:
            return stacked.T
    if arr.ndim == 1 and n_targets == 1:
        return arr.reshape(-1, 1)
    raise ValueError(f"Unexpected probability shape: {arr.shape}")


def main():
    args = _parse_args()
    repo_root = _resolve_repo_root()

    qspr_dir = repo_root / "qspr"
    if str(qspr_dir) not in sys.path:
        sys.path.insert(0, str(qspr_dir))

    from qspr_common import (
        draw_morgan_bit_grid,
        draw_morgan_bit_overlay,
        fingerprint_mol_with_bit_info,
    )

    data_path = Path(args.data_path) if args.data_path else DATA_PATH
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (repo_root / "qsar" / OUTPUT_DIRNAME / "random-forest-motifs")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_agg, x, y, target_names, fp_generator = _build_dataset(
        data_path=data_path,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    x_df = pd.DataFrame(x, columns=[f"bit_{i}" for i in range(args.n_bits)])

    boruta_n_trials = BORUTA_N_TRIALS if args.boruta_trials is None else args.boruta_trials
    boruta_sample = BORUTA_SAMPLE if args.boruta_sample is None else args.boruta_sample
    boruta_normalize = BORUTA_NORMALIZE if args.boruta_normalize is None else args.boruta_normalize

    model = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            class_weight="balanced_subsample",
        )
    )
    model.fit(x, y)
    probabilities = _ensure_probability_matrix(
        model.predict_proba(x),
        n_samples=x.shape[0],
        n_targets=y.shape[1],
    )

    if len(model.estimators_) != len(target_names):
        raise RuntimeError(
            f"Estimator count {len(model.estimators_)} does not match target count {len(target_names)}."
        )

    custom_mode = args.smiles is not None
    if custom_mode:
        custom_mol = Chem.MolFromSmiles(args.smiles)
        if custom_mol is None:
            raise SystemExit(f"Invalid SMILES: {args.smiles}")

        custom_fp = _fingerprint_array(custom_mol, fp_generator, args.n_bits)
        custom_probs = _ensure_probability_matrix(
            model.predict_proba(custom_fp.reshape(1, -1)),
            n_samples=1,
            n_targets=y.shape[1],
        )[0]
        _, custom_bit_info_map = fingerprint_mol_with_bit_info(
            custom_mol,
            radius=args.radius,
            n_bits=args.n_bits,
            generator=fp_generator,
        )
    else:
        custom_mol = None
        custom_probs = None
        custom_bit_info_map = None

    global_rows = []

    for target_idx, target_name in enumerate(target_names):
        if custom_mode:
            selected_idx = None
            selected_mol = custom_mol
            selected_smiles = str(args.smiles)
            selected_prob = float(custom_probs[target_idx])
            bit_info_map = custom_bit_info_map
            stem = f"{target_name}_custom"
        else:
            target_col = y[:, target_idx]
            positive_idx = np.where(target_col == 1)[0]
            if len(positive_idx) > 0:
                best_local = int(np.argmax(probabilities[positive_idx, target_idx]))
                selected_idx = int(positive_idx[best_local])
            else:
                selected_idx = int(np.argmax(probabilities[:, target_idx]))

            selected_row = df_agg.iloc[selected_idx]
            selected_mol = selected_row["mol"]
            selected_smiles = str(selected_row["smiles"])
            selected_prob = float(probabilities[selected_idx, target_idx])
            _, bit_info_map = fingerprint_mol_with_bit_info(
                selected_mol,
                radius=args.radius,
                n_bits=args.n_bits,
                generator=fp_generator,
            )
            stem = target_name

        bit_scores, boruta_meta = _compute_borutashap_bit_scores_for_target(
            x_df=x_df,
            y_target=y[:, target_idx],
            n_bits=args.n_bits,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
            boruta_n_trials=boruta_n_trials,
            boruta_sample=boruta_sample,
            boruta_normalize=boruta_normalize,
            boruta_train_or_test=BORUTA_TRAIN_OR_TEST,
            boruta_rf_n_estimators=BORUTA_RF_N_ESTIMATORS,
            boruta_rf_max_depth=BORUTA_RF_MAX_DEPTH,
        )

        try:
            map_img, bit_rows = draw_morgan_bit_grid(
                mol=selected_mol,
                bit_info_map=bit_info_map,
                bit_scores=bit_scores,
                top_n=args.top_n_bits,
                mols_per_row=args.mols_per_row,
            )
            overlay_img, _ = draw_morgan_bit_overlay(
                mol=selected_mol,
                bit_info_map=bit_info_map,
                bit_scores=bit_scores,
                top_n=args.top_n_bits,
                mols_per_row=args.mols_per_row,
            )
        except ValueError as exc:
            print(f"[{target_name}] skipped: {exc}")
            continue

        map_path = out_dir / f"{stem}_motif_map.png"
        overlay_path = out_dir / f"{stem}_motif_overlay.png"
        bits_path = out_dir / f"{stem}_motif_bits.csv"
        map_img.save(map_path)
        overlay_img.save(overlay_path)

        bit_df = pd.DataFrame(bit_rows)
        bit_df.insert(0, "target", target_name)
        bit_df.insert(1, "smiles", selected_smiles)
        bit_df.insert(2, "selected_probability", selected_prob)
        bit_df.insert(3, "selection_mode", "custom_smiles" if custom_mode else "dataset_positive")
        bit_df.insert(4, "importance_source", f"BorutaShap:{boruta_meta['score_source']}")
        bit_df.insert(5, "boruta_scope", boruta_meta["scope"])
        bit_df.to_csv(bits_path, index=False)

        print(
            f"[{target_name}] saved map={map_path.name}, overlay={overlay_path.name}, bits={bits_path.name}, "
            f"selected_p={selected_prob:.4f}, boruta_scope={boruta_meta['scope']}, "
            f"accepted={boruta_meta['accepted_count']}, positive_scores={boruta_meta['positive_count']}"
        )
        global_rows.append(
            {
                "target": target_name,
                "selected_smiles": selected_smiles,
                "selected_probability": selected_prob,
                "selection_mode": "custom_smiles" if custom_mode else "dataset_positive",
                "selected_dataset_index": selected_idx if selected_idx is not None else "",
                "importance_source": f"BorutaShap:{boruta_meta['score_source']}",
                "boruta_scope": boruta_meta["scope"],
                "boruta_accepted_count": boruta_meta["accepted_count"],
                "boruta_tentative_count": boruta_meta["tentative_count"],
                "boruta_rejected_count": boruta_meta["rejected_count"],
                "boruta_positive_count": boruta_meta["positive_count"],
                "map_path": str(map_path),
                "overlay_path": str(overlay_path),
                "bits_path": str(bits_path),
            }
        )

    summary_path = out_dir / "target_image_summary.csv"
    pd.DataFrame(global_rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Targets processed: {len(global_rows)} / {len(target_names)}")


if __name__ == "__main__":
    main()

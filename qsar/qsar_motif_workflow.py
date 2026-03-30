from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from qsar_motif_utils import (
    compute_atom_weights_from_top_bits,
    draw_morgan_bit_grid,
    file_signature,
    fingerprint_mol_with_bit_info,
    load_pickle_cache,
    render_plain_molecule_image,
    save_similarity_map_png,
    save_similarity_map_svg,
    save_pickle_cache,
)

CACHE_VERSION = 1


def resolve_repo_root(script_file: str) -> Path:
    script_path = Path(script_file).resolve()
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        script_path.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "qsar" / "nr_ic_merged.csv").exists() and (candidate / "README.md").exists():
            return candidate.resolve()
    raise RuntimeError("Could not locate repository root.")


def build_parser(
    *,
    description: str,
    n_estimators_help: str,
    n_jobs_help: str,
    output_dir_help: str,
    ecfp_radius: int,
    ecfp_n_bits: int,
    n_estimators_default: int,
    top_n_bits_default: int,
    random_seed_default: int,
    n_jobs_default: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        default=ecfp_radius,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=ecfp_n_bits,
        help="Morgan fingerprint size.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=n_estimators_default,
        help=n_estimators_help,
    )
    parser.add_argument(
        "--top-n-bits",
        type=int,
        default=top_n_bits_default,
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
        default=random_seed_default,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=n_jobs_default,
        help=n_jobs_help,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=output_dir_help,
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
    parser.add_argument(
        "--draw-size",
        type=int,
        default=700,
        help="Square image size in pixels for similarity-map rendering.",
    )
    return parser


def build_dataset(data_path: Path, radius: int, n_bits: int):
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


def fingerprint_array(mol, generator, n_bits: int):
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def parse_bit_index(feature_name) -> int | None:
    feature_name = str(feature_name)
    if not feature_name.startswith("bit_"):
        return None
    try:
        return int(feature_name.split("_", 1)[1])
    except ValueError:
        return None


def patch_borutashap_for_current_dependencies():
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


def compute_borutashap_bit_scores_for_target(
    *,
    x_df: pd.DataFrame,
    y_target: np.ndarray,
    n_bits: int,
    random_seed: int,
    n_jobs: int,
    boruta_n_trials: int,
    boruta_sample: bool,
    boruta_normalize: bool,
    boruta_train_or_test: str,
    estimator_builder: Callable[[int, int], object],
):
    BorutaShap = patch_borutashap_for_current_dependencies()

    selector = BorutaShap(
        model=estimator_builder(random_seed, n_jobs),
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
        bit_idx = parse_bit_index(feature_name)
        if bit_idx is None or bit_idx < 0 or bit_idx >= n_bits:
            continue
        raw_scores[bit_idx] = float(score)

    accepted = [str(name) for name in getattr(selector, "accepted", [])]
    rejected = [str(name) for name in getattr(selector, "rejected", [])]
    tentative = [str(name) for name in getattr(selector, "tentative", [])]

    accepted_mask = np.zeros((n_bits,), dtype=bool)
    for feature_name in accepted:
        bit_idx = parse_bit_index(feature_name)
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


def ensure_probability_matrix(proba, n_samples: int, n_targets: int):
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


def run_motif_pipeline(
    *,
    args,
    script_file: str,
    data_path_default: Path,
    output_dirname: str,
    output_subdir: str,
    cache_subdir: str,
    model_cache_filename: str,
    model_cache_kind: str,
    model_cache_meta_extra: dict,
    model_cache_status_key: str,
    model_label: str,
    boruta_n_trials_default: int,
    boruta_sample_default: bool,
    boruta_normalize_default: bool,
    boruta_train_or_test: str,
    boruta_rf_n_estimators: int,
    boruta_rf_max_depth: int,
    model_builder: Callable[[argparse.Namespace], OneVsRestClassifier],
    boruta_estimator_builder: Callable[[int, int], object],
):
    repo_root = resolve_repo_root(script_file)

    data_path = Path(args.data_path) if args.data_path else data_path_default
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (repo_root / "qsar" / output_dirname / output_subdir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_agg, x, y, target_names, fp_generator = build_dataset(
        data_path=data_path,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    x_df = pd.DataFrame(x, columns=[f"bit_{i}" for i in range(args.n_bits)])
    data_sig = file_signature(data_path)
    cache_dir = repo_root / "qsar" / "cache" / cache_subdir

    boruta_n_trials = boruta_n_trials_default if args.boruta_trials is None else args.boruta_trials
    boruta_sample = boruta_sample_default if args.boruta_sample is None else args.boruta_sample
    boruta_normalize = boruta_normalize_default if args.boruta_normalize is None else args.boruta_normalize

    model_cache_path = cache_dir / model_cache_filename
    model_cache_meta = {
        "version": CACHE_VERSION,
        "kind": model_cache_kind,
        "data": data_sig,
        "ecfp_radius": int(args.radius),
        "ecfp_n_bits": int(args.n_bits),
        "n_estimators": int(args.n_estimators),
        "random_seed": int(args.random_seed),
        "n_jobs": int(args.n_jobs),
        "target_names": list(target_names),
    }
    model_cache_meta.update(model_cache_meta_extra)

    model = load_pickle_cache(model_cache_path, model_cache_meta)
    if model is None:
        model = model_builder(args)
        model.fit(x, y)
        save_pickle_cache(model_cache_path, model_cache_meta, model)
        model_cache_status = "miss"
    else:
        model_cache_status = "hit"

    probabilities = ensure_probability_matrix(
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

        custom_fp = fingerprint_array(custom_mol, fp_generator, args.n_bits)
        custom_probs = ensure_probability_matrix(
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

        boruta_cache_path = cache_dir / f"boruta_target_{target_name}.pkl"
        boruta_cache_meta = {
            "version": CACHE_VERSION,
            "kind": "qsar_boruta_target_scores",
            "target_name": str(target_name),
            "data": data_sig,
            "ecfp_radius": int(args.radius),
            "ecfp_n_bits": int(args.n_bits),
            "random_seed": int(args.random_seed),
            "n_jobs": int(args.n_jobs),
            "boruta_n_trials": int(boruta_n_trials),
            "boruta_sample": bool(boruta_sample),
            "boruta_normalize": bool(boruta_normalize),
            "boruta_train_or_test": str(boruta_train_or_test),
            "boruta_rf_n_estimators": int(boruta_rf_n_estimators),
            "boruta_rf_max_depth": int(boruta_rf_max_depth),
        }

        cached_boruta = load_pickle_cache(boruta_cache_path, boruta_cache_meta)
        if cached_boruta is None:
            bit_scores, boruta_meta = compute_borutashap_bit_scores_for_target(
                x_df=x_df,
                y_target=y[:, target_idx],
                n_bits=args.n_bits,
                random_seed=args.random_seed,
                n_jobs=args.n_jobs,
                boruta_n_trials=boruta_n_trials,
                boruta_sample=boruta_sample,
                boruta_normalize=boruta_normalize,
                boruta_train_or_test=boruta_train_or_test,
                estimator_builder=boruta_estimator_builder,
            )
            save_pickle_cache(
                boruta_cache_path,
                boruta_cache_meta,
                {"bit_scores": bit_scores, "boruta_meta": boruta_meta},
            )
            boruta_cache_status = "miss"
        else:
            bit_scores = np.asarray(cached_boruta["bit_scores"], dtype=np.float32)
            boruta_meta = dict(cached_boruta["boruta_meta"])
            boruta_cache_status = "hit"

        try:
            _, bit_rows = draw_morgan_bit_grid(
                mol=selected_mol,
                bit_info_map=bit_info_map,
                bit_scores=bit_scores,
                top_n=args.top_n_bits,
                mols_per_row=args.mols_per_row,
            )
            atom_weights = compute_atom_weights_from_top_bits(
                mol=selected_mol,
                bit_info_map=bit_info_map,
                bit_scores=bit_scores,
                top_n=args.top_n_bits,
                aggregation="sum",
            )
        except ValueError as exc:
            print(f"[{target_name}] skipped: {exc}")
            continue

        map_path = out_dir / f"{stem}_motif_map.png"
        map_svg_path = out_dir / f"{stem}_motif_map.svg"
        overlay_path = out_dir / f"{stem}_motif_overlay.png"
        bits_path = out_dir / f"{stem}_motif_bits.csv"

        save_similarity_map_png(
            mol=selected_mol,
            atom_weights=atom_weights,
            output_path=map_path,
            draw_size=int(args.draw_size),
        )
        save_similarity_map_svg(
            mol=selected_mol,
            atom_weights=atom_weights,
            output_path=map_svg_path,
            draw_size=int(args.draw_size),
        )
        plain_image = render_plain_molecule_image(selected_mol, draw_size=int(args.draw_size))
        plain_image.save(overlay_path)

        bit_df = pd.DataFrame(bit_rows)
        bit_df.insert(0, "target", target_name)
        bit_df.insert(1, "smiles", selected_smiles)
        bit_df.insert(2, "selected_probability", selected_prob)
        bit_df.insert(3, "selection_mode", "custom_smiles" if custom_mode else "dataset_positive")
        bit_df.insert(4, "importance_source", f"BorutaShap:{boruta_meta['score_source']}")
        bit_df.insert(5, "boruta_scope", boruta_meta["scope"])
        bit_df.to_csv(bits_path, index=False)

        print(
            f"[{target_name}] saved map={map_path.name}, map_svg={map_svg_path.name}, overlay={overlay_path.name}, bits={bits_path.name}, "
            f"selected_p={selected_prob:.4f}, cache({model_label}={model_cache_status}, boruta={boruta_cache_status}), "
            f"boruta_scope={boruta_meta['scope']}, "
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
                model_cache_status_key: model_cache_status,
                "boruta_cache_status": boruta_cache_status,
                "map_path": str(map_path),
                "map_svg_path": str(map_svg_path),
                "overlay_path": str(overlay_path),
                "bits_path": str(bits_path),
            }
        )

    summary_path = out_dir / "target_image_summary.csv"
    pd.DataFrame(global_rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Targets processed: {len(global_rows)} / {len(target_names)}")

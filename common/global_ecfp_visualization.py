from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

DEFAULT_MPL_DIR = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

CACHE_VERSION = 1
DEFAULT_DRAW_SIZE = 700
SHAP_FEATURE_MATRIX_VERSION = 2
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

RECEPTOR_CZECH_NAMES = {
    "ar": "Androgenní receptor",
    "era": "Estrogenový receptor alfa",
    "erb": "Estrogenový receptor beta",
    "gr": "Glukokortikoidní receptor",
    "mr": "Mineralokortikoidní receptor",
    "pr": "Progesteronový receptor",
}

_PANEL_FONT_CACHE: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def _resolve_repo_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "qspr" / "qspr_common.py").exists() and (candidate / "qsar" / "nr_ic_merged.csv").exists():
            return candidate.resolve()
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = _resolve_repo_root()
QSPR_DIR = REPO_ROOT / "qspr"
QSAR_DIR = REPO_ROOT / "qsar"

for path in (QSPR_DIR, QSAR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qspr_common import (  # noqa: E402
    apply_plot_style,
    build_feature_matrix_with_metadata,
    file_signature,
    fingerprint_mol_with_bit_info,
    load_dataset,
    load_pickle_cache,
    make_binary_target,
    save_pickle_cache,
)
from qspr_config import (  # noqa: E402
    BORUTA_NORMALIZE as QSPR_BORUTA_NORMALIZE,
    BORUTA_N_TRIALS as QSPR_BORUTA_N_TRIALS,
    BORUTA_RF_MAX_DEPTH as QSPR_BORUTA_RF_MAX_DEPTH,
    BORUTA_RF_N_ESTIMATORS as QSPR_BORUTA_RF_N_ESTIMATORS,
    BORUTA_SAMPLE as QSPR_BORUTA_SAMPLE,
    BORUTA_TRAIN_OR_TEST as QSPR_BORUTA_TRAIN_OR_TEST,
    DATA_PATH as QSPR_DATA_PATH,
    ECFP_N_BITS as QSPR_ECFP_N_BITS,
    ECFP_RADIUS as QSPR_ECFP_RADIUS,
    N_ESTIMATORS as QSPR_N_ESTIMATORS,
    N_JOBS as QSPR_N_JOBS,
    RANDOM_SEED as QSPR_RANDOM_SEED,
    TOP_N_BITS as QSPR_TOP_N_BITS,
)
from qsar_config import (  # noqa: E402
    BORUTA_NORMALIZE as QSAR_BORUTA_NORMALIZE,
    BORUTA_N_TRIALS as QSAR_BORUTA_N_TRIALS,
    BORUTA_RF_MAX_DEPTH as QSAR_BORUTA_RF_MAX_DEPTH,
    BORUTA_RF_N_ESTIMATORS as QSAR_BORUTA_RF_N_ESTIMATORS,
    BORUTA_SAMPLE as QSAR_BORUTA_SAMPLE,
    BORUTA_TRAIN_OR_TEST as QSAR_BORUTA_TRAIN_OR_TEST,
    DATA_PATH as QSAR_DATA_PATH,
    ECFP_N_BITS as QSAR_ECFP_N_BITS,
    ECFP_RADIUS as QSAR_ECFP_RADIUS,
    N_ESTIMATORS as QSAR_N_ESTIMATORS,
    N_JOBS as QSAR_N_JOBS,
    RANDOM_SEED as QSAR_RANDOM_SEED,
    TOP_N_BITS as QSAR_TOP_N_BITS,
)


MODE_DEFAULTS = {
    "qspr": {
        "data_path": QSPR_DATA_PATH,
        "radius": QSPR_ECFP_RADIUS,
        "n_bits": QSPR_ECFP_N_BITS,
        "n_estimators": QSPR_N_ESTIMATORS,
        "n_jobs": QSPR_N_JOBS,
        "random_seed": QSPR_RANDOM_SEED,
        "top_n_bits": QSPR_TOP_N_BITS,
        "boruta_n_trials": QSPR_BORUTA_N_TRIALS,
        "boruta_sample": QSPR_BORUTA_SAMPLE,
        "boruta_normalize": QSPR_BORUTA_NORMALIZE,
        "boruta_train_or_test": QSPR_BORUTA_TRAIN_OR_TEST,
        "boruta_rf_n_estimators": QSPR_BORUTA_RF_N_ESTIMATORS,
        "boruta_rf_max_depth": QSPR_BORUTA_RF_MAX_DEPTH,
    },
    "qsar": {
        "data_path": QSAR_DATA_PATH,
        "radius": QSAR_ECFP_RADIUS,
        "n_bits": QSAR_ECFP_N_BITS,
        "n_estimators": QSAR_N_ESTIMATORS,
        "n_jobs": QSAR_N_JOBS,
        "random_seed": QSAR_RANDOM_SEED,
        "top_n_bits": QSAR_TOP_N_BITS,
        "boruta_n_trials": QSAR_BORUTA_N_TRIALS,
        "boruta_sample": QSAR_BORUTA_SAMPLE,
        "boruta_normalize": QSAR_BORUTA_NORMALIZE,
        "boruta_train_or_test": QSAR_BORUTA_TRAIN_OR_TEST,
        "boruta_rf_n_estimators": QSAR_BORUTA_RF_N_ESTIMATORS,
        "boruta_rf_max_depth": QSAR_BORUTA_RF_MAX_DEPTH,
    },
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Global ECFP fragment visualization for this repository. "
            "Select QSPR or QSAR, and SHAP or BorutaShap, then render a "
            "single-molecule similarity map from globally important Morgan fragments."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("qspr", "qsar"),
        required=True,
        help="Repository branch to explain.",
    )
    parser.add_argument(
        "--importance",
        choices=("shap", "boruta-shap"),
        required=True,
        help="Global bit-importance source.",
    )
    parser.add_argument(
        "--model",
        choices=("rf", "xgb"),
        default="rf",
        help="Model backend for prediction and SHAP/Boruta explainability.",
    )
    parser.add_argument(
        "--model-delta",
        action="store_true",
        help="Compare RF vs XGBoost bit importance for the current run and save delta outputs.",
    )
    parser.add_argument(
        "--receptor",
        type=str,
        default=None,
        help="QSAR receptor to explain (required for --mode qsar).",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Custom SMILES string to visualize.",
    )
    parser.add_argument(
        "--delta-smiles",
        type=str,
        default=None,
        help="Optional second SMILES string to compare against --smiles.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Legacy dataset row selector (not supported; use --smiles).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional dataset override.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Morgan fingerprint radius override.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help="Morgan fingerprint size override.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Model n_estimators override.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="CPU workers override.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--top-n-bits",
        type=int,
        default=None,
        help="Number of globally important bits to convert into fragment queries.",
    )
    parser.add_argument(
        "--atom-aggregation",
        choices=("sum", "mean"),
        default="sum",
        help="How to aggregate fragment hits on atoms.",
    )
    parser.add_argument(
        "--boruta-trials",
        type=int,
        default=None,
        help="BorutaShap trials override.",
    )
    parser.add_argument(
        "--boruta-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable BorutaShap sampling.",
    )
    parser.add_argument(
        "--boruta-normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable BorutaShap normalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix without extension.",
    )
    parser.add_argument(
        "--draw-size",
        type=int,
        default=DEFAULT_DRAW_SIZE,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=1024,
        help="Maximum number of rows to use when computing SHAP importances.",
    )
    parser.add_argument(
        "--receptor-grid",
        type=str,
        default=None,
        help="QSAR only: render all receptor maps into a COLSxROWS grid, e.g. 3x2.",
    )
    parser.add_argument(
        "--receptor-grid-top-n",
        type=int,
        default=None,
        help="QSAR grid only: limit the number of top bits used in each receptor panel visualization.",
    )
    parser.add_argument(
        "--top-n-chart",
        action="store_true",
        help="Save a top-N bit-importance bar chart for the selected explainability run.",
    )
    return parser


def _sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned[:80] or "item"


def _parse_grid_shape(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", str(value).lower())
    if not match:
        raise SystemExit("--receptor-grid must be formatted as COLSxROWS, for example 3x2.")
    cols = int(match.group(1))
    rows = int(match.group(2))
    if cols <= 0 or rows <= 0:
        raise SystemExit("--receptor-grid dimensions must be positive integers.")
    return cols, rows


def _parse_bit_index(feature_name: str) -> int | None:
    value = str(feature_name)
    if not value.startswith("bit_"):
        return None
    try:
        return int(value.split("_", 1)[1])
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
            return binomtest(
                k=int(k),
                n=int(n) if n is not None else None,
                p=p,
                alternative=alternative,
            ).pvalue

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
        values = explainer.shap_values(data, check_additivity=False)
        arr = _shap_values_to_feature_matrix(values, n_samples=data.shape[0], n_features=data.shape[1])
        self.shap_values = np.abs(arr).mean(axis=0)

    BorutaShap.explain = _patched_explain
    return BorutaShap


def _shap_values_to_feature_matrix(values, n_samples: int, n_features: int) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[1] == n_samples and arr.shape[2] == n_features:
                return np.asarray(arr[-1], dtype=np.float32)
            if arr.shape[0] == n_samples and arr.shape[1] == n_features:
                return np.asarray(arr[:, :, -1], dtype=np.float32)
        raise ValueError(f"Unsupported SHAP list shape: {arr.shape}")

    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape == (n_samples, n_features):
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return arr[:, :, -1].astype(np.float32)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return arr[-1].astype(np.float32)
        if arr.shape[0] == n_features and arr.shape[1] == n_samples:
            return arr[:, :, -1].T.astype(np.float32)
    raise ValueError(f"Unsupported SHAP array shape: {arr.shape}")


def _compute_shap_bit_scores(model, x: np.ndarray, sample_size: int, random_seed: int):
    import shap

    if sample_size > 0 and len(x) > sample_size:
        rng = np.random.default_rng(random_seed)
        indices = np.sort(rng.choice(len(x), size=int(sample_size), replace=False))
        x_eval = x[indices]
    else:
        indices = None
        x_eval = x

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        approximate=True,
    )
    shap_matrix = _shap_values_to_feature_matrix(
        explainer.shap_values(x_eval, check_additivity=False),
        n_samples=x_eval.shape[0],
        n_features=x_eval.shape[1],
    )
    bit_scores = np.abs(shap_matrix).mean(axis=0).astype(np.float32)
    return bit_scores, {
        "score_source": "mean_abs_shap",
        "scope": "all_bits",
        "positive_count": int(np.count_nonzero(bit_scores > 0)),
        "sample_size": int(x_eval.shape[0]),
        "sampled": bool(indices is not None),
    }


def _compute_borutashap_bit_scores(
    x: np.ndarray,
    y: np.ndarray,
    n_bits: int,
    model_backend: str,
    classification: bool,
    random_seed: int,
    n_jobs: int,
    boruta_n_trials: int,
    boruta_sample: bool,
    boruta_normalize: bool,
    boruta_train_or_test: str,
    boruta_rf_n_estimators: int,
    boruta_rf_max_depth: int,
):
    BorutaShap = _patch_borutashap_for_current_dependencies()

    feature_names = [f"bit_{idx}" for idx in range(n_bits)]
    x_df = pd.DataFrame(x, columns=feature_names)
    y_series = pd.Series(y)

    if model_backend == "rf":
        boruta_model = RandomForestClassifier(
            n_estimators=boruta_rf_n_estimators,
            max_depth=boruta_rf_max_depth,
            n_jobs=n_jobs,
            random_state=random_seed,
            class_weight="balanced_subsample" if classification else None,
        )
    elif model_backend == "xgb":
        XGBClassifier, _ = _require_xgboost()
        boruta_model = XGBClassifier(
            n_estimators=boruta_rf_n_estimators,
            max_depth=boruta_rf_max_depth,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            objective="binary:logistic",
            eval_metric="logloss",
            importance_type="gain",
            n_jobs=n_jobs,
            random_state=random_seed,
            verbosity=0,
        )
    else:
        raise SystemExit(f"Unsupported model backend: {model_backend}")

    selector = BorutaShap(
        model=boruta_model,
        importance_measure="shap",
        classification=classification,
    )
    selector.fit(
        X=x_df,
        y=y_series,
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


def _smiles_to_mol(smiles: str):
    if pd.isna(smiles) or str(smiles).lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def _fingerprint_array(mol, generator, n_bits: int) -> np.ndarray:
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _load_qspr_problem(data_path: Path, radius: int, n_bits: int):
    df = load_dataset(data_path)
    df, x, mols, bit_info_maps = build_feature_matrix_with_metadata(
        df,
        radius=radius,
        n_bits=n_bits,
    )
    y, cutoff = make_binary_target(df["Solubility"].to_numpy())
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return {
        "df": df,
        "x": x,
        "y": y,
        "y_value": df["Solubility"].to_numpy(dtype=np.float32),
        "mols": mols,
        "bit_info_maps": bit_info_maps,
        "generator": generator,
        "target_name": "is_soluble",
        "cutoff": float(cutoff),
    }


def _build_qsar_dataset(data_path: Path, radius: int, n_bits: int):
    df = pd.read_csv(data_path)
    df["mol"] = df["Smiles"].map(_smiles_to_mol)
    df = df[df["mol"].notna()].copy()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    bit_info_maps = []
    fp_arrays = []
    for mol in df["mol"]:
        arr, bit_info_map = fingerprint_mol_with_bit_info(
            mol,
            radius=radius,
            n_bits=n_bits,
            generator=generator,
        )
        fp_arrays.append(arr.astype(np.float32))
        bit_info_maps.append(bit_info_map)

    df["fp"] = fp_arrays
    df["bit_info_map"] = bit_info_maps
    df["fp_tuple"] = df["fp"].apply(lambda arr: tuple(int(v) for v in arr))

    grouped = df.groupby("fp_tuple")
    receptor_lists = grouped["Target"].apply(lambda values: sorted({str(v) for v in values if pd.notna(v)}))
    smiles = grouped["Smiles"].first()
    mols = grouped["mol"].first()
    bit_infos = grouped["bit_info_map"].first()

    df_agg = receptor_lists.reset_index(name="receptors")
    df_agg["smiles"] = df_agg["fp_tuple"].map(smiles)
    df_agg["mol"] = df_agg["fp_tuple"].map(mols)
    df_agg["bit_info_map"] = df_agg["fp_tuple"].map(bit_infos)
    df_agg["fp"] = df_agg["fp_tuple"].apply(lambda row: np.asarray(row, dtype=np.float32))
    df_agg = df_agg.drop(columns=["fp_tuple"])

    x = np.vstack(df_agg["fp"].values).astype(np.float32)
    receptor_names = sorted({name for row in df_agg["receptors"] for name in row})
    return df_agg, x, receptor_names, generator


def _load_qsar_problem(data_path: Path, radius: int, n_bits: int, receptor: str):
    df, x, receptor_names, generator = _build_qsar_dataset(data_path, radius=radius, n_bits=n_bits)
    if receptor not in receptor_names:
        available = ", ".join(receptor_names)
        raise SystemExit(f"Unknown receptor '{receptor}'. Available receptors: {available}")

    y = df["receptors"].apply(lambda values: int(receptor in values)).to_numpy(dtype=np.int8)
    positive_count = int(y.sum())
    if positive_count == 0:
        raise SystemExit(f"Receptor '{receptor}' has no positive rows after aggregation.")
    if positive_count == len(y):
        raise SystemExit(f"Receptor '{receptor}' is positive for every aggregated row; cannot train a binary explainer.")

    return {
        "df": df,
        "x": x,
        "y": y,
        "mols": list(df["mol"].values),
        "bit_info_maps": list(df["bit_info_map"].values),
        "generator": generator,
        "target_name": receptor,
        "positive_count": positive_count,
        "receptor_names": receptor_names,
    }


def _select_qspr_target(problem, model, args, radius: int, n_bits: int):
    if args.smiles:
        smiles = str(args.smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SystemExit(f"Invalid SMILES: {smiles}")
        vector, bit_info_map = fingerprint_mol_with_bit_info(
            mol,
            radius=radius,
            n_bits=n_bits,
            generator=problem["generator"],
        )
        predicted = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
        return {
            "selection_mode": "custom_smiles",
            "dataset_index": None,
            "smiles": smiles,
            "mol": mol,
            "vector": vector,
            "bit_info_map": bit_info_map,
            "predicted_probability": predicted,
            "receptor_name": problem["target_name"],
        }

    df = problem["df"]
    x = problem["x"]
    if args.dataset_index is None:
        proba = model.predict_proba(x)[:, 1]
        confidence = np.abs(proba - 0.5)
        dataset_index = int(np.argmax(confidence))
    else:
        dataset_index = int(args.dataset_index)

    if dataset_index < 0 or dataset_index >= len(df):
        raise SystemExit(f"--dataset-index must be in [0, {len(df) - 1}]")

    row = df.iloc[dataset_index]
    predicted = float(model.predict_proba(x[dataset_index].reshape(1, -1))[0, 1])
    return {
        "selection_mode": "dataset_row",
        "dataset_index": dataset_index,
        "smiles": str(row["SMILES"]),
        "mol": problem["mols"][dataset_index],
        "vector": x[dataset_index],
        "bit_info_map": problem["bit_info_maps"][dataset_index],
        "predicted_probability": predicted,
    }


def _select_qsar_target(problem, model, args, radius: int, n_bits: int):
    if args.smiles:
        smiles = str(args.smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SystemExit(f"Invalid SMILES: {smiles}")
        vector, bit_info_map = fingerprint_mol_with_bit_info(
            mol,
            radius=radius,
            n_bits=n_bits,
            generator=problem["generator"],
        )
        predicted = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
        return {
            "selection_mode": "custom_smiles",
            "dataset_index": None,
            "smiles": smiles,
            "mol": mol,
            "vector": vector,
            "bit_info_map": bit_info_map,
            "predicted_probability": predicted,
            "receptor_name": problem["target_name"],
        }

    df = problem["df"]
    x = problem["x"]
    y = problem["y"]
    proba = model.predict_proba(x)[:, 1]

    if args.dataset_index is not None:
        dataset_index = int(args.dataset_index)
    else:
        positive_idx = np.where(y == 1)[0]
        if len(positive_idx) > 0:
            dataset_index = int(positive_idx[np.argmax(proba[positive_idx])])
        else:
            dataset_index = int(np.argmax(proba))

    if dataset_index < 0 or dataset_index >= len(df):
        raise SystemExit(f"--dataset-index must be in [0, {len(df) - 1}]")

    row = df.iloc[dataset_index]
    predicted = float(proba[dataset_index])
    return {
        "selection_mode": "dataset_row",
        "dataset_index": dataset_index,
        "smiles": str(row["smiles"]),
        "mol": problem["mols"][dataset_index],
        "vector": x[dataset_index],
        "bit_info_map": problem["bit_info_maps"][dataset_index],
        "predicted_probability": predicted,
        "receptor_name": problem["target_name"],
    }


def _fallback_candidate_indices(problem, model, mode: str) -> list[int]:
    x = problem["x"]
    proba = model.predict_proba(x)[:, 1]

    if mode == "qspr":
        confidence = np.abs(proba - 0.5)
        return [int(idx) for idx in np.argsort(confidence)[::-1]]

    y = problem["y"]
    positive_idx = np.where(y == 1)[0]
    ordered = []
    if len(positive_idx) > 0:
        ordered.extend(int(idx) for idx in positive_idx[np.argsort(proba[positive_idx])[::-1]])
    seen = set(ordered)
    for idx in np.argsort(proba)[::-1]:
        idx_int = int(idx)
        if idx_int not in seen:
            ordered.append(idx_int)
    return ordered


def _bitinfo_to_smarts(bit_info_map, mol):
    bit_to_smarts = {}
    for bit, atom_info in bit_info_map.items():
        fragments = set()
        for atom_id, radius in atom_info:
            env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_id)))
            atoms_to_use = {int(atom_id)}
            for bond_idx in env:
                bond = mol.GetBondWithIdx(int(bond_idx))
                atoms_to_use.add(bond.GetBeginAtomIdx())
                atoms_to_use.add(bond.GetEndAtomIdx())

            enlarged_env = set(env)
            for atom_idx in atoms_to_use:
                atom = mol.GetAtomWithIdx(int(atom_idx))
                for bond in atom.GetBonds():
                    if bond.GetIdx() not in enlarged_env:
                        enlarged_env.add(bond.GetIdx())

            outside_neighbors = set()
            for atom_idx in atoms_to_use:
                atom = mol.GetAtomWithIdx(int(atom_idx))
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in atoms_to_use:
                        outside_neighbors.add(neighbor_idx)

            alt_mol = Chem.Mol(mol)
            for atom_idx in outside_neighbors:
                alt_mol.GetAtomWithIdx(int(atom_idx)).SetAtomicNum(0)

            submol = Chem.PathToSubmol(alt_mol, sorted(enlarged_env))
            smarts = Chem.MolToSmarts(submol).replace("[#0]", "*")
            if smarts:
                fragments.add(smarts)

        if fragments:
            bit_to_smarts[int(bit)] = fragments
    return bit_to_smarts


def _build_global_fragment_table(
    mols,
    bit_info_maps,
    bit_scores: np.ndarray,
    selected_bits: np.ndarray,
    allow_negative: bool = False,
):
    stats = {}
    selected_set = {int(bit) for bit in selected_bits}
    for mol, bit_info_map in zip(mols, bit_info_maps):
        filtered_bit_info_map = {
            int(bit): occurrences
            for bit, occurrences in bit_info_map.items()
            if int(bit) in selected_set
        }
        if not filtered_bit_info_map:
            continue

        bit_to_smarts = _bitinfo_to_smarts(filtered_bit_info_map, mol)
        for bit, smarts_set in bit_to_smarts.items():
            score = float(bit_scores[bit])
            if not smarts_set:
                continue
            if allow_negative:
                if np.isclose(score, 0.0):
                    continue
            elif score <= 0:
                continue
            shared_score = score / float(len(smarts_set))
            for smarts in smarts_set:
                row = stats.setdefault(
                    smarts,
                    {
                        "smarts": smarts,
                        "occurrence_count": 0,
                        "contribution_sum": 0.0,
                        "source_bits": set(),
                    },
                )
                row["occurrence_count"] += 1
                row["contribution_sum"] += shared_score
                row["source_bits"].add(int(bit))

    fragment_rows = []
    for row in stats.values():
        occurrence_count = int(row["occurrence_count"])
        contribution_sum = float(row["contribution_sum"])
        fragment_rows.append(
            {
                "smarts": row["smarts"],
                "occurrence_count": occurrence_count,
                "contribution_sum": contribution_sum,
                "contribution_mean": contribution_sum / float(occurrence_count),
                "source_bits": ";".join(str(bit) for bit in sorted(row["source_bits"])),
                "source_bit_count": len(row["source_bits"]),
            }
        )

    if not fragment_rows:
        raise SystemExit("No fragment SMARTS could be derived from the selected important bits.")

    fragment_df = pd.DataFrame(fragment_rows).sort_values(
        ["contribution_mean", "contribution_sum", "occurrence_count"],
        ascending=[False, False, False],
    )
    fragment_df.reset_index(drop=True, inplace=True)
    fragment_df.insert(0, "rank", np.arange(1, len(fragment_df) + 1))
    return fragment_df


def _match_fragments_to_molecule(mol, fragment_df: pd.DataFrame, aggregation: str):
    atom_weights = np.zeros((mol.GetNumAtoms(),), dtype=np.float32)
    atom_hit_counts = np.zeros((mol.GetNumAtoms(),), dtype=np.int32)
    matched_rows = []

    for row in fragment_df.itertuples(index=False):
        query = Chem.MolFromSmarts(row.smarts)
        if query is None:
            continue
        matches = mol.GetSubstructMatches(query)
        if not matches:
            continue

        unique_matches = []
        seen = set()
        for match in matches:
            match_key = tuple(int(idx) for idx in match)
            if match_key not in seen:
                seen.add(match_key)
                unique_matches.append(match_key)

        for match in unique_matches:
            for atom_idx in match:
                atom_weights[atom_idx] += float(row.contribution_mean)
                atom_hit_counts[atom_idx] += 1

        matched_rows.append(
            {
                "fragment_rank": int(row.rank),
                "smarts": row.smarts,
                "occurrence_count": int(row.occurrence_count),
                "contribution_sum": float(row.contribution_sum),
                "contribution_mean": float(row.contribution_mean),
                "source_bits": row.source_bits,
                "source_bit_count": int(row.source_bit_count),
                "matches_on_target": len(unique_matches),
                "matched_atoms": "; ".join(",".join(str(idx) for idx in match) for match in unique_matches),
            }
        )

    if aggregation == "mean":
        nonzero = atom_hit_counts > 0
        atom_weights = atom_weights.copy()
        atom_weights[nonzero] = atom_weights[nonzero] / atom_hit_counts[nonzero]

    atom_rows = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_rows.append(
            {
                "atom_index": idx,
                "symbol": atom.GetSymbol(),
                "weight": float(atom_weights[idx]),
                "hit_count": int(atom_hit_counts[idx]),
            }
        )

    return atom_weights, pd.DataFrame(atom_rows), pd.DataFrame(matched_rows)


def _delta_similarity_colormap():
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        "delta_purple",
        ["#4A148C", "#FFFFFF", "#9C4DCC"],
        N=256,
    )


def _draw_similarity_map_png(mol, weights: np.ndarray, output_path: Path, draw_size: int, color_map=None):
    drawer = Draw.MolDraw2DCairo(draw_size, draw_size)
    kwargs = {"draw2d": drawer}
    if color_map is not None:
        kwargs["colorMap"] = color_map
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(map(float, weights)), **kwargs)
    drawer.FinishDrawing()
    output_path.write_bytes(drawer.GetDrawingText())


def _draw_similarity_map_svg(mol, weights: np.ndarray, output_path: Path, draw_size: int, color_map=None):
    drawer = rdMolDraw2D.MolDraw2DSVG(draw_size, draw_size)
    kwargs = {"draw2d": drawer}
    if color_map is not None:
        kwargs["colorMap"] = color_map
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(map(float, weights)), **kwargs)
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def _select_target_with_fallback(problem, model, args, mode: str, radius: int, n_bits: int, fragment_df: pd.DataFrame):
    selector = _select_qspr_target if mode == "qspr" else _select_qsar_target
    target_info = selector(problem, model, args, radius=radius, n_bits=n_bits)
    atom_weights, atom_df, matched_df = _match_fragments_to_molecule(
        target_info["mol"],
        fragment_df,
        aggregation=args.atom_aggregation,
    )

    if np.any(np.abs(atom_weights) > 0):
        return target_info, atom_weights, atom_df, matched_df

    # Only auto-fallback when the user did not pin a specific molecule.
    if args.smiles is not None or args.dataset_index is not None:
        raise SystemExit("No global fragments matched the selected molecule.")

    for dataset_index in _fallback_candidate_indices(problem, model, mode):
        if target_info["dataset_index"] == dataset_index:
            continue
        fallback_args = argparse.Namespace(smiles=None, dataset_index=dataset_index)
        candidate_info = selector(problem, model, fallback_args, radius=radius, n_bits=n_bits)
        atom_weights, atom_df, matched_df = _match_fragments_to_molecule(
            candidate_info["mol"],
            fragment_df,
            aggregation=args.atom_aggregation,
        )
        if np.any(np.abs(atom_weights) > 0):
            candidate_info = dict(candidate_info)
            candidate_info["selection_mode"] = f"{candidate_info['selection_mode']}_fallback"
            return candidate_info, atom_weights, atom_df, matched_df

    raise SystemExit("No global fragments matched any automatically selected dataset candidate.")


def _resolve_output_prefix(args, target_info) -> str:
    if args.output_prefix:
        return args.output_prefix

    parts = [args.mode, args.model, args.importance]
    if args.mode == "qsar":
        parts.append(args.receptor)
    if target_info["dataset_index"] is None:
        parts.append(f"smiles_{_sanitize_label(target_info['smiles'])}")
    else:
        parts.append(f"dataset_{target_info['dataset_index']}")
    if int(args.draw_size) != int(DEFAULT_DRAW_SIZE):
        parts.append(f"size_{int(args.draw_size)}")
    return "_".join(_sanitize_label(part) for part in parts if part)


def _ensure_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        pieces = [REPO_ROOT / "common" / "outputs" / "global-ecfp", args.mode, args.model, args.importance]
        if args.mode == "qsar" and args.receptor:
            pieces.append(args.receptor)
        out_dir = Path(*pieces)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _print_section(title: str):
    print(f"\n== {title} ==")


def _print_key_value(label: str, value):
    print(f"{label}: {value}")


def _require_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise SystemExit("XGBoost backend requested, but the `xgboost` package is not available.") from exc
    return XGBClassifier, XGBRegressor


def _model_label(model_backend: str) -> str:
    return "XGBoost" if model_backend == "xgb" else "RandomForest"


def _model_cache_metadata(model_backend: str) -> dict:
    meta = {"model_backend": str(model_backend)}
    if model_backend == "xgb":
        meta.update(
            {
                "max_depth": int(XGB_MAX_DEPTH),
                "learning_rate": float(XGB_LEARNING_RATE),
                "subsample": float(XGB_SUBSAMPLE),
                "colsample_bytree": float(XGB_COLSAMPLE_BYTREE),
            }
        )
    return meta


def _make_classifier(model_backend: str, n_estimators: int, random_seed: int, n_jobs: int):
    if model_backend == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_seed,
            class_weight="balanced_subsample",
        )
    if model_backend == "xgb":
        XGBClassifier, _ = _require_xgboost()
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            objective="binary:logistic",
            eval_metric="logloss",
            importance_type="gain",
            n_jobs=n_jobs,
            random_state=random_seed,
            verbosity=0,
        )
    raise SystemExit(f"Unsupported model backend: {model_backend}")


def _make_regressor(model_backend: str, n_estimators: int, random_seed: int, n_jobs: int):
    if model_backend == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_seed,
        )
    if model_backend == "xgb":
        _, XGBRegressor = _require_xgboost()
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            objective="reg:squarederror",
            eval_metric="rmse",
            importance_type="gain",
            n_jobs=n_jobs,
            random_state=random_seed,
            verbosity=0,
        )
    raise SystemExit(f"Unsupported model backend: {model_backend}")


def _load_or_fit_binary_model(
    cache_dir: Path,
    mode: str,
    target_name: str,
    model_backend: str,
    data_sig,
    radius: int,
    n_bits: int,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    x: np.ndarray,
    y: np.ndarray,
):
    model_cache_path = cache_dir / f"{mode}_{target_name}_{model_backend}_model.pkl"
    model_cache_meta = {
        "version": CACHE_VERSION,
        "kind": "global_ecfp_classifier",
        "mode": mode,
        "target_name": str(target_name),
        "data": data_sig,
        "ecfp_radius": int(radius),
        "ecfp_n_bits": int(n_bits),
        "n_estimators": int(n_estimators),
        "random_seed": int(random_seed),
        "n_jobs": int(n_jobs),
    }
    model_cache_meta.update(_model_cache_metadata(model_backend))
    model = load_pickle_cache(model_cache_path, model_cache_meta)
    if model is None:
        model = _make_classifier(
            model_backend=model_backend,
            n_estimators=n_estimators,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        model.fit(x, y)
        save_pickle_cache(model_cache_path, model_cache_meta, model)
        cache_status = "miss"
    else:
        cache_status = "hit"
    return model, cache_status


def _load_or_compute_importance_scores(
    cache_dir: Path,
    mode: str,
    target_name: str,
    model_backend: str,
    importance: str,
    data_sig,
    radius: int,
    n_bits: int,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    shap_sample_size: int,
    boruta_n_trials: int,
    boruta_sample: bool,
    boruta_normalize: bool,
    boruta_train_or_test: str,
    boruta_rf_n_estimators: int,
    boruta_rf_max_depth: int,
    x: np.ndarray,
    y: np.ndarray,
    model,
    classification: bool = True,
):
    importance_cache_path = cache_dir / f"{mode}_{target_name}_{model_backend}_{importance}_bit_scores.pkl"
    importance_cache_meta = {
        "version": CACHE_VERSION,
        "kind": "global_ecfp_bit_scores",
        "mode": mode,
        "target_name": str(target_name),
        "importance": importance,
        "data": data_sig,
        "ecfp_radius": int(radius),
        "ecfp_n_bits": int(n_bits),
        "n_estimators": int(n_estimators),
        "random_seed": int(random_seed),
        "n_jobs": int(n_jobs),
        "shap_sample_size": int(shap_sample_size),
        "boruta_n_trials": int(boruta_n_trials),
        "boruta_sample": bool(boruta_sample),
        "boruta_normalize": bool(boruta_normalize),
        "boruta_train_or_test": str(boruta_train_or_test),
        "boruta_rf_n_estimators": int(boruta_rf_n_estimators),
        "boruta_rf_max_depth": int(boruta_rf_max_depth),
        "shap_feature_matrix_version": int(SHAP_FEATURE_MATRIX_VERSION),
    }
    importance_cache_meta.update(_model_cache_metadata(model_backend))
    cached_importance = load_pickle_cache(importance_cache_path, importance_cache_meta)
    if cached_importance is None:
        if importance == "shap":
            bit_scores, importance_meta = _compute_shap_bit_scores(
                model,
                x,
                sample_size=int(shap_sample_size),
                random_seed=random_seed,
            )
        else:
            bit_scores, importance_meta = _compute_borutashap_bit_scores(
                x=x,
                y=y,
                n_bits=n_bits,
                model_backend=model_backend,
                classification=classification,
                random_seed=random_seed,
                n_jobs=n_jobs,
                boruta_n_trials=boruta_n_trials,
                boruta_sample=boruta_sample,
                boruta_normalize=boruta_normalize,
                boruta_train_or_test=boruta_train_or_test,
                boruta_rf_n_estimators=boruta_rf_n_estimators,
                boruta_rf_max_depth=boruta_rf_max_depth,
            )
        save_pickle_cache(
            importance_cache_path,
            importance_cache_meta,
            {"bit_scores": bit_scores, "importance_meta": importance_meta},
        )
        importance_cache_status = "miss"
    else:
        bit_scores = np.asarray(cached_importance["bit_scores"], dtype=np.float32)
        importance_meta = dict(cached_importance["importance_meta"])
        importance_cache_status = "hit"
    return bit_scores, importance_meta, importance_cache_status


def _load_backend_importance_bundle(
    cache_dir: Path,
    mode: str,
    target_name: str,
    model_backend: str,
    importance: str,
    data_sig,
    radius: int,
    n_bits: int,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    shap_sample_size: int,
    boruta_n_trials: int,
    boruta_sample: bool,
    boruta_normalize: bool,
    boruta_train_or_test: str,
    boruta_rf_n_estimators: int,
    boruta_rf_max_depth: int,
    x: np.ndarray,
    y: np.ndarray,
    classification: bool = True,
):
    model, model_cache_status = _load_or_fit_binary_model(
        cache_dir=cache_dir,
        mode=mode,
        target_name=target_name,
        model_backend=model_backend,
        data_sig=data_sig,
        radius=radius,
        n_bits=n_bits,
        n_estimators=n_estimators,
        random_seed=random_seed,
        n_jobs=n_jobs,
        x=x,
        y=y,
    )
    bit_scores, importance_meta, importance_cache_status = _load_or_compute_importance_scores(
        cache_dir=cache_dir,
        mode=mode,
        target_name=target_name,
        model_backend=model_backend,
        importance=importance,
        data_sig=data_sig,
        radius=radius,
        n_bits=n_bits,
        n_estimators=n_estimators,
        random_seed=random_seed,
        n_jobs=n_jobs,
        shap_sample_size=shap_sample_size,
        boruta_n_trials=boruta_n_trials,
        boruta_sample=boruta_sample,
        boruta_normalize=boruta_normalize,
        boruta_train_or_test=boruta_train_or_test,
        boruta_rf_n_estimators=boruta_rf_n_estimators,
        boruta_rf_max_depth=boruta_rf_max_depth,
        x=x,
        y=y,
        model=model,
        classification=classification,
    )
    return {
        "model": model,
        "model_cache_status": model_cache_status,
        "bit_scores": bit_scores,
        "importance_meta": importance_meta,
        "importance_cache_status": importance_cache_status,
    }


def _build_qsar_receptor_probability_table(
    problem,
    selected_receptor: str,
    selected_model,
    model_backend: str,
    cache_dir: Path,
    data_sig,
    radius: int,
    n_bits: int,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    target_vector: np.ndarray,
):
    rows = []
    for receptor_name in problem["receptor_names"]:
        if receptor_name == selected_receptor:
            model = selected_model
            cache_status = "active"
        else:
            y_receptor = problem["df"]["receptors"].apply(
                lambda values: int(receptor_name in values)
            ).to_numpy(dtype=np.int8)
            model, cache_status = _load_or_fit_binary_model(
                cache_dir=cache_dir,
                mode="qsar",
                target_name=receptor_name,
                model_backend=model_backend,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                x=problem["x"],
                y=y_receptor,
            )

        probability = float(model.predict_proba(target_vector.reshape(1, -1))[0, 1])
        rows.append(
            {
                "receptor": receptor_name,
                "receptor_czech_name": _get_receptor_czech_name(receptor_name),
                "probability": probability,
                "confidence": float(max(probability, 1.0 - probability)),
                "is_selected_receptor": int(receptor_name == selected_receptor),
                "model_cache_status": cache_status,
            }
        )

    receptor_df = pd.DataFrame(rows).sort_values(["probability", "receptor"], ascending=[False, True])
    receptor_df.reset_index(drop=True, inplace=True)
    receptor_df.insert(0, "rank", np.arange(1, len(receptor_df) + 1))
    return receptor_df


def _attach_prediction_outputs(mode: str, target_info: dict, value_model=None):
    target_info = dict(target_info)
    predicted_probability = float(target_info["predicted_probability"])
    target_info["predicted_class"] = int(predicted_probability >= 0.5)
    target_info["prediction_confidence"] = float(max(predicted_probability, 1.0 - predicted_probability))
    if mode == "qspr":
        target_info["predicted_value"] = float(value_model.predict(target_info["vector"].reshape(1, -1))[0])
    else:
        target_info["predicted_value"] = predicted_probability
        target_info["receptor_name"] = target_info.get("receptor_name", "")
        target_info["receptor_probability"] = predicted_probability
    return target_info


def _select_exclusive_model_delta_bits(
    rf_scores: np.ndarray,
    xgb_scores: np.ndarray,
    top_n_bits: int,
) -> tuple[np.ndarray, set[int], set[int], set[int]]:
    rf_top_bits = {int(bit) for bit in _select_top_bits(rf_scores, top_n_bits=top_n_bits)}
    xgb_top_bits = {int(bit) for bit in _select_top_bits(xgb_scores, top_n_bits=top_n_bits)}
    common_bits = rf_top_bits & xgb_top_bits
    exclusive_bits = list(rf_top_bits ^ xgb_top_bits)
    if not exclusive_bits:
        raise SystemExit("RF and XGBoost top-N bit sets are identical; there are no exclusive bits to compare.")

    delta_scores = np.asarray(xgb_scores, dtype=np.float32) - np.asarray(rf_scores, dtype=np.float32)
    exclusive_bits.sort(key=lambda bit: abs(float(delta_scores[int(bit)])), reverse=True)
    selected_bits = np.asarray(exclusive_bits, dtype=np.int32)
    return selected_bits, rf_top_bits, xgb_top_bits, common_bits


def _build_delta_metrics_table(
    mode: str,
    primary_target_info: dict,
    comparison_target_info: dict,
    primary_atom_df: pd.DataFrame,
    comparison_atom_df: pd.DataFrame,
    primary_matched_df: pd.DataFrame,
    comparison_matched_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if mode == "qspr":
        rows.extend(
            [
                {
                    "metric": "predicted_probability",
                    "primary": float(primary_target_info["predicted_probability"]),
                    "comparison": float(comparison_target_info["predicted_probability"]),
                },
                {
                    "metric": "predicted_value",
                    "primary": float(primary_target_info["predicted_value"]),
                    "comparison": float(comparison_target_info["predicted_value"]),
                },
                {
                    "metric": "prediction_confidence",
                    "primary": float(primary_target_info["prediction_confidence"]),
                    "comparison": float(comparison_target_info["prediction_confidence"]),
                },
            ]
        )
    else:
        rows.extend(
            [
                {
                    "metric": "selected_receptor_probability",
                    "primary": float(primary_target_info["receptor_probability"]),
                    "comparison": float(comparison_target_info["receptor_probability"]),
                },
                {
                    "metric": "prediction_confidence",
                    "primary": float(primary_target_info["prediction_confidence"]),
                    "comparison": float(comparison_target_info["prediction_confidence"]),
                },
            ]
        )

    rows.extend(
        [
            {
                "metric": "matched_fragment_count",
                "primary": float(len(primary_matched_df)),
                "comparison": float(len(comparison_matched_df)),
            },
            {
                "metric": "matched_atom_count",
                "primary": float(np.count_nonzero(primary_atom_df["hit_count"].to_numpy() > 0)),
                "comparison": float(np.count_nonzero(comparison_atom_df["hit_count"].to_numpy() > 0)),
            },
            {
                "metric": "atom_count",
                "primary": float(len(primary_atom_df)),
                "comparison": float(len(comparison_atom_df)),
            },
        ]
    )

    delta_df = pd.DataFrame(rows)
    delta_df["delta"] = delta_df["comparison"] - delta_df["primary"]
    delta_df.insert(0, "comparison_smiles", str(comparison_target_info["smiles"]))
    delta_df.insert(0, "primary_smiles", str(primary_target_info["smiles"]))
    return delta_df


def _build_qsar_receptor_delta_table(
    primary_receptor_df: pd.DataFrame,
    comparison_receptor_df: pd.DataFrame,
) -> pd.DataFrame:
    primary_df = primary_receptor_df.rename(
        columns={
            "probability": "primary_probability",
            "confidence": "primary_confidence",
            "is_selected_receptor": "primary_is_selected_receptor",
            "model_cache_status": "primary_model_cache_status",
        }
    )
    comparison_df = comparison_receptor_df.rename(
        columns={
            "probability": "comparison_probability",
            "confidence": "comparison_confidence",
            "is_selected_receptor": "comparison_is_selected_receptor",
            "model_cache_status": "comparison_model_cache_status",
        }
    )
    delta_df = primary_df.merge(
        comparison_df,
        on=["receptor", "receptor_czech_name"],
        how="inner",
    )
    delta_df["probability_delta"] = delta_df["comparison_probability"] - delta_df["primary_probability"]
    delta_df["confidence_delta"] = delta_df["comparison_confidence"] - delta_df["primary_confidence"]
    delta_df["abs_probability_delta"] = delta_df["probability_delta"].abs()
    delta_df = delta_df.sort_values(["abs_probability_delta", "receptor"], ascending=[False, True]).reset_index(drop=True)
    delta_df.insert(0, "rank", np.arange(1, len(delta_df) + 1))
    return delta_df


def _select_top_bits(bit_scores: np.ndarray, top_n_bits: int) -> np.ndarray:
    selected_bits = np.argsort(bit_scores)[::-1]
    selected_bits = selected_bits[bit_scores[selected_bits] > 0][:top_n_bits]
    if len(selected_bits) == 0:
        raise SystemExit("No positive bit importances were found.")
    return selected_bits


def _importance_axis_label(importance: str, importance_meta: dict) -> str:
    if importance == "shap" or importance_meta.get("score_source") == "mean_abs_shap":
        return "mean(|SHAP value|)"
    return "importance"


def _save_model_delta_chart(model_delta_df: pd.DataFrame, png_path: Path, svg_path: Path):
    import matplotlib.pyplot as plt

    apply_plot_style()

    plot_df = model_delta_df.sort_values("delta", ascending=True).reset_index(drop=True)
    colors = np.where(plot_df["delta"].to_numpy() >= 0, "#8E44AD", "#D2B4DE")
    fig_height = max(4.5, 0.42 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.barh(range(len(plot_df)), plot_df["delta"].to_numpy(), color=colors, alpha=0.92)
    ax.axvline(0.0, color="#5E3370", linewidth=1.0, alpha=0.9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f"bit {int(bit)}" for bit in plot_df["bit"].to_numpy()])
    ax.set_xlabel("importance delta (XGBoost - RandomForest)")
    ax.set_title(f"RF vs XGBoost delta (exclusive top-N bits, n={len(plot_df)})")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _save_top_n_chart(
    bits_df: pd.DataFrame,
    importance: str,
    importance_meta: dict,
    chart_title: str,
    png_path: Path,
    svg_path: Path,
):
    import matplotlib.pyplot as plt

    apply_plot_style()

    top_df = bits_df.copy()
    top_df["bit_label"] = top_df["bit"].map(lambda bit: f"bit {int(bit)}")
    plot_df = top_df.iloc[::-1].reset_index(drop=True)

    fig_height = max(4.5, 0.42 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    ax.barh(range(len(plot_df)), plot_df["importance"].to_numpy(), color="#4C78A8", alpha=0.9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["bit_label"].tolist())
    ax.set_xlabel(_importance_axis_label(importance, importance_meta))
    ax.set_title(chart_title)
    fig.tight_layout()

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _render_similarity_map_image(mol, weights: np.ndarray, draw_size: int) -> Image.Image:
    drawer = Draw.MolDraw2DCairo(draw_size, draw_size)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(map(float, weights)), draw2d=drawer)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _render_plain_molecule_image(mol, draw_size: int) -> Image.Image:
    image = Draw.MolToImage(mol, size=(draw_size, draw_size))
    return image.convert("RGBA")


def _get_receptor_czech_name(receptor_name: str) -> str:
    return RECEPTOR_CZECH_NAMES.get(str(receptor_name).lower(), str(receptor_name).upper())


def _load_panel_font(size: int, bold: bool = False):
    cache_key = (int(size), bool(bold))
    cached_font = _PANEL_FONT_CACHE.get(cache_key)
    if cached_font is not None:
        return cached_font

    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates: list[Path] = []
    try:
        import matplotlib

        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / font_name)
    except Exception:
        pass
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu") / font_name,
            Path("/Library/Fonts") / font_name,
            Path("/System/Library/Fonts/Supplemental") / "Arial.ttf",
            Path("/System/Library/Fonts/Supplemental") / "Arial Unicode.ttf",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            font = ImageFont.truetype(str(candidate), size=int(size))
            _PANEL_FONT_CACHE[cache_key] = font
            return font
        except OSError:
            continue

    font = ImageFont.load_default()
    _PANEL_FONT_CACHE[cache_key] = font
    return font


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_panel_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        candidate_width, _ = _measure_text(draw, candidate_text, font)
        if current_words and candidate_width > max_width:
            lines.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        lines.append(" ".join(current_words))
    return lines


def _fit_panel_subtitle(draw: ImageDraw.ImageDraw, text: str, width: int):
    for size in range(max(18, int(round(width * 0.05))), 11, -1):
        font = _load_panel_font(size, bold=False)
        lines = _wrap_panel_text(draw, text, font, max_width=width)
        if not lines:
            continue
        if len(lines) > 2:
            continue
        if max(_measure_text(draw, line, font)[0] for line in lines) <= width:
            return font, lines

    fallback_font = _load_panel_font(12, bold=False)
    fallback_lines = _wrap_panel_text(draw, text, fallback_font, max_width=width)
    return fallback_font, fallback_lines[:2]


def _build_receptor_panel_image(
    base_image: Image.Image,
    receptor_name: str,
) -> Image.Image:
    width, height = base_image.size
    czech_name = _get_receptor_czech_name(receptor_name)
    side_padding = max(14, int(round(width * 0.04)))
    top_padding = max(10, int(round(width * 0.018)))
    subtitle_line_gap = max(2, int(round(width * 0.006)))

    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    subtitle_font, subtitle_lines = _fit_panel_subtitle(
        probe_draw,
        czech_name,
        width=max(40, width - side_padding * 2),
    )
    _, subtitle_line_height = _measure_text(probe_draw, "Ag", subtitle_font)
    subtitle_total_height = subtitle_line_height * len(subtitle_lines)
    subtitle_total_height += subtitle_line_gap * max(0, len(subtitle_lines) - 1)
    header_height = top_padding * 2 + subtitle_total_height

    panel = Image.new("RGBA", (width, height + header_height), (255, 255, 255, 255))
    panel.paste(base_image, (0, header_height))

    draw = ImageDraw.Draw(panel)
    subtitle_y = top_padding
    for line in subtitle_lines:
        draw.text(
            (side_padding, subtitle_y),
            line,
            fill=(32, 32, 32),
            font=subtitle_font,
        )
        subtitle_y += subtitle_line_height + subtitle_line_gap
    return panel


def _build_panel_grid_image(panels: list[Image.Image], cols: int, rows: int) -> Image.Image:
    if len(panels) > cols * rows:
        raise SystemExit(
            f"--receptor-grid {cols}x{rows} is too small for {len(panels)} panels."
        )

    if not panels:
        raise SystemExit("No panels were available for the receptor grid.")

    panel_width, panel_height = panels[0].size
    gap = 18
    canvas_width = cols * panel_width + (cols - 1) * gap
    canvas_height = rows * panel_height + (rows - 1) * gap
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * (panel_width + gap)
        y = row * (panel_height + gap)
        canvas.paste(panel, (x, y))

    return canvas


def _build_qsar_receptor_grid(
    problem,
    selected_receptor: str,
    selected_model,
    model_backend: str,
    importance: str,
    cache_dir: Path,
    data_sig,
    radius: int,
    n_bits: int,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    shap_sample_size: int,
    boruta_n_trials: int,
    boruta_sample: bool,
    boruta_normalize: bool,
    boruta_train_or_test: str,
    boruta_rf_n_estimators: int,
    boruta_rf_max_depth: int,
    target_info: dict,
    draw_size: int,
    atom_aggregation: str,
    top_n_bits: int,
    grid_cols: int,
    grid_rows: int,
    receptor_grid_top_n: int,
):
    panels = []
    rows = []
    total = len(problem["receptor_names"])
    grid_bit_limit = top_n_bits if receptor_grid_top_n <= 0 else int(receptor_grid_top_n)

    for idx, receptor_name in enumerate(problem["receptor_names"], start=1):
        y_receptor = problem["df"]["receptors"].apply(
            lambda values: int(receptor_name in values)
        ).to_numpy(dtype=np.int8)
        if receptor_name == selected_receptor:
            model = selected_model
            model_cache_status = "active"
        else:
            model, model_cache_status = _load_or_fit_binary_model(
                cache_dir=cache_dir,
                mode="qsar",
                target_name=receptor_name,
                model_backend=model_backend,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                x=problem["x"],
                y=y_receptor,
            )

        bit_scores, importance_meta, importance_cache_status = _load_or_compute_importance_scores(
            cache_dir=cache_dir,
            mode="qsar",
            target_name=receptor_name,
            model_backend=model_backend,
            importance=importance,
            data_sig=data_sig,
            radius=radius,
            n_bits=n_bits,
            n_estimators=n_estimators,
            random_seed=random_seed,
            n_jobs=n_jobs,
            shap_sample_size=shap_sample_size,
            boruta_n_trials=boruta_n_trials,
            boruta_sample=boruta_sample,
            boruta_normalize=boruta_normalize,
            boruta_train_or_test=boruta_train_or_test,
            boruta_rf_n_estimators=boruta_rf_n_estimators,
            boruta_rf_max_depth=boruta_rf_max_depth,
            x=problem["x"],
            y=y_receptor,
            model=model,
            classification=True,
        )
        selected_bits = _select_top_bits(bit_scores, top_n_bits=grid_bit_limit)
        fragment_df = _build_global_fragment_table(
            mols=problem["mols"],
            bit_info_maps=problem["bit_info_maps"],
            bit_scores=bit_scores,
            selected_bits=selected_bits,
        )
        probability = float(model.predict_proba(target_info["vector"].reshape(1, -1))[0, 1])
        confidence = float(max(probability, 1.0 - probability))
        atom_weights, atom_df, matched_df = _match_fragments_to_molecule(
            target_info["mol"],
            fragment_df,
            aggregation=atom_aggregation,
        )

        if np.any(np.abs(atom_weights) > 0):
            base_image = _render_similarity_map_image(target_info["mol"], atom_weights, draw_size=draw_size)
            render_mode = "similarity_map"
        else:
            base_image = _render_plain_molecule_image(target_info["mol"], draw_size=draw_size)
            render_mode = "plain_molecule"

        panel = _build_receptor_panel_image(
            base_image=base_image,
            receptor_name=receptor_name,
        )
        panels.append(panel)
        rows.append(
            {
                "receptor": receptor_name,
                "receptor_czech_name": _get_receptor_czech_name(receptor_name),
                "probability": probability,
                "confidence": confidence,
                "is_selected_receptor": int(receptor_name == selected_receptor),
                "model_cache_status": model_cache_status,
                "importance_cache_status": importance_cache_status,
                "importance_source": importance_meta.get("score_source", ""),
                "importance_scope": importance_meta.get("scope", ""),
                "matched_fragment_count": int(len(matched_df)),
                "matched_atom_count": int(np.count_nonzero(atom_df["hit_count"].to_numpy() > 0)),
                "panel_render_mode": render_mode,
                "selected_bits": ";".join(str(bit) for bit in selected_bits),
            }
        )
        print(
            f"[{idx}/{total}] receptor={receptor_name} "
            f"({_get_receptor_czech_name(receptor_name)}) "
            f"p={probability:.4f} conf={confidence:.4f} "
            f"model_cache={model_cache_status} importance_cache={importance_cache_status}"
        )

    grid_image = _build_panel_grid_image(panels, cols=grid_cols, rows=grid_rows)
    receptor_df = pd.DataFrame(rows).sort_values(["receptor"], ascending=[True]).reset_index(drop=True)
    receptor_df.insert(0, "rank", np.arange(1, len(receptor_df) + 1))
    return receptor_df, grid_image


def main():
    args = _build_arg_parser().parse_args()
    defaults = MODE_DEFAULTS[args.mode]
    receptor_grid_shape = None

    if args.mode == "qsar" and not args.receptor:
        raise SystemExit("--receptor is required when --mode qsar.")
    if args.mode == "qspr" and args.receptor:
        raise SystemExit("--receptor is only valid with --mode qsar.")
    if args.mode == "qspr" and args.smiles is None:
        raise SystemExit("--smiles is required when --mode qspr.")
    if args.mode == "qspr" and args.dataset_index is not None:
        raise SystemExit("--dataset-index is not supported when --mode qspr; use --smiles.")
    if args.mode == "qsar" and args.smiles is None:
        raise SystemExit("--smiles is required when --mode qsar.")
    if args.mode == "qsar" and args.dataset_index is not None:
        raise SystemExit("--dataset-index is not supported when --mode qsar; use --smiles.")
    if args.receptor_grid is not None:
        if args.mode != "qsar":
            raise SystemExit("--receptor-grid is only supported when --mode qsar.")
        receptor_grid_shape = _parse_grid_shape(args.receptor_grid)
    if args.delta_smiles is not None and receptor_grid_shape is not None:
        raise SystemExit("--delta-smiles cannot be combined with --receptor-grid.")
    if args.delta_smiles is not None and args.model_delta:
        raise SystemExit("--delta-smiles cannot be combined with --model-delta.")

    radius = defaults["radius"] if args.radius is None else int(args.radius)
    n_bits = defaults["n_bits"] if args.n_bits is None else int(args.n_bits)
    n_estimators = defaults["n_estimators"] if args.n_estimators is None else int(args.n_estimators)
    n_jobs = defaults["n_jobs"] if args.n_jobs is None else int(args.n_jobs)
    random_seed = defaults["random_seed"] if args.random_seed is None else int(args.random_seed)
    top_n_bits = defaults["top_n_bits"] if args.top_n_bits is None else int(args.top_n_bits)
    receptor_grid_top_n = 0 if args.receptor_grid_top_n is None else int(args.receptor_grid_top_n)

    if top_n_bits <= 0:
        raise SystemExit("--top-n-bits must be > 0.")
    if receptor_grid_top_n < 0:
        raise SystemExit("--receptor-grid-top-n must be >= 0.")
    if receptor_grid_top_n > 0 and receptor_grid_shape is None:
        raise SystemExit("--receptor-grid-top-n requires --receptor-grid.")

    boruta_n_trials = defaults["boruta_n_trials"] if args.boruta_trials is None else int(args.boruta_trials)
    boruta_sample = defaults["boruta_sample"] if args.boruta_sample is None else bool(args.boruta_sample)
    boruta_normalize = defaults["boruta_normalize"] if args.boruta_normalize is None else bool(args.boruta_normalize)
    boruta_train_or_test = defaults["boruta_train_or_test"]
    boruta_rf_n_estimators = defaults["boruta_rf_n_estimators"]
    boruta_rf_max_depth = defaults["boruta_rf_max_depth"]

    mpl_dir = REPO_ROOT / ".cache" / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    RDLogger.DisableLog("rdApp.warning")

    data_path = Path(defaults["data_path"] if args.data_path is None else args.data_path).resolve()
    _print_section("Setup")
    _print_key_value("Mode", args.mode)
    _print_key_value("Model backend", f"{args.model} ({_model_label(args.model)})")
    _print_key_value("Importance", args.importance)
    if args.mode == "qsar":
        _print_key_value("Receptor", args.receptor)
    if args.delta_smiles is not None:
        _print_key_value("Delta SMILES", args.delta_smiles)
    if args.model_delta:
        _print_key_value("Model delta", "rf vs xgb")
    _print_key_value("Data path", data_path)
    _print_key_value("Radius", radius)
    _print_key_value("N bits", n_bits)
    _print_key_value("N estimators", n_estimators)
    _print_key_value("Top N bits", top_n_bits)
    if receptor_grid_shape is not None:
        _print_key_value("Receptor grid", f"{receptor_grid_shape[0]}x{receptor_grid_shape[1]}")
        _print_key_value("Grid top-N bits", receptor_grid_top_n if receptor_grid_top_n > 0 else "off")
    _print_key_value("Top-N chart", args.top_n_chart)

    _print_section("Load Data")
    if args.mode == "qspr":
        problem = _load_qspr_problem(data_path=data_path, radius=radius, n_bits=n_bits)
        _print_key_value("Loaded molecules", len(problem["df"]))
        _print_key_value("Binary cutoff", f"{problem['cutoff']:.5f}")
    else:
        problem = _load_qsar_problem(
            data_path=data_path,
            radius=radius,
            n_bits=n_bits,
            receptor=str(args.receptor),
        )
        _print_key_value("Aggregated molecules", len(problem["df"]))
        _print_key_value("Positive rows", problem["positive_count"])
        _print_key_value("Receptor count", len(problem["receptor_names"]))
        if receptor_grid_shape is not None and len(problem["receptor_names"]) > receptor_grid_shape[0] * receptor_grid_shape[1]:
            raise SystemExit(
                f"--receptor-grid {receptor_grid_shape[0]}x{receptor_grid_shape[1]} is too small for "
                f"{len(problem['receptor_names'])} receptors."
            )

    x = problem["x"]
    y = problem["y"]
    classification = True

    cache_dir = REPO_ROOT / "common" / "cache" / "global-ecfp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_sig = file_signature(data_path)

    _print_section("Train Or Load Classifier")
    model, model_cache_status = _load_or_fit_binary_model(
        cache_dir=cache_dir,
        mode=args.mode,
        target_name=str(problem["target_name"]),
        model_backend=args.model,
        data_sig=data_sig,
        radius=radius,
        n_bits=n_bits,
        n_estimators=n_estimators,
        random_seed=random_seed,
        n_jobs=n_jobs,
        x=x,
        y=y,
    )
    _print_key_value("Classifier cache", model_cache_status)

    value_model = None
    value_model_cache_status = ""
    if args.mode == "qspr":
        _print_section("Train Or Load Regressor")
        value_model_cache_path = cache_dir / f"qspr_solubility_{args.model}_regressor.pkl"
        value_model_cache_meta = {
            "version": CACHE_VERSION,
            "kind": "global_ecfp_regressor",
            "mode": args.mode,
            "target_name": "solubility",
            "data": data_sig,
            "ecfp_radius": int(radius),
            "ecfp_n_bits": int(n_bits),
            "n_estimators": int(n_estimators),
            "random_seed": int(random_seed),
            "n_jobs": int(n_jobs),
        }
        value_model_cache_meta.update(_model_cache_metadata(args.model))
        value_model = load_pickle_cache(value_model_cache_path, value_model_cache_meta)
        if value_model is None:
            value_model = _make_regressor(
                model_backend=args.model,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
            )
            value_model.fit(x, problem["y_value"])
            save_pickle_cache(value_model_cache_path, value_model_cache_meta, value_model)
            value_model_cache_status = "miss"
        else:
            value_model_cache_status = "hit"
        _print_key_value("Regressor cache", value_model_cache_status)

    _print_section("Global Importance")
    if args.importance == "shap":
        _print_key_value("SHAP sample size", int(args.shap_sample_size))
    else:
        _print_key_value("Boruta trials", boruta_n_trials)
    bit_scores, importance_meta, importance_cache_status = _load_or_compute_importance_scores(
        cache_dir=cache_dir,
        mode=args.mode,
        target_name=str(problem["target_name"]),
        model_backend=args.model,
        importance=args.importance,
        data_sig=data_sig,
        radius=radius,
        n_bits=n_bits,
        n_estimators=n_estimators,
        random_seed=random_seed,
        n_jobs=n_jobs,
        shap_sample_size=int(args.shap_sample_size),
        boruta_n_trials=boruta_n_trials,
        boruta_sample=boruta_sample,
        boruta_normalize=boruta_normalize,
        boruta_train_or_test=boruta_train_or_test,
        boruta_rf_n_estimators=boruta_rf_n_estimators,
        boruta_rf_max_depth=boruta_rf_max_depth,
        x=x,
        y=y,
        model=model,
        classification=classification,
    )
    _print_key_value("Importance cache", importance_cache_status)
    _print_key_value("Importance source", importance_meta.get("score_source", ""))
    _print_key_value("Importance scope", importance_meta.get("scope", ""))

    selected_bits = _select_top_bits(bit_scores, top_n_bits=top_n_bits)
    _print_key_value("Selected bits", ", ".join(str(bit) for bit in selected_bits[: min(10, len(selected_bits))]))

    bits_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_bits) + 1),
            "bit": selected_bits.astype(int),
            "importance": bit_scores[selected_bits].astype(np.float32),
        }
    )
    bits_df["relative_importance"] = bits_df["importance"] / float(bits_df["importance"].sum())
    bits_df["activation_rate"] = x[:, bits_df["bit"].astype(int)].mean(axis=0)

    fragment_df = _build_global_fragment_table(
        mols=problem["mols"],
        bit_info_maps=problem["bit_info_maps"],
        bit_scores=bit_scores,
        selected_bits=selected_bits,
    )
    _print_key_value("Global fragments", len(fragment_df))

    _print_section("Select Molecule")
    target_info, atom_weights, atom_df, matched_df = _select_target_with_fallback(
        problem,
        model,
        args,
        mode=args.mode,
        radius=radius,
        n_bits=n_bits,
        fragment_df=fragment_df,
    )
    target_info = _attach_prediction_outputs(args.mode, target_info, value_model=value_model)
    _print_key_value("Selection mode", target_info["selection_mode"])
    _print_key_value("Selected SMILES", target_info["smiles"])
    if args.mode == "qspr":
        _print_key_value("Predicted probability", f"{target_info['predicted_probability']:.4f}")
        _print_key_value("Predicted value", f"{target_info['predicted_value']:.4f}")
        _print_key_value("Prediction confidence", f"{target_info['prediction_confidence']:.4f}")
    else:
        _print_key_value(
            f"Selected receptor probability [{target_info['receptor_name']}]",
            f"{target_info['receptor_probability']:.4f}",
        )
        _print_key_value("Prediction confidence", f"{target_info['prediction_confidence']:.4f}")

    comparison_target_info = None
    comparison_atom_weights = None
    comparison_atom_df = None
    comparison_matched_df = None
    delta_metrics_df = None
    receptor_delta_df = None
    model_delta_bits_df = None
    model_delta_fragment_df = None
    model_delta_atom_weights = None
    model_delta_atom_df = None
    model_delta_matched_df = None
    model_delta_rf_bundle = None
    model_delta_xgb_bundle = None
    if args.delta_smiles is not None:
        _print_section("Delta Molecule")
        comparison_args = argparse.Namespace(
            smiles=str(args.delta_smiles),
            dataset_index=None,
            atom_aggregation=args.atom_aggregation,
        )
        comparison_target_info, comparison_atom_weights, comparison_atom_df, comparison_matched_df = _select_target_with_fallback(
            problem,
            model,
            comparison_args,
            mode=args.mode,
            radius=radius,
            n_bits=n_bits,
            fragment_df=fragment_df,
        )
        comparison_target_info = _attach_prediction_outputs(
            args.mode,
            comparison_target_info,
            value_model=value_model,
        )
        _print_key_value("Comparison SMILES", comparison_target_info["smiles"])
        if args.mode == "qspr":
            _print_key_value("Comparison probability", f"{comparison_target_info['predicted_probability']:.4f}")
            _print_key_value("Comparison value", f"{comparison_target_info['predicted_value']:.4f}")
        else:
            _print_key_value(
                f"Comparison receptor probability [{comparison_target_info['receptor_name']}]",
                f"{comparison_target_info['receptor_probability']:.4f}",
            )
        delta_metrics_df = _build_delta_metrics_table(
            mode=args.mode,
            primary_target_info=target_info,
            comparison_target_info=comparison_target_info,
            primary_atom_df=atom_df,
            comparison_atom_df=comparison_atom_df,
            primary_matched_df=matched_df,
            comparison_matched_df=comparison_matched_df,
        )
        _print_section("Delta Summary")
        print(delta_metrics_df[["metric", "primary", "comparison", "delta"]].to_string(index=False))

    if args.model_delta:
        _print_section("RF vs XGBoost Delta")
        if args.model == "rf":
            model_delta_rf_bundle = {
                "model": model,
                "model_cache_status": "active",
                "bit_scores": bit_scores,
                "importance_meta": importance_meta,
                "importance_cache_status": importance_cache_status,
            }
        else:
            model_delta_rf_bundle = _load_backend_importance_bundle(
                cache_dir=cache_dir,
                mode=args.mode,
                target_name=str(problem["target_name"]),
                model_backend="rf",
                importance=args.importance,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                shap_sample_size=int(args.shap_sample_size),
                boruta_n_trials=boruta_n_trials,
                boruta_sample=boruta_sample,
                boruta_normalize=boruta_normalize,
                boruta_train_or_test=boruta_train_or_test,
                boruta_rf_n_estimators=boruta_rf_n_estimators,
                boruta_rf_max_depth=boruta_rf_max_depth,
                x=x,
                y=y,
                classification=classification,
            )

        if args.model == "xgb":
            model_delta_xgb_bundle = {
                "model": model,
                "model_cache_status": "active",
                "bit_scores": bit_scores,
                "importance_meta": importance_meta,
                "importance_cache_status": importance_cache_status,
            }
        else:
            model_delta_xgb_bundle = _load_backend_importance_bundle(
                cache_dir=cache_dir,
                mode=args.mode,
                target_name=str(problem["target_name"]),
                model_backend="xgb",
                importance=args.importance,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                shap_sample_size=int(args.shap_sample_size),
                boruta_n_trials=boruta_n_trials,
                boruta_sample=boruta_sample,
                boruta_normalize=boruta_normalize,
                boruta_train_or_test=boruta_train_or_test,
                boruta_rf_n_estimators=boruta_rf_n_estimators,
                boruta_rf_max_depth=boruta_rf_max_depth,
                x=x,
                y=y,
                classification=classification,
            )

        rf_scores = np.asarray(model_delta_rf_bundle["bit_scores"], dtype=np.float32)
        xgb_scores = np.asarray(model_delta_xgb_bundle["bit_scores"], dtype=np.float32)
        delta_scores = xgb_scores - rf_scores
        delta_bits, rf_top_bits, xgb_top_bits, common_top_bits = _select_exclusive_model_delta_bits(
            rf_scores=rf_scores,
            xgb_scores=xgb_scores,
            top_n_bits=top_n_bits,
        )
        rf_rank_order = np.argsort(rf_scores)[::-1]
        xgb_rank_order = np.argsort(xgb_scores)[::-1]
        rf_rank_map = {int(bit): idx + 1 for idx, bit in enumerate(rf_rank_order)}
        xgb_rank_map = {int(bit): idx + 1 for idx, bit in enumerate(xgb_rank_order)}
        model_delta_bits_df = pd.DataFrame(
            {
                "rank": np.arange(1, len(delta_bits) + 1),
                "bit": delta_bits.astype(int),
                "rf_importance": rf_scores[delta_bits].astype(np.float32),
                "xgb_importance": xgb_scores[delta_bits].astype(np.float32),
                "delta": delta_scores[delta_bits].astype(np.float32),
                "abs_delta": np.abs(delta_scores[delta_bits]).astype(np.float32),
                "rf_rank": [rf_rank_map[int(bit)] for bit in delta_bits],
                "xgb_rank": [xgb_rank_map[int(bit)] for bit in delta_bits],
                "is_rf_top_n": [int(int(bit) in rf_top_bits) for bit in delta_bits],
                "is_xgb_top_n": [int(int(bit) in xgb_top_bits) for bit in delta_bits],
            }
        )
        model_delta_fragment_df = _build_global_fragment_table(
            mols=problem["mols"],
            bit_info_maps=problem["bit_info_maps"],
            bit_scores=delta_scores,
            selected_bits=delta_bits,
            allow_negative=True,
        )
        model_delta_atom_weights, model_delta_atom_df, model_delta_matched_df = _match_fragments_to_molecule(
            target_info["mol"],
            model_delta_fragment_df,
            aggregation=args.atom_aggregation,
        )
        _print_key_value("RF importance cache", model_delta_rf_bundle["importance_cache_status"])
        _print_key_value("XGB importance cache", model_delta_xgb_bundle["importance_cache_status"])
        _print_key_value("Shared top-N bits removed", len(common_top_bits))
        _print_key_value("Exclusive delta bits", len(delta_bits))
        print(
            model_delta_bits_df[
                ["rank", "bit", "rf_importance", "xgb_importance", "delta", "is_rf_top_n", "is_xgb_top_n"]
            ]
            .head(min(10, len(model_delta_bits_df)))
            .to_string(index=False)
        )

    out_dir = _ensure_output_dir(args)
    prefix = _resolve_output_prefix(args, target_info)

    png_path = out_dir / f"{prefix}_similarity_map.png"
    svg_path = out_dir / f"{prefix}_similarity_map.svg"
    bits_path = out_dir / f"{prefix}_top_bits.csv"
    top_n_chart_png_path = out_dir / f"{prefix}_top_n_chart.png"
    top_n_chart_svg_path = out_dir / f"{prefix}_top_n_chart.svg"
    fragments_path = out_dir / f"{prefix}_global_fragments.csv"
    matched_path = out_dir / f"{prefix}_matched_fragments.csv"
    atoms_path = out_dir / f"{prefix}_atom_weights.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    receptor_prob_path = out_dir / f"{prefix}_receptor_probabilities.csv"
    receptor_grid_path = out_dir / f"{prefix}_receptor_grid.png"
    comparison_tag = ""
    comparison_png_path = None
    comparison_svg_path = None
    comparison_matched_path = None
    comparison_atoms_path = None
    delta_metrics_path = None
    receptor_delta_path = None
    model_delta_bits_path = None
    model_delta_chart_png_path = None
    model_delta_chart_svg_path = None
    model_delta_fragments_path = None
    model_delta_matched_path = None
    model_delta_atoms_path = None
    model_delta_png_path = None
    model_delta_svg_path = None
    if comparison_target_info is not None:
        comparison_tag = _sanitize_label(comparison_target_info["smiles"])[:24]
        comparison_png_path = out_dir / f"{prefix}_comparison_{comparison_tag}_similarity_map.png"
        comparison_svg_path = out_dir / f"{prefix}_comparison_{comparison_tag}_similarity_map.svg"
        comparison_matched_path = out_dir / f"{prefix}_comparison_{comparison_tag}_matched_fragments.csv"
        comparison_atoms_path = out_dir / f"{prefix}_comparison_{comparison_tag}_atom_weights.csv"
        delta_metrics_path = out_dir / f"{prefix}_comparison_{comparison_tag}_delta_metrics.csv"
        if args.mode == "qsar":
            receptor_delta_path = out_dir / f"{prefix}_comparison_{comparison_tag}_receptor_delta.csv"
    if args.model_delta:
        model_delta_bits_path = out_dir / f"{prefix}_rf_xgb_delta_top_bits.csv"
        model_delta_chart_png_path = out_dir / f"{prefix}_rf_xgb_delta_chart.png"
        model_delta_chart_svg_path = out_dir / f"{prefix}_rf_xgb_delta_chart.svg"
        model_delta_fragments_path = out_dir / f"{prefix}_rf_xgb_delta_global_fragments.csv"
        model_delta_matched_path = out_dir / f"{prefix}_rf_xgb_delta_matched_fragments.csv"
        model_delta_atoms_path = out_dir / f"{prefix}_rf_xgb_delta_atom_weights.csv"
        model_delta_png_path = out_dir / f"{prefix}_rf_xgb_delta_similarity_map.png"
        model_delta_svg_path = out_dir / f"{prefix}_rf_xgb_delta_similarity_map.svg"

    _print_section("Render Outputs")
    _draw_similarity_map_png(target_info["mol"], atom_weights, png_path, draw_size=int(args.draw_size))
    _draw_similarity_map_svg(target_info["mol"], atom_weights, svg_path, draw_size=int(args.draw_size))

    bits_df.to_csv(bits_path, index=False)
    fragment_df.to_csv(fragments_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    atom_df.to_csv(atoms_path, index=False)
    if comparison_target_info is not None:
        _draw_similarity_map_png(
            comparison_target_info["mol"],
            comparison_atom_weights,
            comparison_png_path,
            draw_size=int(args.draw_size),
        )
        _draw_similarity_map_svg(
            comparison_target_info["mol"],
            comparison_atom_weights,
            comparison_svg_path,
            draw_size=int(args.draw_size),
        )
        comparison_matched_df.to_csv(comparison_matched_path, index=False)
        comparison_atom_df.to_csv(comparison_atoms_path, index=False)
        delta_metrics_df.to_csv(delta_metrics_path, index=False)
    if args.model_delta:
        _draw_similarity_map_png(
            target_info["mol"],
            model_delta_atom_weights,
            model_delta_png_path,
            draw_size=int(args.draw_size),
            color_map=_delta_similarity_colormap(),
        )
        _draw_similarity_map_svg(
            target_info["mol"],
            model_delta_atom_weights,
            model_delta_svg_path,
            draw_size=int(args.draw_size),
            color_map=_delta_similarity_colormap(),
        )
        model_delta_bits_df.to_csv(model_delta_bits_path, index=False)
        model_delta_fragment_df.to_csv(model_delta_fragments_path, index=False)
        model_delta_matched_df.to_csv(model_delta_matched_path, index=False)
        model_delta_atom_df.to_csv(model_delta_atoms_path, index=False)
        _save_model_delta_chart(
            model_delta_df=model_delta_bits_df,
            png_path=model_delta_chart_png_path,
            svg_path=model_delta_chart_svg_path,
        )
    if args.top_n_chart:
        if args.mode == "qspr":
            chart_title = (
                f"QSPR {_model_label(args.model)}: "
                f"Top {len(bits_df)} bits by {args.importance} importance"
            )
        else:
            chart_title = (
                f"QSAR {problem['target_name']} {_model_label(args.model)}: "
                f"Top {len(bits_df)} bits by {args.importance} importance"
            )
        _save_top_n_chart(
            bits_df=bits_df,
            importance=args.importance,
            importance_meta=importance_meta,
            chart_title=chart_title,
            png_path=top_n_chart_png_path,
            svg_path=top_n_chart_svg_path,
        )

    receptor_df = None
    receptor_grid_image = None
    if args.mode == "qsar":
        _print_section("All Receptor Probabilities")
        if receptor_grid_shape is None:
            receptor_df = _build_qsar_receptor_probability_table(
                problem=problem,
                selected_receptor=str(problem["target_name"]),
                selected_model=model,
                model_backend=args.model,
                cache_dir=cache_dir,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                target_vector=target_info["vector"],
            )
        else:
            _print_key_value("Grid layout", f"{receptor_grid_shape[0]}x{receptor_grid_shape[1]}")
            receptor_df, receptor_grid_image = _build_qsar_receptor_grid(
                problem=problem,
                selected_receptor=str(problem["target_name"]),
                selected_model=model,
                model_backend=args.model,
                importance=args.importance,
                cache_dir=cache_dir,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                shap_sample_size=int(args.shap_sample_size),
                boruta_n_trials=boruta_n_trials,
                boruta_sample=boruta_sample,
                boruta_normalize=boruta_normalize,
                boruta_train_or_test=boruta_train_or_test,
                boruta_rf_n_estimators=boruta_rf_n_estimators,
                boruta_rf_max_depth=boruta_rf_max_depth,
                target_info=target_info,
                draw_size=int(args.draw_size),
                atom_aggregation=args.atom_aggregation,
                top_n_bits=top_n_bits,
                grid_cols=receptor_grid_shape[0],
                grid_rows=receptor_grid_shape[1],
                receptor_grid_top_n=receptor_grid_top_n,
            )
        receptor_df.to_csv(receptor_prob_path, index=False)
        if receptor_grid_image is not None:
            receptor_grid_image.save(receptor_grid_path)
        display_columns = [
            col
            for col in [
                "rank",
                "receptor",
                "receptor_czech_name",
                "probability",
                "confidence",
                "is_selected_receptor",
            ]
            if col in receptor_df.columns
        ]
        print(receptor_df[display_columns].to_string(index=False))
        if comparison_target_info is not None:
            comparison_receptor_df = _build_qsar_receptor_probability_table(
                problem=problem,
                selected_receptor=str(problem["target_name"]),
                selected_model=model,
                model_backend=args.model,
                cache_dir=cache_dir,
                data_sig=data_sig,
                radius=radius,
                n_bits=n_bits,
                n_estimators=n_estimators,
                random_seed=random_seed,
                n_jobs=n_jobs,
                target_vector=comparison_target_info["vector"],
            )
            receptor_delta_df = _build_qsar_receptor_delta_table(
                primary_receptor_df=receptor_df,
                comparison_receptor_df=comparison_receptor_df,
            )
            receptor_delta_df.to_csv(receptor_delta_path, index=False)
            _print_section("Receptor Delta")
            print(
                receptor_delta_df[
                    [
                        "rank",
                        "receptor",
                        "receptor_czech_name",
                        "primary_probability",
                        "comparison_probability",
                        "probability_delta",
                    ]
                ].to_string(index=False)
            )

    summary_df = pd.DataFrame(
        [
            {
                "mode": args.mode,
                "model_backend": args.model,
                "importance": args.importance,
                "receptor": args.receptor or "",
                "selection_mode": target_info["selection_mode"],
                "dataset_index": "" if target_info["dataset_index"] is None else int(target_info["dataset_index"]),
                "smiles": target_info["smiles"],
                "delta_smiles": comparison_target_info["smiles"] if comparison_target_info is not None else "",
                "predicted_probability": target_info["predicted_probability"],
                "prediction_confidence": target_info["prediction_confidence"],
                "predicted_value": target_info["predicted_value"],
                "delta_predicted_probability": (
                    comparison_target_info["predicted_probability"] - target_info["predicted_probability"]
                    if comparison_target_info is not None
                    else ""
                ),
                "delta_predicted_value": (
                    comparison_target_info["predicted_value"] - target_info["predicted_value"]
                    if comparison_target_info is not None
                    else ""
                ),
                "predicted_class": target_info["predicted_class"],
                "receptor_name": target_info.get("receptor_name", ""),
                "receptor_probability": target_info.get("receptor_probability", ""),
                "delta_receptor_probability": (
                    comparison_target_info.get("receptor_probability", 0.0) - target_info.get("receptor_probability", 0.0)
                    if comparison_target_info is not None
                    else ""
                ),
                "radius": radius,
                "n_bits": n_bits,
                "n_estimators": n_estimators,
                "top_n_bits": top_n_bits,
                "receptor_grid_top_n": receptor_grid_top_n,
                "shap_sample_size": int(args.shap_sample_size),
                "atom_aggregation": args.atom_aggregation,
                "model_cache_status": model_cache_status,
                "value_model_cache_status": value_model_cache_status,
                "importance_cache_status": importance_cache_status,
                "importance_scope": importance_meta.get("scope", ""),
                "importance_source": importance_meta.get("score_source", ""),
                "positive_bit_count": int(importance_meta.get("positive_count", int(np.count_nonzero(bit_scores > 0)))),
                "model_delta_enabled": bool(args.model_delta),
                "matched_fragment_count": int(len(matched_df)),
                "matched_atom_count": int(np.count_nonzero(atom_df["hit_count"].to_numpy() > 0)),
                "png_path": str(png_path),
                "svg_path": str(svg_path),
                "comparison_png_path": str(comparison_png_path) if comparison_png_path is not None else "",
                "comparison_svg_path": str(comparison_svg_path) if comparison_svg_path is not None else "",
                "model_delta_png_path": str(model_delta_png_path) if model_delta_png_path is not None else "",
                "model_delta_svg_path": str(model_delta_svg_path) if model_delta_svg_path is not None else "",
                "bits_path": str(bits_path),
                "model_delta_bits_path": str(model_delta_bits_path) if model_delta_bits_path is not None else "",
                "top_n_chart_png_path": str(top_n_chart_png_path) if args.top_n_chart else "",
                "top_n_chart_svg_path": str(top_n_chart_svg_path) if args.top_n_chart else "",
                "model_delta_chart_png_path": str(model_delta_chart_png_path) if model_delta_chart_png_path is not None else "",
                "model_delta_chart_svg_path": str(model_delta_chart_svg_path) if model_delta_chart_svg_path is not None else "",
                "fragments_path": str(fragments_path),
                "matched_path": str(matched_path),
                "atoms_path": str(atoms_path),
                "comparison_matched_path": str(comparison_matched_path) if comparison_matched_path is not None else "",
                "comparison_atoms_path": str(comparison_atoms_path) if comparison_atoms_path is not None else "",
                "model_delta_fragments_path": str(model_delta_fragments_path) if model_delta_fragments_path is not None else "",
                "model_delta_matched_path": str(model_delta_matched_path) if model_delta_matched_path is not None else "",
                "model_delta_atoms_path": str(model_delta_atoms_path) if model_delta_atoms_path is not None else "",
                "delta_metrics_path": str(delta_metrics_path) if delta_metrics_path is not None else "",
                "receptor_probabilities_path": str(receptor_prob_path) if receptor_df is not None else "",
                "receptor_delta_path": str(receptor_delta_path) if receptor_delta_path is not None else "",
                "receptor_grid_path": str(receptor_grid_path) if receptor_grid_image is not None else "",
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    _print_section("Outputs")
    _print_key_value("Matched fragments on target", len(matched_df))
    _print_key_value("Similarity map PNG", png_path)
    _print_key_value("Similarity map SVG", svg_path)
    _print_key_value("Top bits CSV", bits_path)
    if args.top_n_chart:
        _print_key_value("Top-N chart PNG", top_n_chart_png_path)
        _print_key_value("Top-N chart SVG", top_n_chart_svg_path)
    _print_key_value("Global fragments CSV", fragments_path)
    _print_key_value("Matched fragments CSV", matched_path)
    _print_key_value("Atom weights CSV", atoms_path)
    if comparison_target_info is not None:
        _print_key_value("Comparison similarity map PNG", comparison_png_path)
        _print_key_value("Comparison similarity map SVG", comparison_svg_path)
        _print_key_value("Comparison matched fragments CSV", comparison_matched_path)
        _print_key_value("Comparison atom weights CSV", comparison_atoms_path)
        _print_key_value("Delta metrics CSV", delta_metrics_path)
    if args.model_delta:
        _print_key_value("RF/XGB delta similarity map PNG", model_delta_png_path)
        _print_key_value("RF/XGB delta similarity map SVG", model_delta_svg_path)
        _print_key_value("RF/XGB delta bits CSV", model_delta_bits_path)
        _print_key_value("RF/XGB delta chart PNG", model_delta_chart_png_path)
        _print_key_value("RF/XGB delta chart SVG", model_delta_chart_svg_path)
        _print_key_value("RF/XGB delta fragments CSV", model_delta_fragments_path)
        _print_key_value("RF/XGB delta matched fragments CSV", model_delta_matched_path)
        _print_key_value("RF/XGB delta atom weights CSV", model_delta_atoms_path)
    if receptor_df is not None:
        _print_key_value("Receptor probabilities CSV", receptor_prob_path)
    if receptor_delta_df is not None:
        _print_key_value("Receptor delta CSV", receptor_delta_path)
    if receptor_grid_image is not None:
        _print_key_value("Receptor grid PNG", receptor_grid_path)
    _print_key_value("Summary CSV", summary_path)


if __name__ == "__main__":
    main()

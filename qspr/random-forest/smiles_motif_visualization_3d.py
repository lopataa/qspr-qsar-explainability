from pathlib import Path
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "random-forest"
CACHE_VERSION = 1


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
        description="Render 3D highlighted Morgan motifs for a QSPR molecule using BorutaShap scores."
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
        default=8,
        help="Number of important active bits to render for the selected molecule.",
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
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="3D view elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-62.0,
        help="3D view azimuth angle.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="Output image DPI.",
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

    meta = {
        "scope": scope,
        "score_source": "history_x_mean",
        "accepted_count": len(accepted),
        "tentative_count": len(getattr(selector, "tentative", [])),
        "rejected_count": len(getattr(selector, "rejected", [])),
        "positive_count": int(np.count_nonzero(bit_scores > 0)),
    }
    return bit_scores, meta


def _normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _lerp_rgb(color_a, color_b, t):
    return tuple(float(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def _rank_bits_for_mol(bit_info_map, bit_scores, top_n):
    rows = []
    for bit, occurrences in bit_info_map.items():
        score = float(bit_scores[bit])
        if score <= 0:
            continue
        rows.append((int(bit), score, len(occurrences)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _atom_and_bond_payload(mol, bit_info_map, bit_scores, top_n):
    ranked = _rank_bits_for_mol(bit_info_map, bit_scores, top_n=top_n)
    if not ranked:
        raise ValueError("No active bits with positive score were found for this molecule.")

    score_values = [score for _, score, _ in ranked]
    norm_scores = _normalize_scores(score_values)
    low_color = (0.62, 0.86, 0.98)
    high_color = (0.60, 0.46, 0.95)
    min_alpha, max_alpha = 0.0, 0.75

    atom_values = {}
    bond_values = {}
    bit_rows = []

    for idx, (bit, score, occurrences_count) in enumerate(ranked):
        rank = idx + 1
        norm = float(norm_scores[idx])
        alpha = float(min_alpha + (max_alpha - min_alpha) * norm)
        rgb = _lerp_rgb(low_color, high_color, norm)
        rgba = (*rgb, alpha)

        highlight_atoms = set()
        highlight_bonds = set()
        fragments = set()

        for center_atom_idx, radius in bit_info_map[bit]:
            env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx))
            atoms = {center_atom_idx}
            for bond_id in env:
                bond = mol.GetBondWithIdx(bond_id)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())

            highlight_atoms.update(atoms)
            highlight_bonds.update(env)

            fragment = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=sorted(atoms),
                bondsToUse=sorted(env),
                canonical=True,
                isomericSmiles=False,
            )
            if fragment:
                fragments.add(fragment)

        for atom_idx in highlight_atoms:
            atom_values.setdefault(atom_idx, []).append((rgba, norm))
        for bond_idx in highlight_bonds:
            bond_values.setdefault(bond_idx, []).append((rgba, norm))

        bit_rows.append(
            {
                "rank": rank,
                "bit": bit,
                "importance": score,
                "importance_norm": norm,
                "highlight_alpha": alpha,
                "occurrences": occurrences_count,
                "fragments": "; ".join(sorted(fragments)),
            }
        )

    atom_styles = {}
    for atom_idx, values in atom_values.items():
        values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
        rgba = values_sorted[0][0]
        atom_styles[atom_idx] = {"rgba": rgba, "norm": values_sorted[0][1]}

    bond_styles = {}
    for bond_idx, values in bond_values.items():
        values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
        rgba = values_sorted[0][0]
        bond_styles[bond_idx] = {"rgba": rgba, "norm": values_sorted[0][1]}

    return atom_styles, bond_styles, bit_rows


def _build_3d_mol(base_mol, random_seed):
    mol = Chem.AddHs(Chem.Mol(base_mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError("RDKit could not generate a 3D conformer for this molecule.")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass
    return Chem.RemoveHs(mol)


def _set_equal_axes(ax, coords):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    mid = (mins + maxs) / 2.0
    max_range = float((maxs - mins).max()) * 0.60 + 0.20
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def _render_3d_overlay(mol3d, atom_styles, bond_styles, output_path, elev, azim, dpi):
    conf = mol3d.GetConformer()
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol3d.GetNumAtoms())],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        xs = [coords[i, 0], coords[j, 0]]
        ys = [coords[i, 1], coords[j, 1]]
        zs = [coords[i, 2], coords[j, 2]]
        style = bond_styles.get(bond.GetIdx())
        if style is None:
            ax.plot(xs, ys, zs, color=(0.55, 0.58, 0.62), alpha=0.30, linewidth=1.5, zorder=1)
        else:
            rgba = style["rgba"]
            ax.plot(xs, ys, zs, color=rgba[:3], alpha=max(0.30, rgba[3]), linewidth=2.8, zorder=2)

    atom_colors = []
    atom_sizes = []
    for atom_idx in range(mol3d.GetNumAtoms()):
        style = atom_styles.get(atom_idx)
        if style is None:
            atom_colors.append((0.67, 0.70, 0.74, 0.42))
            atom_sizes.append(56)
        else:
            rgba = style["rgba"]
            atom_colors.append((rgba[0], rgba[1], rgba[2], max(0.35, rgba[3])))
            atom_sizes.append(130 + 140 * style["norm"])

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=atom_sizes,
        c=atom_colors,
        edgecolors=(0.10, 0.10, 0.10, 0.35),
        linewidths=0.5,
        depthshade=True,
        zorder=3,
    )

    for atom_idx, style in atom_styles.items():
        atom = mol3d.GetAtomWithIdx(atom_idx)
        x, y, z = coords[atom_idx]
        ax.text(
            x,
            y,
            z,
            atom.GetSymbol(),
            fontsize=8,
            color=style["rgba"][:3],
            alpha=0.9,
            zorder=4,
        )

    _set_equal_axes(ax, coords)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
        file_signature,
        fingerprint_mol_with_bit_info,
        load_pickle_cache,
        load_dataset,
        make_binary_target,
        save_pickle_cache,
    )

    RDLogger.DisableLog("rdApp.warning")

    df = load_dataset(DATA_PATH)
    df, x, mols, bit_info_maps = build_feature_matrix_with_metadata(
        df,
        radius=ECFP_RADIUS,
        n_bits=ECFP_N_BITS,
    )
    y, cutoff = make_binary_target(df["Solubility"].to_numpy())

    data_sig = file_signature(DATA_PATH)
    cache_dir = qspr_root / MODEL_DIR / "cache"

    rf_cache_path = cache_dir / "rf_binary_model.pkl"
    rf_cache_meta = {
        "version": CACHE_VERSION,
        "kind": "rf_binary_model",
        "data": data_sig,
        "ecfp_radius": int(ECFP_RADIUS),
        "ecfp_n_bits": int(ECFP_N_BITS),
        "n_estimators": int(N_ESTIMATORS),
        "n_jobs": int(N_JOBS),
        "random_seed": int(RANDOM_SEED),
    }
    rf = load_pickle_cache(rf_cache_path, rf_cache_meta)
    if rf is None:
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            n_jobs=N_JOBS,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
        )
        rf.fit(x, y)
        save_pickle_cache(rf_cache_path, rf_cache_meta, rf)
        rf_cache_status = "miss"
    else:
        rf_cache_status = "hit"

    boruta_n_trials = BORUTA_N_TRIALS if args.boruta_trials is None else args.boruta_trials
    boruta_sample = BORUTA_SAMPLE if args.boruta_sample is None else args.boruta_sample
    boruta_normalize = BORUTA_NORMALIZE if args.boruta_normalize is None else args.boruta_normalize

    boruta_cache_path = cache_dir / "boruta_binary_scores.pkl"
    boruta_cache_meta = {
        "version": CACHE_VERSION,
        "kind": "boruta_binary_scores",
        "data": data_sig,
        "ecfp_radius": int(ECFP_RADIUS),
        "ecfp_n_bits": int(ECFP_N_BITS),
        "random_seed": int(RANDOM_SEED),
        "n_jobs": int(N_JOBS),
        "boruta_n_trials": int(boruta_n_trials),
        "boruta_sample": bool(boruta_sample),
        "boruta_normalize": bool(boruta_normalize),
        "boruta_train_or_test": str(BORUTA_TRAIN_OR_TEST),
        "boruta_rf_n_estimators": int(BORUTA_RF_N_ESTIMATORS),
        "boruta_rf_max_depth": int(BORUTA_RF_MAX_DEPTH),
    }
    cached_boruta = load_pickle_cache(boruta_cache_path, boruta_cache_meta)
    if cached_boruta is None:
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

    atom_styles, bond_styles, bit_rows = _atom_and_bond_payload(
        mol=target_mol,
        bit_info_map=target_bit_info,
        bit_scores=bit_scores,
        top_n=args.top_n_bits,
    )
    mol3d = _build_3d_mol(target_mol, random_seed=RANDOM_SEED)

    if args.output_prefix:
        output_prefix = args.output_prefix
    elif target_index is None:
        output_prefix = f"smiles_{_sanitize_label(target_smiles)}"
    else:
        output_prefix = f"dataset_idx_{target_index}_id_{_sanitize_label(str(target_id))}"

    out_dir = qspr_root / MODEL_DIR / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{output_prefix}_motif_3d_overlay.png"
    bits_path = out_dir / f"{output_prefix}_motif_3d_bits.csv"

    _render_3d_overlay(
        mol3d=mol3d,
        atom_styles=atom_styles,
        bond_styles=bond_styles,
        output_path=image_path,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )

    summary_df = pd.DataFrame(bit_rows)
    summary_df.insert(0, "smiles", target_smiles)
    summary_df.insert(1, "rf_predicted_p_soluble", probability)
    summary_df.to_csv(bits_path, index=False)

    print(f"Trained on {len(df)} molecules. Binary cutoff (median solubility): {cutoff:.5f}")
    print(f"Cache: rf={rf_cache_status}, boruta={boruta_cache_status}")
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
    print(f"Saved 3D overlay image: {image_path}")
    print(f"Saved 3D bit summary: {bits_path}")


if __name__ == "__main__":
    main()

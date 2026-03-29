from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder


def load_qsar_dataset(path):
    return pd.read_csv(path)


def smiles_to_mol(value):
    if pd.isna(value) or str(value).lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(str(value))
    except Exception:
        return None


def add_mol_column(df, smiles_column="Smiles", mol_column="mol"):
    out = df.copy()
    out[mol_column] = out[smiles_column].map(smiles_to_mol)
    out = out[out[mol_column].notna()].copy()
    return out


def build_morgan_fingerprints(
    df,
    mol_column="mol",
    output_column="fp",
    radius=2,
    n_bits=4096,
):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def mol_to_fp_array(mol):
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    out = df.copy()
    out[output_column] = out[mol_column].map(mol_to_fp_array)
    return out


def encode_targets(df, target_column="Target", output_column="target_encoded"):
    encoder = OneHotEncoder(sparse_output=False)
    target_encoded = encoder.fit_transform(df[[target_column]])
    target_names = encoder.categories_[0]

    out = df.copy()
    out[output_column] = target_encoded.tolist()
    return out, encoder, target_names


def aggregate_targets_by_fingerprint(
    df,
    fp_column="fp",
    encoded_target_column="target_encoded",
    aggregated_target_column="target",
):
    out = df.copy()
    out["fp_tuple"] = out[fp_column].apply(lambda value: tuple(value))

    grouped = out.groupby("fp_tuple")[encoded_target_column].apply(
        lambda rows: np.any(np.vstack(rows.values), axis=0).astype(int)
    )

    out_agg = grouped.reset_index()
    out_agg[fp_column] = out_agg["fp_tuple"].apply(lambda value: np.array(value))
    out_agg = out_agg.rename(columns={encoded_target_column: aggregated_target_column})
    out_agg = out_agg.drop(columns=["fp_tuple"])
    return out_agg


def stack_features_and_targets(df, fp_column="fp", target_column="target"):
    x = np.vstack(df[fp_column].values)
    y = np.vstack(df[target_column].values)
    return x, y


def roc_auc_per_target(y_true, y_pred_prob_matrix):
    return [
        roc_auc_score(y_true[:, idx], y_pred_prob_matrix[:, idx])
        for idx in range(y_true.shape[1])
    ]


def to_target_probability_matrix(y_pred_prob, n_targets):
    # OneVsRest outputs can differ by estimator/version; normalize to (n_samples, n_targets).
    if isinstance(y_pred_prob, list):
        if len(y_pred_prob) == 0:
            return np.zeros((0, n_targets), dtype=float)

        first = np.asarray(y_pred_prob[0])
        if first.ndim == 1:
            return np.column_stack(y_pred_prob)

        if first.ndim == 2 and first.shape[1] == 2:
            return np.column_stack([np.asarray(prob)[:, 1] for prob in y_pred_prob])

    matrix = np.asarray(y_pred_prob)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D probability output.")

    if matrix.shape[1] == n_targets:
        return matrix

    if matrix.shape[0] == n_targets:
        return matrix.T

    raise ValueError(
        f"Could not resolve probability matrix shape {matrix.shape} for n_targets={n_targets}."
    )


def plot_multitarget_roc_curves(
    y_true,
    y_pred_prob_matrix,
    target_names,
    save_path,
    title,
    czech_names=None,
):
    n_targets = y_true.shape[1]
    cmap = matplotlib.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in range(n_targets):
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred_prob_matrix[:, idx])
        auc_score = roc_auc_score(y_true[:, idx], y_pred_prob_matrix[:, idx])
        color = cmap(idx / n_targets)

        label_name = target_names[idx]
        if czech_names is not None:
            label_name = czech_names.get(str(target_names[idx]).lower(), target_names[idx])

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{label_name} (AUC = {auc_score:.2f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format=save_path.suffix.lstrip("."), bbox_inches="tight")
    plt.show()
    return save_path

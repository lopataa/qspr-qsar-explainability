from pathlib import Path

import numpy as np
import pandas as pd

from qspr_config import FIG_DPI, OUTPUT_DIRNAME


def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["SMILES", "Solubility"]).reset_index(drop=True)
    return df


def build_feature_matrix(df, radius, n_bits):
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    features = []
    keep_rows = []
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    for idx, smi in df["SMILES"].items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)
        keep_rows.append(idx)

    df = df.loc[keep_rows].reset_index(drop=True)
    x = np.asarray(features, dtype=np.float32)
    return df, x


def make_binary_target(values):
    median = np.median(values)
    y = (values >= median).astype(int)
    return y, median


def apply_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
        }
    )


def resolve_output_dir(model_dir_name):
    cwd = Path.cwd()
    if cwd.name == model_dir_name:
        base_dir = cwd
    else:
        base_dir = cwd / model_dir_name
    out_dir = base_dir / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

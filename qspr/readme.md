# Identifikace strukturních motivů chemických látek klíčových pro rozhodování u regresních QSPR modelů

## Environment (uv)

Run from the `qspr` directory:

```
uv venv .venv
uv pip install "numpy<2" "shap>=0.48" "numba>=0.57" xgboost==1.7.6 \
  BorutaShap scikit-learn pandas scipy statsmodels matplotlib seaborn rdkit-pypi nbformat nbconvert ipykernel
```

## Random Forest motif visualization

From the repository root:

```bash
./.venv/bin/python qspr/random-forest/smiles_motif_visualization.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

or visualize a filtered AqSolDB row:

```bash
./.venv/bin/python qspr/random-forest/smiles_motif_visualization.py --dataset-index 0 --top-n-bits 8
```

Outputs are written to `qspr/random-forest/outputs/` as:

- `*_motif_grid.png`: one panel per top bit, with normalized opacity-based highlighting
- `*_motif_overlay.png`: top active Morgan-bit motifs overlaid on a single molecule
- `*_motif_bits.csv`: bit ids, BorutaShap importance, occurrence counts, and motif fragments

The motif ranking in this script is computed from BorutaShap (SHAP-based) importances.
Useful runtime controls:

```bash
./.venv/bin/python qspr/random-forest/smiles_motif_visualization.py \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --boruta-trials 10 \
  --boruta-sample
```

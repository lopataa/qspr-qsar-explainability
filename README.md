# qspr-explainability

## Project structure

- `common/` shared utilities for ECFP generation and visualization
- `qsar/` classification-oriented QSAR notebooks, configs, datasets, and generated outputs
- `qspr/` regression-oriented QSPR notebooks, configs, datasets, and motif visualization scripts
- `docs/` LaTeX sources
- `out/` compiled document artifacts
- `tmp/` local scratch outputs

## Environment

The project declares Python `>=3.11` and pins the notebook/scientific stack directly in [`pyproject.toml`](pyproject.toml).

Example setup with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Typical workflows

Open notebooks under `qsar/`, `qspr/`, and `common/` for model training, parameter sweeps, cross-validation, and explainability experiments.

Notebook-by-notebook notes for both QSAR and QSPR are available in [`docs/notebooks.md`](docs/notebooks.md).

The QSPR motif visualization workflow is documented in [`qspr/readme.md`](qspr/readme.md). Example:

```bash
./.venv/bin/python qspr/random-forest/smiles_motif_visualization.py \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Generated figures and CSV summaries are typically written into `outputs/` or `cache/` folders inside the corresponding experiment directories.

## CLI tools

The repository also includes a shared CLI for global ECFP fragment visualization:

[`common/global_ecfp_visualization.py`](common/global_ecfp_visualization.py)

It renders a single-molecule similarity map from globally important Morgan fingerprint fragments and supports both repository branches:

- `--mode qspr` for regression-style QSPR experiments
- `--mode qsar` for receptor-specific QSAR experiments

Example QSPR run with SHAP and XGBoost:

```bash
./.venv/bin/python common/global_ecfp_visualization.py \
  --mode qspr \
  --importance shap \
  --model xgb \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Example QSAR run with BorutaShap for a receptor:

```bash
./.venv/bin/python common/global_ecfp_visualization.py \
  --mode qsar \
  --importance boruta-shap \
  --model rf \
  --receptor ar \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

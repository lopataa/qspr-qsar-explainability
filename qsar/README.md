# Identification and visualization of structural motifs in chemical compounds that are key to decision-making in QSAR and QSPR models

## Environment (uv)

Notebook-by-notebook descriptions for both QSAR and QSPR are in [`../docs/notebooks.md`](../docs/notebooks.md).

Run from the repository root:

```bash
uv venv .venv
uv sync
```

## QSAR data and shared settings

- Dataset: `qsar/nr_ic_merged.csv`
- Shared preprocessing/utilities: `qsar/qsar_common.py`
- Central configuration: `qsar/qsar_config.py`

The QSAR workflow uses Morgan fingerprints (ECFP), multi-target one-vs-rest classification, and explainability based on SHAP and BorutaShap.

## Main QSAR notebooks

Open notebooks in `qsar/` and run them from top to bottom:

- `qsar/random-forest/random-forest.ipynb`: baseline Random Forest classifier
- `qsar/xgboost/xgboost.ipynb`: baseline XGBoost classifier
- `qsar/random-forest/random-forest-shap.ipynb`: SHAP feature importance for RF
- `qsar/xgboost/xgboost-shap.ipynb`: SHAP feature importance for XGB
- `qsar/random-forest/random-forest-boruta-shap.ipynb`: BorutaShap for RF
- `qsar/xgboost/xgboost-boruta-shap.ipynb`: BorutaShap for XGB
- `qsar/model_performance_comparison.ipynb`: RF vs XGB ROC-AUC comparison
- `qsar/explainability_comparison.ipynb`: SHAP/Boruta and RF/XGB explainability alignment

## Aggregated comparison outputs

Model and explainability comparison notebooks export CSV summaries to:

- `qsar/outputs/model-comparison/model_performance_per_target.csv`
- `qsar/outputs/model-comparison/model_performance_summary.csv`
- `qsar/outputs/model-comparison/explainability_feature_comparison.csv`
- `qsar/outputs/model-comparison/explainability_target_summary.csv`

## Motif visualization scripts

### Random Forest receptor-wise motifs

From the repository root:

```bash
./.venv/bin/python qsar/random-forest/random-forest-motif-visualization.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Outputs are written by default to `qsar/outputs/random-forest-motifs/`.

### XGBoost receptor-wise motifs

From the repository root:

```bash
./.venv/bin/python qsar/xgboost/xgboost-motif-visualization.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Outputs are written by default to `qsar/outputs/xgboost-motifs/`.

### Global ECFP visualization (QSAR mode)

Use the shared CLI for global fragment visualization and receptor-specific maps:

```bash
./.venv/bin/python common/global_ecfp_visualization.py \
  --mode qsar \
  --importance boruta-shap \
  --model rf \
  --receptor ar \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

You can switch `--importance` between `shap` and `boruta-shap`, and `--model` between `rf` and `xgb`.

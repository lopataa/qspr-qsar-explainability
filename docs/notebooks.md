# Notebook documentation

This document summarizes notebooks used in the QSAR and QSPR parts of the project. Each entry states the purpose and the main output of the notebook.

## QSAR notebooks

### `qsar/random-forest/random-forest.ipynb`
This notebook is the baseline QSAR multi-target classification workflow with Random Forest. It includes data preparation, training, and final ROC evaluation.

### `qsar/random-forest/random-forest-shap.ipynb`
This notebook performs SHAP-based feature selection for the Random Forest QSAR model. It stores per-target checkpoints and evaluates the reduced feature set.

### `qsar/random-forest/random-forest-boruta-shap.ipynb`
This notebook performs BorutaShap feature selection for Random Forest across all targets. It saves per-target checkpoints and generates Z-score based summary plots.

### `qsar/random-forest/random-forest-cross-validation.ipynb`
This notebook runs cross-validation for the Random Forest QSAR classifier. It reports fold-level ROC-AUC statistics.

### `qsar/random-forest/random-forest-parameter-testing.ipynb`
This notebook tests selected Random Forest hyperparameters for the QSAR classification task. It is used to compare model behavior across parameter values.

### `qsar/xgboost/xgboost.ipynb`
This notebook is the baseline QSAR multi-target classification workflow with XGBoost. It mirrors the main training and evaluation structure of the Random Forest baseline.

### `qsar/xgboost/xgboost-shap.ipynb`
This notebook performs SHAP-based feature selection for the XGBoost QSAR model. It stores per-target checkpoints and validates the selected features.

### `qsar/xgboost/xgboost-boruta-shap.ipynb`
This notebook performs BorutaShap feature selection for XGBoost across all targets. It saves per-target checkpoints and produces Z-score summary visualizations.

### `qsar/xgboost/xgboost-cross-validation.ipynb`
This notebook runs cross-validation for the XGBoost QSAR classifier. It reports ROC-AUC behavior across folds.

### `qsar/xgboost/xgboost-parameter-testing.ipynb`
This notebook tests selected XGBoost hyperparameters for the QSAR classification task. It provides a controlled comparison of parameter settings.

## QSPR notebooks

### `qspr/ecfp.ipynb`
This notebook prepares and checks ECFP representations for QSPR regression experiments. It is used to verify fingerprint settings before model training.

### `qspr/boruta-shap.ipynb`
This notebook applies BorutaShap feature selection in the QSPR workflow. It identifies robust fingerprint bits for downstream modeling.

### `qspr/random-forest.ipynb`
This notebook is the baseline QSPR regression workflow with Random Forest. It provides the reference regression pipeline for RF experiments.

### `qspr/xgboost.ipynb`
This notebook is the baseline QSPR regression workflow with XGBoost. It provides the reference regression pipeline for XGBoost experiments.

### `qspr/shap.ipynb`
This notebook performs SHAP-based interpretation for QSPR models. It is focused on global and local feature effect analysis.

### `qspr/shap_rf_vs_xgboost.ipynb`
This notebook compares SHAP outputs between Random Forest and XGBoost QSPR models. It is used to evaluate consistency of feature importance patterns.

### `qspr/roc-cross-validation.ipynb`
This notebook evaluates ROC-oriented metrics under cross-validation in the QSPR workspace. It supports threshold-based interpretation of predictions.

### `qspr/roc-n-estimators.ipynb`
This notebook analyzes how ROC-oriented metrics change with estimator count. It is used for model complexity tuning.

## QSPR Random Forest experiment notebooks

### `qspr/random-forest/bit_importance_histogram.ipynb`
This notebook visualizes the distribution of feature importances for QSPR Random Forest. It helps assess how concentrated the model signal is.

### `qspr/random-forest/bit_importance_top_n.ipynb`
This notebook reports the top-N most important bits for QSPR Random Forest. It is used as a shortlist for motif-level interpretation.

### `qspr/random-forest/cv_mae.ipynb`
This notebook computes cross-validated MAE for QSPR Random Forest. It is used as a fold-based error benchmark.

### `qspr/random-forest/cv_threshold_accuracy.ipynb`
This notebook computes threshold-based accuracy under cross-validation for QSPR Random Forest. It supports decision-style evaluation.

### `qspr/random-forest/n_bits_vs_accuracy.ipynb`
This notebook evaluates the effect of fingerprint size on threshold accuracy for QSPR Random Forest. It supports fingerprint length selection.

### `qspr/random-forest/n_bits_vs_mae.ipynb`
This notebook evaluates the effect of fingerprint size on MAE for QSPR Random Forest. It is used to choose a practical feature dimension.

### `qspr/random-forest/n_bits_vs_rmse.ipynb`
This notebook evaluates the effect of fingerprint size on RMSE for QSPR Random Forest. It is useful when larger residuals are emphasized.

### `qspr/random-forest/n_bits_vs_roc_auc.ipynb`
This notebook evaluates the effect of fingerprint size on ROC-AUC for QSPR Random Forest. It complements MAE and RMSE with ranking quality.

### `qspr/random-forest/n_estimators_vs_accuracy.ipynb`
This notebook evaluates threshold accuracy as the number of estimators changes for QSPR Random Forest. It supports estimator count tuning.

### `qspr/random-forest/n_estimators_vs_mae.ipynb`
This notebook evaluates MAE as the number of estimators changes for QSPR Random Forest. It helps identify a stable complexity level.

### `qspr/random-forest/n_estimators_vs_rmse.ipynb`
This notebook evaluates RMSE as the number of estimators changes for QSPR Random Forest. It tracks sensitivity of larger errors to model size.

### `qspr/random-forest/n_estimators_vs_roc_auc.ipynb`
This notebook evaluates ROC-AUC as the number of estimators changes for QSPR Random Forest. It provides a ranking-oriented view of estimator scaling.

### `qspr/random-forest/predicted_vs_actual.ipynb`
This notebook plots predicted versus actual values for QSPR Random Forest. It is used for residual and calibration diagnostics.

### `qspr/random-forest/radius_vs_mae.ipynb`
This notebook evaluates the effect of Morgan radius on MAE for QSPR Random Forest. It supports radius selection for fingerprint generation.

### `qspr/random-forest/roc_auc_cv.ipynb`
This notebook summarizes cross-validated ROC-AUC for QSPR Random Forest. It provides a compact performance stability report.

## QSPR XGBoost experiment notebooks

### `qspr/xgboost/bit_importance_histogram.ipynb`
This notebook visualizes the distribution of feature importances for QSPR XGBoost. It helps assess concentration of model influence.

### `qspr/xgboost/bit_importance_top_n.ipynb`
This notebook reports the top-N most important bits for QSPR XGBoost. It is used for targeted interpretation of key features.

### `qspr/xgboost/cv_mae.ipynb`
This notebook computes cross-validated MAE for QSPR XGBoost. It serves as a fold-based error reference.

### `qspr/xgboost/cv_threshold_accuracy.ipynb`
This notebook computes threshold-based accuracy under cross-validation for QSPR XGBoost. It supports decision-oriented analysis.

### `qspr/xgboost/n_bits_vs_accuracy.ipynb`
This notebook evaluates the effect of fingerprint size on threshold accuracy for QSPR XGBoost. It supports feature-space sizing decisions.

### `qspr/xgboost/n_bits_vs_mae.ipynb`
This notebook evaluates the effect of fingerprint size on MAE for QSPR XGBoost. It is used to balance dimensionality and error.

### `qspr/xgboost/n_bits_vs_rmse.ipynb`
This notebook evaluates the effect of fingerprint size on RMSE for QSPR XGBoost. It highlights sensitivity of larger residuals.

### `qspr/xgboost/n_bits_vs_roc_auc.ipynb`
This notebook evaluates the effect of fingerprint size on ROC-AUC for QSPR XGBoost. It complements direct regression error metrics.

### `qspr/xgboost/n_estimators_vs_accuracy.ipynb`
This notebook evaluates threshold accuracy as estimator count changes for QSPR XGBoost. It is used for estimator count calibration.

### `qspr/xgboost/n_estimators_vs_mae.ipynb`
This notebook evaluates MAE as estimator count changes for QSPR XGBoost. It supports practical model size selection.

### `qspr/xgboost/n_estimators_vs_rmse.ipynb`
This notebook evaluates RMSE as estimator count changes for QSPR XGBoost. It tracks error behavior with increasing complexity.

### `qspr/xgboost/n_estimators_vs_roc_auc.ipynb`
This notebook evaluates ROC-AUC as estimator count changes for QSPR XGBoost. It provides a ranking-focused tuning perspective.

### `qspr/xgboost/predicted_vs_actual.ipynb`
This notebook plots predicted versus actual values for QSPR XGBoost. It is used for calibration and residual diagnostics.

### `qspr/xgboost/radius_vs_mae.ipynb`
This notebook evaluates the effect of Morgan radius on MAE for QSPR XGBoost. It supports radius choice in feature engineering.

### `qspr/xgboost/roc_auc_cv.ipynb`
This notebook summarizes cross-validated ROC-AUC for QSPR XGBoost. It provides a compact view of stability across folds.

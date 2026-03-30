from pathlib import Path
import sys

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

_SCRIPT_DIR = Path(__file__).resolve().parent
_QSAR_DIR = _SCRIPT_DIR.parent
if str(_QSAR_DIR) not in sys.path:
    sys.path.insert(0, str(_QSAR_DIR))

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
from qsar_motif_workflow import build_parser, run_motif_pipeline

XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8


def _model_builder(args):
    return OneVsRestClassifier(
        XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            random_state=args.random_seed,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=args.n_jobs,
        )
    )


def _boruta_estimator_builder(random_seed, n_jobs):
    return XGBClassifier(
        n_estimators=BORUTA_RF_N_ESTIMATORS,
        max_depth=BORUTA_RF_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=random_seed,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=n_jobs,
    )


def main():
    parser = build_parser(
        description="Generate QSAR motif maps/overlays per activity type using XGBoost.",
        n_estimators_help="XGBoost n_estimators for each one-vs-rest classifier.",
        n_jobs_help="CPU workers for XGBoost.",
        output_dir_help="Optional output directory (defaults to qsar/outputs/xgboost-motifs).",
        ecfp_radius=ECFP_RADIUS,
        ecfp_n_bits=ECFP_N_BITS,
        n_estimators_default=N_ESTIMATORS,
        top_n_bits_default=TOP_N_BITS,
        random_seed_default=RANDOM_SEED,
        n_jobs_default=N_JOBS,
    )
    args = parser.parse_args()

    run_motif_pipeline(
        args=args,
        script_file=__file__,
        data_path_default=DATA_PATH,
        output_dirname=OUTPUT_DIRNAME,
        output_subdir="xgboost-motifs",
        cache_subdir="xgboost-motifs",
        model_cache_filename="qsar_ovr_xgb_model.pkl",
        model_cache_kind="qsar_ovr_xgb_model",
        model_cache_meta_extra={
            "max_depth": int(XGB_MAX_DEPTH),
            "learning_rate": float(XGB_LEARNING_RATE),
            "subsample": float(XGB_SUBSAMPLE),
            "colsample_bytree": float(XGB_COLSAMPLE_BYTREE),
        },
        model_cache_status_key="xgb_cache_status",
        model_label="xgb",
        boruta_n_trials_default=BORUTA_N_TRIALS,
        boruta_sample_default=BORUTA_SAMPLE,
        boruta_normalize_default=BORUTA_NORMALIZE,
        boruta_train_or_test=BORUTA_TRAIN_OR_TEST,
        boruta_rf_n_estimators=BORUTA_RF_N_ESTIMATORS,
        boruta_rf_max_depth=BORUTA_RF_MAX_DEPTH,
        model_builder=_model_builder,
        boruta_estimator_builder=_boruta_estimator_builder,
    )


if __name__ == "__main__":
    main()

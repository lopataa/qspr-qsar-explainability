from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

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


def _model_builder(args):
    return OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            class_weight="balanced_subsample",
        )
    )


def _boruta_estimator_builder(random_seed, n_jobs):
    return RandomForestClassifier(
        n_estimators=BORUTA_RF_N_ESTIMATORS,
        max_depth=BORUTA_RF_MAX_DEPTH,
        n_jobs=n_jobs,
        random_state=random_seed,
        class_weight="balanced_subsample",
    )


def main():
    parser = build_parser(
        description="Generate QSAR motif maps/overlays per activity type using Random Forest.",
        n_estimators_help="RandomForest n_estimators for each one-vs-rest classifier.",
        n_jobs_help="CPU workers for RandomForest.",
        output_dir_help="Optional output directory (defaults to qsar/outputs/random-forest-motifs).",
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
        output_subdir="random-forest-motifs",
        cache_subdir="random-forest-motifs",
        model_cache_filename="qsar_ovr_rf_model.pkl",
        model_cache_kind="qsar_ovr_rf_model",
        model_cache_meta_extra={},
        model_cache_status_key="rf_cache_status",
        model_label="rf",
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

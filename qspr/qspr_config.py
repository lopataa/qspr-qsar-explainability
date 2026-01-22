from pathlib import Path

# Data
DATA_PATH = Path(__file__).resolve().parent / "AqSolDB_v1.0_min.csv"

# ECFP fingerprinting
ECFP_RADIUS = 4  # number of bond hops (neighbors)
ECFP_N_BITS = 2048
RADIUS_GRID = [1, 2, 3, 4, 5]
N_BITS_GRID = [64, 128, 256, 512, 1024, 2048, 4096]

# Model settings
N_ESTIMATORS = 500
N_ESTIMATORS_GRID = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
TOP_N_BITS = 25
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = 4

# Output
OUTPUT_DIRNAME = "outputs"
FIG_DPI = 250
FIGSIZE_WIDE = (7.5, 4.5)
FIGSIZE_TALL = (7.5, 6.0)
FIGSIZE_SQUARE = (6.0, 6.0)

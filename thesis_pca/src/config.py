"""실험 전역 설정: 난수 시드, k 목록, 경로."""
from pathlib import Path

SEED = 42

# 데이터
N_TRAIN = 60_000
N_TEST = 10_000
N_FEATURES = 784

# 실험 조건: 784는 PCA 미적용 원본
K_LIST = [2, 5, 10, 20, 50, 100, 200, 784]
VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

# 분류 모델 (모든 k에서 동일)
LR_PARAMS = dict(solver="lbfgs", max_iter=1000, C=1.0, random_state=SEED)
N_TIMING_REPEATS = 3

# 그림
FIG_DPI = 300
N_SCATTER_SAMPLES = 5_000

# 경로
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

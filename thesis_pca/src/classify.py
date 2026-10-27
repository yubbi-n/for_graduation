"""k별 로지스틱 회귀 학습·평가."""
import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from . import config


def _fit_timed(X, y):
    """동일 설정으로 N_TIMING_REPEATS회 학습해 평균 학습 시간과 마지막 모델을 반환한다."""
    times = []
    converged = True
    for _ in range(config.N_TIMING_REPEATS):
        model = LogisticRegression(**config.LR_PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            start = time.perf_counter()
            model.fit(X, y)
            times.append(time.perf_counter() - start)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False
    return model, sum(times) / len(times), converged


def run_experiments(X_train, X_test, y_train, y_test, pca, k_values, cumulative_ratio):
    """각 k에 대해 LR을 학습하고 결과 표를 반환한다.

    k == 784는 PCA를 적용하지 않은 원본 픽셀을 사용한다.
    """
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    rows = []
    for k in k_values:
        if k == config.N_FEATURES:
            Xtr, Xte, cum_var = X_train, X_test, 1.0
        else:
            Xtr, Xte = Z_train[:, :k], Z_test[:, :k]
            cum_var = float(cumulative_ratio[k - 1])

        model, fit_time, converged = _fit_timed(Xtr, y_train)
        row = {
            "k": k,
            "cum_var": cum_var,
            "test_acc": model.score(Xte, y_test),
            "train_acc": model.score(Xtr, y_train),
            "fit_time_sec": fit_time,
            "reduction_rate": 1 - k / config.N_FEATURES,
        }
        rows.append(row)
        note = "" if converged else "  (ConvergenceWarning: max_iter 도달)"
        print(f"  k={k:>3d}  R_k={cum_var:.4f}  test={row['test_acc']:.4f}  "
              f"train={row['train_acc']:.4f}  time={fit_time:.2f}s{note}", flush=True)
    return pd.DataFrame(rows)

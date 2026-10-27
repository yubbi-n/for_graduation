"""졸업논문 실험: PCA 고유값 구조에 따른 차원 축소와 분류 성능 분석.

`python main.py` 한 번으로 results/ 표와 figures/ 그림을 모두 생성한다.
"""
import random

import numpy as np
import pandas as pd

from src import classify, config, data, pca_analysis, plots


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] MNIST 로드")
    X_train, X_test, y_train, y_test = data.load_mnist()
    print(f"  train {X_train.shape}, test {X_test.shape}")

    print("[2/5] PCA fit (train only)")
    pca = pca_analysis.fit_pca(X_train)
    eig_df = pca_analysis.eigenvalue_table(pca)
    thr_df = pca_analysis.threshold_ks(eig_df["cumulative_ratio"])
    eig_df.to_csv(config.RESULTS_DIR / "eigenvalues.csv", index=False)
    thr_df.to_csv(config.RESULTS_DIR / "thresholds.csv", index=False)
    print(thr_df.to_string(index=False))

    print("[3/5] k별 로지스틱 회귀")
    k_values = sorted(set(config.K_LIST) | set(thr_df["k"].astype(int)))
    res_df = classify.run_experiments(
        X_train, X_test, y_train, y_test, pca, k_values, eig_df["cumulative_ratio"].to_numpy()
    )
    res_df.to_csv(config.RESULTS_DIR / "pca_results.csv", index=False)

    print("[4/5] 그림 생성")
    plots.plot_eigenvalues(eig_df)
    plots.plot_cumulative_variance(eig_df, thr_df)
    plots.plot_accuracy_vs_k(res_df, thr_df)
    plots.plot_variance_vs_accuracy(res_df)
    plots.plot_pca_2d(pca.transform(X_test)[:, :2], y_test)

    print("[5/5] 결과 (pca_results.csv)")
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()

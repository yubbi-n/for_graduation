"""논문용 그림 5개 (png, dpi=300, 영문 축 라벨)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

THRESHOLD_COLORS = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    return path


def plot_eigenvalues(eig_df):
    """fig1: 고유값 스펙트럼 (선형축 / 로그축)."""
    idx, lam = eig_df["index"], eig_df["eigenvalue"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log in zip(axes, [False, True]):
        ax.plot(idx, lam, color="#4c72b0", lw=1.5)
        if log:
            # 수치적으로 0에 가까운 꼬리(비양수)는 로그축에서 제외
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(lam[lam > 0].min(), 1e-12))
        ax.set_xlabel("Principal component index")
        ax.set_ylabel("Eigenvalue" + (" (log scale)" if log else ""))
        ax.set_title("Eigenvalue spectrum (" + ("log" if log else "linear") + " scale)")
        ax.set_xlim(1, len(idx))
        ax.grid(alpha=0.3, which="both" if log else "major")
    return _save(fig, "fig1_eigenvalues.png")


def plot_cumulative_variance(eig_df, thr_df):
    """fig2: 누적 설명분산 R_k와 80/90/95/99% 기준선."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eig_df["index"], eig_df["cumulative_ratio"], color="#333333", lw=1.8,
            label="Cumulative explained variance")
    for (t, k), c in zip(thr_df[["threshold", "k"]].itertuples(index=False), THRESHOLD_COLORS):
        ax.axhline(t, ls="--", lw=1, color=c, label=f"{t:.0%} (k = {int(k)})")
        ax.plot(k, eig_df["cumulative_ratio"].iloc[int(k) - 1], "o", color=c, ms=5)
    ax.set_xlabel("Number of principal components (k)")
    ax.set_ylabel("Cumulative explained variance ratio $R_k$")
    ax.set_title("Cumulative explained variance")
    ax.set_xlim(0, config.N_FEATURES)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, "fig2_cumulative_variance.png")


def plot_accuracy_vs_k(res_df, thr_df):
    """fig3: k(로그축)에 따른 test 정확도, 원본 784차원 정확도 수평선."""
    raw = res_df[res_df["k"] == config.N_FEATURES]
    pca = res_df[res_df["k"] != config.N_FEATURES].sort_values("k")
    thr_k = set(thr_df["k"].astype(int))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pca["k"], pca["test_acc"], "o-", color="#4c72b0", label="PCA + LR (test)")
    sel = pca[pca["k"].isin(thr_k)]
    ax.plot(sel["k"], sel["test_acc"], "D", color="#dd8452", ms=7, ls="none",
            label="Variance-threshold k (80/90/95/99%)")
    if not raw.empty:
        acc = float(raw["test_acc"].iloc[0])
        ax.axhline(acc, ls="--", color="#c44e52", label=f"Raw 784-dim (test = {acc:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of principal components k (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs. number of components")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    return _save(fig, "fig3_accuracy_vs_k.png")


def plot_variance_vs_accuracy(res_df):
    """fig4: 누적 설명분산 R_k와 test 정확도의 관계 (각 점에 k 주석)."""
    df = res_df.sort_values("k")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["cum_var"], df["test_acc"], "-", color="#bbbbbb", lw=1, zorder=1)
    ax.scatter(df["cum_var"], df["test_acc"], color="#4c72b0", zorder=2)
    for k, x, y in df[["k", "cum_var", "test_acc"]].itertuples(index=False):
        label = f"k={int(k)}" + (" (raw)" if k == config.N_FEATURES else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("Cumulative explained variance ratio $R_k$")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Explained variance vs. test accuracy")
    ax.grid(alpha=0.3)
    return _save(fig, "fig4_variance_vs_accuracy.png")


def plot_pca_2d(Z_test_2d, y_test):
    """fig5: test에서 무작위 5,000개 추출한 PC1–PC2 산점도."""
    rng = np.random.default_rng(config.SEED)
    n = min(config.N_SCATTER_SAMPLES, len(y_test))
    idx = rng.choice(len(y_test), size=n, replace=False)
    Z, y = Z_test_2d[idx], y_test[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        m = y == digit
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, color=cmap(digit), label=str(digit))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"MNIST test set projected on PC1-PC2 (n = {n})")
    ax.legend(title="Digit", markerscale=3, loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig5_pca_2d.png")

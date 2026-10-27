"""PCA fit(train 전용), 고유값 스펙트럼 및 누적 설명분산 임계 k 계산."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config


def fit_pca(X_train):
    """전체 784개 주성분으로 train에만 한 번 fit한다.

    주성분 k개만 쓸 때는 components_[:k]에 해당하는 열(transform 결과의 앞 k열)을 사용하며,
    이는 PCA(n_components=k)를 따로 fit한 결과와 같다.
    """
    pca = PCA(n_components=config.N_FEATURES, svd_solver="full", random_state=config.SEED)
    pca.fit(X_train)
    return pca


def eigenvalue_table(pca):
    """index, eigenvalue(λ_i), ratio(r_i), cumulative_ratio(R_k) 표."""
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame({
        "index": np.arange(1, len(ratio) + 1),
        "eigenvalue": pca.explained_variance_,
        "ratio": ratio,
        "cumulative_ratio": np.cumsum(ratio),
    })


def threshold_ks(cumulative_ratio, thresholds=config.VARIANCE_THRESHOLDS):
    """누적 설명분산 R_k가 처음으로 각 임계값 이상이 되는 k."""
    cumulative_ratio = np.asarray(cumulative_ratio)
    rows = []
    for t in thresholds:
        idx = np.flatnonzero(cumulative_ratio >= t)
        k = int(idx[0]) + 1 if idx.size else len(cumulative_ratio)
        rows.append({"threshold": t, "k": k})
    return pd.DataFrame(rows)

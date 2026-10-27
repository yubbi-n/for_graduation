# 주성분 분석의 고유값 구조에 따른 차원 축소와 분류 성능 분석

## 실행
```bash
pip install -r requirements.txt
python main.py
```
- 최초 실행 시 `fetch_openml("mnist_784")`로 MNIST를 내려받습니다 (openml.org 접속 필요, 이후 `data/`에 캐시되어 오프라인으로 재실행 가능).
- 실행이 끝나면 `results/`(CSV 3개)와 `figures/`(PNG 5개)가 생성되고 `pca_results` 표가 콘솔에 출력됩니다.

## 구성
| 파일 | 역할 |
|---|---|
| `src/config.py` | 시드(42), k 목록, LR 하이퍼파라미터, 경로 |
| `src/data.py` | MNIST 로드, 60,000/10,000 표준 분할, `/255.0` 정규화 |
| `src/pca_analysis.py` | train에만 `PCA(784, svd_solver="full")` fit, 고유값 표·임계 k(80/90/95/99%) |
| `src/classify.py` | k별 로지스틱 회귀 학습(3회 평균 시간)·평가, k=784는 PCA 미적용 원본 |
| `src/plots.py` | fig1–fig5 (dpi=300) |

## 출력
- `results/eigenvalues.csv`: index, eigenvalue, ratio, cumulative_ratio (784행)
- `results/thresholds.csv`: threshold, k
- `results/pca_results.csv`: k, cum_var, test_acc, train_acc, fit_time_sec, reduction_rate
- `figures/fig1_eigenvalues.png` … `fig5_pca_2d.png`

## 참고
- 실험 k는 {2, 5, 10, 20, 50, 100, 200, 784} ∪ {임계 k}를 정렬·중복 제거한 목록입니다.
- 주성분 k개 특징은 784개 전체로 한 번 fit한 PCA 변환 결과의 앞 k열(`components_[:k]`)이며, PCA(k)를 따로 fit한 것과 같습니다.
- LR이 `max_iter=1000` 안에 수렴하지 않으면 콘솔에 표시됩니다 (하이퍼파라미터는 통제변수로 변경하지 않음).
- `fit_time_sec`은 실행 환경(CPU, BLAS 스레드 수)에 따라 달라집니다.

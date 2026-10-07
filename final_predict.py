# -*- coding: utf-8 -*-
"""
KCB 대출 신청 데이터 TARGET 예측 — 최종 제출 코드
====================================================

이 파일 하나만으로 동작한다 (프로젝트의 다른 .py 파일이나 캐시가 필요 없다).
필요한 것은 학습 데이터 CSV와 아래 라이브러리뿐이다.

--------------------------------------------------------------------------
1. 사용법
--------------------------------------------------------------------------
(1) 함수로 호출
    from final_predict import final_predict
    pred = final_predict("test.csv")                    # 0/1 예측 (numpy 배열, 테스트 행 순서 그대로)
    pred, prob = final_predict(test_df, return_proba=True)  # DataFrame도 받는다, 확률도 함께 반환

    - test  : 테스트셋 CSV 경로 또는 DataFrame. 학습 데이터와 같은 열(TARGET 제외)이 있어야 한다.
    - train : 학습 데이터 CSV 경로 또는 DataFrame. 생략하면 이 파일과 같은 폴더의
              kcb_202306_202405_undersampled_1to4.csv 를 읽는다.

(2) 명령줄 실행
    python final_predict.py --test=test.csv [--train=학습.csv] [--out=submission.csv]
    python final_predict.py --dry-run       # 자체 검증: 2023-12~2024-05를 가짜 테스트셋으로 떼어 채점

--------------------------------------------------------------------------
2. 전체 흐름 (각 단계의 근거가 된 실험 번호는 프로젝트 폴더의 NN_*.py)
--------------------------------------------------------------------------
  [학습 12개월] + [테스트셋]  → 한 표로 합친다 (테스트 TARGET은 비워 둠)
      │
      ├─ 1) 동일인 매칭      : 고객 ID가 없으므로, 변수 값이 거의 같은 신청 건을 같은 사람으로 판정
      │                        (Fellegi-Sunter 확률적 매칭, 라벨 미사용)                  ← 08, 10
      ├─ 2) 구조 변수        : 동일인 짝 개수, 같은 사람 묶음의 집계값, 약한 유사 이웃 수 (라벨 미사용) ← 09, 11, 12
      ├─ 3) kNN 변수         : 비슷한 '과거' 고객들의 실제 TARGET 비율과 거리 (학습 라벨만 사용) ← 13, 14, 19
      ├─ 4) 모델             : LightGBM(3시드) + XGBoost(2시드) + CatBoost(1시드) 확률 평균,
      │                        최근 달일수록 큰 가중치(반감기 3개월)로 학습              ← 02, 06, 18
      └─ 5) 후처리           : 라벨 전이 → 그룹 평균 → 임계값 0.45로 0/1 결정           ← 09, 15

--------------------------------------------------------------------------
3. 성능 (시간 순서 백테스트, 대회 지표 = 2개월 단위 오분류율의 60/30/10 가중합)
--------------------------------------------------------------------------
  - 전부 0으로 예측           : 0.2130 (세 폴드 평균)
  - LightGBM 원변수만         : 0.1714
  - 이 파이프라인             : 0.1501 (세 폴드 평균), 실제 평가와 같은 조건의 시험 운행 0.1431

--------------------------------------------------------------------------
4. 가정과 주의
--------------------------------------------------------------------------
  - 테스트셋은 학습 데이터 바로 다음 기간이고, 학습 데이터와 같은 열과 LNMON(신청 연월, YYYYMM)을 가진다.
  - 테스트셋도 학습 데이터처럼 1:4 언더샘플링되어 있다고 가정한다(임계값 0.45의 전제).
    결과 출력의 '샘플링 진단'에서 월별 행 수와 짝 있는 행 비율이 학습 데이터와 크게 다르면 이 가정을 의심해야 한다.
    (29_population_sim: 언더샘플링이 없으면 양성률이 약 5%로 떨어져 임계값 0.45가 크게 불리해진다)
  - 테스트셋에 TARGET 열이 있어도 읽자마자 지우므로 예측에 쓰이지 않는다.
  - 실행 시간: 8코어 노트북 기준 약 45~60분 (매칭 ~10분, kNN ~10분, 모델 학습 ~30분). 메모리 약 4~5GB.

개발 환경 버전: Python 3.9.13, numpy 1.21.5, pandas 1.4.4, scipy 1.9.1, scikit-learn 1.0.2,
                lightgbm 4.6.0, xgboost 2.1.4, catboost 1.2.10
"""
import os
import sys
import time
import itertools
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ==========================================================================
# 0. 설정값
#    모든 숫자는 시간 순서 백테스트(세 폴드)에서 고른 값이다. 바꾸면 성능이 달라진다.
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(HERE, "kcb_202306_202405_undersampled_1to4.csv")

ID_COLS = ["LNMON", "TARGET"]  # 예측 변수로 쓰지 않는 열: 신청 연월, 정답
# 범주형 변수 4개. 나머지는 모두 수치형(float32)으로 읽는다.
CAT_COLS = ["SS1200000", "AS1C10148", "TS_VOLATILITY_APS001_FLIPS", "TS_MOMENTUM_APS001_STATE"]

THRESHOLD = 0.45   # 확률 > 0.45 이면 1. 세 폴드 모두 최적 구간이 0.425~0.475 → 가운데 값
HALF_LIFE = 3      # 학습 행 가중치 = 0.5 ** (학습 마지막 달로부터 개월 수 / 3). 최근 데이터를 더 믿는다 (02)

# --- 동일인 매칭 ---
MAX_GAP = 2        # 몇 개월 차이까지 같은 사람의 재신청으로 볼지. 그보다 먼 재신청은 데이터에 거의 없었다
MAX_BUCKET = 8     # LSH 버킷에 이보다 많은 행이 모이면 '흔한 값'이라 보고 건너뛴다 (후보 폭증 방지)
M_CLIP = (0.5, 0.999)  # Fellegi-Sunter m 확률(동일인일 때 일치 확률)의 하한·상한
F_FLOOR = 1e-5     # 값 빈도의 하한 (로그 계산 안정화)
FS_THR = 300       # 매칭 점수가 이보다 크면 같은 사람. 점수 분포의 골짜기, 150~500에서 성능 차이 < 0.13%p (22)

# --- 구조 변수 ---
N_CHANGING = 12    # 같은 사람의 신청 건마다 자주 바뀌는 변수 상위 12개로 그룹 집계 변수를 만든다
WEAK_BUCKET = 6    # 약한 이웃용 LSH 버킷 상한 (동일인 매칭보다 조금 엄격)

# --- kNN ---
KNN_DIM = 180      # 거리 공간에 쓰는 변수 수. 학습 기간이 길수록 클수록 좋았다 (19: 10~200 탐색)
KNN_KS = [10, 50, 200]  # 이웃 수별 TARGET 평균
KNN_EXTRA = 10     # 같은 사람 이웃을 빼고도 200명이 남도록 여유로 더 찾는 수
KNN_FEATS = ["KNN_MEAN_k%d" % k for k in KNN_KS] + ["KNN_WMEAN_k50", "KNN_DIST1", "KNN_DIST_k10",
                                                     "KNN_DIST_POS", "KNN_DIST_NEG"]

# --- 모델 하이퍼파라미터 ---
# kNN 거리 공간의 변수 중요도를 재는 데 쓰는 기본 LightGBM 설정
LGB_BASE = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                verbose=-1, num_threads=8)
# 최종 LightGBM: Optuna 40회 탐색에서 고른 설정 (06), 1100회 반복
LGB_TUNED = dict(objective="binary", learning_rate=0.05, num_leaves=35, min_child_samples=50,
                 feature_fraction=0.52, bagging_fraction=0.957, bagging_freq=1, lambda_l1=0.001,
                 lambda_l2=11.31, min_gain_to_split=0.17, verbose=-1, num_threads=8)
LGB_ROUNDS = 1100
XGB_P = dict(objective="binary:logistic", tree_method="hist", eta=0.05, max_depth=6, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.5, reg_lambda=10, max_bin=256, nthread=8)
XGB_ROUNDS = 1200
CAT_P = dict(loss_function="Logloss", learning_rate=0.08, depth=6, l2_leaf_reg=10, rsm=0.5,
             bootstrap_type="Bernoulli", subsample=0.8, border_count=254, nan_mode="Min",
             thread_count=8, verbose=0)
CAT_ROUNDS = 2000

_T0 = time.time()


def log(msg, verbose=True):
    """경과 시간과 함께 진행 상황을 출력한다."""
    if verbose:
        print("[%5ds] %s" % (time.time() - _T0, msg), flush=True)


# ==========================================================================
# 1. 데이터 읽기
# ==========================================================================
def load_train(train):
    """학습 데이터를 읽고 형식을 맞춘다.

    - 범주형 4개는 pandas category, 나머지 변수는 float32 (메모리를 절반으로 줄이고, 모든 단계에서 같은 형식을 쓰기 위해)
    - LNMON(YYYYMM 정수), TARGET(0/1)은 그대로 둔다.
    """
    df = pd.read_csv(train, encoding="utf-8-sig", low_memory=False) if isinstance(train, str) else train.copy()
    for c in df.columns:
        if c in ID_COLS:
            continue
        df[c] = df[c].astype("category") if c in CAT_COLS else df[c].astype(np.float32)
    return df.reset_index(drop=True)


def prepare_test(test, train):
    """테스트셋을 학습 데이터와 같은 열 순서·형식으로 맞춘다.

    - 학습 데이터에 있는 열이 빠져 있으면 어떤 열인지 알려주고 멈춘다.
    - 학습 데이터에 없는 열은 버린다.
    - TARGET은 있든 없든 결측으로 만든다 → 테스트 정답이 어떤 경로로도 예측에 들어가지 않는다.
    - 범주형은 학습 데이터의 범주 목록을 그대로 쓴다 (처음 보는 범주는 결측이 된다).
    """
    te = pd.read_csv(test, encoding="utf-8-sig", low_memory=False) if isinstance(test, str) else test.copy()
    missing = [c for c in train.columns if c not in te.columns and c != "TARGET"]
    if missing:
        raise ValueError("테스트셋에 없는 열 %d개: %s ..." % (len(missing), missing[:5]))
    te["TARGET"] = np.nan
    te = te[train.columns].reset_index(drop=True)
    for c in te.columns:
        if c in CAT_COLS:
            te[c] = pd.Categorical(te[c], categories=train[c].cat.categories)
        elif c not in ID_COLS:
            te[c] = te[c].astype(np.float32)
    return te


def feature_cols(df):
    """예측에 쓸 변수 목록: 식별자(LNMON, TARGET)와 값이 하나뿐인 상수 열을 뺀 나머지."""
    nun = df.nunique()
    return [c for c in df.columns if c not in ID_COLS and nun[c] > 1]


def month_index(lnmon):
    """YYYYMM → 연속된 월 번호. 202312 → 24288, 202401 → 24289 처럼 해가 바뀌어도 차이가 1이 된다."""
    lnmon = np.asarray(lnmon)
    return (lnmon // 100) * 12 + lnmon % 100


# ==========================================================================
# 2. 동일인 매칭 (라벨을 쓰지 않는다)
# --------------------------------------------------------------------------
# 발견: 고객 ID는 없지만, 거의 모든 변수 값이 같은 신청 건이 같은 달이나 인접한 달에 반복해서 나온다.
#       짧은 기간에 대출을 여러 번 신청한 같은 사람으로 보인다.
#       이렇게 판정한 쌍은 TARGET이 99.9% 일치했다 → TARGET은 사람 단위로 정해진다.
# 방법: 수십만 행의 모든 쌍(수백억 개)을 비교할 수 없으므로
#   (1) LSH: 무작위로 고른 변수 몇 개의 값이 완전히 같은 행끼리만 후보 쌍으로 모은다.
#   (2) 1차: 값 종류가 많은 변수로 후보를 모아, 전체 변수의 90% 이상이 같은 쌍을 '확실한 쌍'으로 본다.
#   (3) 확실한 쌍에서 거의 바뀌지 않는 변수(불일치 < 2%)를 '안정 변수'로 고르고, 이 변수로 후보를 다시 모은다.
#   (4) 모든 후보 쌍에 Fellegi-Sunter 점수를 매긴다. 흔한 값(예: 0)이 같은 것보다 드문 값이 같은 것을
#       더 강한 증거로 본다. 점수 > 300 인 쌍을 같은 사람으로 판정한다.
# ==========================================================================
def lsh_pairs(df, cols, n_bands, k, seed, max_bucket=MAX_BUCKET):
    """LSH 후보 쌍 찾기.

    n_bands번 반복: cols 중 k개 변수를 무작위로 골라, 그 k개 값이 모두 같은 행들을 한 버킷으로 묶는다.
    버킷 크기가 2 이상 max_bucket 이하이면 버킷 안의 모든 쌍을 후보로 추가한다.
    (결측은 groupby에서 빠지므로 결측끼리는 버킷이 되지 않는다)
    seed를 고정해 매번 같은 결과가 나온다.
    """
    rng = np.random.default_rng(seed)
    pairs = set()
    for _ in range(n_bands):
        band = list(rng.choice(cols, k, replace=False))
        for idx in df.groupby(band, sort=False).indices.values():
            if 1 < len(idx) <= max_bucket:
                pairs.update(itertools.combinations(idx, 2))
    return np.array(sorted(pairs))


def match_rate(X, pairs):
    """쌍마다 변수 값이 같은 비율 (둘 다 결측이면 같다고 본다)."""
    a, b = X[pairs[:, 0]], X[pairs[:, 1]]
    return ((a == b) | (np.isnan(a) & np.isnan(b))).mean(1)


def find_candidate_pairs(df):
    """(2)~(3) 단계: 1차·2차 LSH로 후보 쌍을 모으고 안정 변수 목록을 정한다.

    반환: 후보 쌍 표(i, j, gap=월 차이, sim_stable=안정 변수 일치율, sim_all=전체 일치율), 안정 변수 목록
    """
    feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    X = df[feats].values.astype(np.float64)
    mi = month_index(df.LNMON.values)
    nun = df[feats].nunique()
    # 0이거나 결측인 행이 30% 미만인 변수만 키로 쓴다 (대부분 0인 변수는 '같다'가 아무 정보도 주지 않는다)
    informative = ((X == 0) | np.isnan(X)).mean(0) < 0.3

    # 1차: 값 종류가 500개 넘는 변수 3개 조합 × 40번
    hc = [c for c, ok in zip(feats, informative) if ok and nun[c] > 500]
    p1 = lsh_pairs(df, hc, 40, 3, 0)
    sure = p1[match_rate(X, p1) > 0.9]

    # 확실한 쌍에서 불일치율 2% 미만인 변수 = 같은 사람이면 거의 안 바뀌는 '안정 변수'
    a, b = X[sure[:, 0]], X[sure[:, 1]]
    col_diff = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), feats)
    stable = [c for c in feats if col_diff[c] < 0.02]
    stable_keys = [c for c, ok in zip(feats, informative) if ok and nun[c] > 100 and c in set(stable)]

    # 2차: 안정 변수 4개 조합 × 80번
    p2 = lsh_pairs(df, stable_keys, 80, 4, 1)
    p = np.unique(np.vstack([p1, p2]), axis=0)
    gap = np.abs(mi[p[:, 0]] - mi[p[:, 1]])
    p, gap = p[gap <= MAX_GAP], gap[gap <= MAX_GAP]
    Xs = df[stable].values.astype(np.float64)
    out = pd.DataFrame(dict(i=p[:, 0], j=p[:, 1], gap=gap, sim_stable=match_rate(Xs, p), sim_all=match_rate(X, p)))
    return out, stable


def fs_weights(df, feats, sure_pairs):
    """Fellegi-Sunter 가중치 추정.

    - m_k  : 같은 사람일 때 변수 k가 일치할 확률. 확실한 쌍에서 직접 센다.
    - f_k(v): 값 v의 전체 빈도. 흔한 값이 일치하는 것은 약한 증거다.
    - u_k  : 무작위 두 행이 변수 k에서 일치할 확률 = Σ_v f_k(v)^2
    결측끼리는 일치로 보기 위해 결측을 아주 작은 수(-1e18)로 바꿔 둔다.
    """
    X = df[feats].values.astype(np.float64)
    Xn = np.where(np.isnan(X), -1e18, X)
    a, b = Xn[sure_pairs[:, 0]], Xn[sure_pairs[:, 1]]
    m = np.clip((a == b).mean(0), *M_CLIP)
    freq_maps, u = [], np.zeros(len(feats))
    for k in range(len(feats)):
        vc = pd.Series(Xn[:, k]).value_counts(normalize=True)
        freq_maps.append(vc)
        u[k] = float((vc.values ** 2).sum())
    return Xn, m, u, freq_maps


def fs_score(Xn, pairs, m, u, freq_maps):
    """쌍별 Fellegi-Sunter 점수 = Σ_k [일치하면 log(m_k / f_k(v)), 다르면 log((1 - m_k) / (1 - u_k))].

    값이 클수록 같은 사람일 가능성이 높다. 드문 값이 같으면 크게 더하고, 다르면 뺀다.
    """
    a, b = Xn[pairs[:, 0]], Xn[pairs[:, 1]]
    agree = a == b
    score = np.zeros(len(pairs))
    for k in range(Xn.shape[1]):
        f = np.maximum(freq_maps[k].reindex(a[:, k]).values, F_FLOOR)
        w_agree = np.log(m[k] / f)
        w_dis = np.log((1 - m[k]) / max(1 - u[k], 1e-6))
        score += np.where(agree[:, k], w_agree, w_dis)
    return score


def find_fs_pairs(df):
    """(4) 단계: 후보를 더 넓게 모은 뒤 모든 후보에 Fellegi-Sunter 점수를 매긴다.

    반환: 후보 쌍 표(i, j, gap, fs=점수, sim_stable). 이 중 fs > FS_THR 인 쌍이 '같은 사람'이다.
    """
    df = df.reset_index(drop=True)
    base, stable = find_candidate_pairs(df)
    sure = base[(base.sim_stable > 0.95)][["i", "j"]].values  # 가중치 추정용 '아주 확실한 쌍'
    feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    nun = df[feats].nunique()
    X = df[feats].values
    informative = ((X == 0) | np.isnan(X)).mean(0) < 0.3
    keys = [c for c, ok in zip(feats, informative) if ok and nun[c] > 100 and c in set(stable)]
    extra = lsh_pairs(df, keys, 200, 3, 2)  # 안정 변수 3개 조합 × 200번으로 후보를 넓게 추가
    mi = month_index(df.LNMON.values)
    cand = np.unique(np.vstack([base[["i", "j"]].values, extra]), axis=0)
    gap = np.abs(mi[cand[:, 0]] - mi[cand[:, 1]])
    cand, gap = cand[gap <= MAX_GAP], gap[gap <= MAX_GAP]
    Xn, m, u, fm = fs_weights(df, feats, sure)
    out = pd.DataFrame(dict(i=cand[:, 0], j=cand[:, 1], gap=gap, fs=fs_score(Xn, cand, m, u, fm)))
    return out.merge(base[["i", "j", "sim_stable"]], on=["i", "j"], how="left")


def components(n, pairs):
    """쌍을 간선으로 보고 연결 성분 번호를 매긴다. A-B, B-C가 짝이면 A, B, C는 한 사람(한 그룹)."""
    g = coo_matrix((np.ones(len(pairs)), (pairs.i.values, pairs.j.values)), shape=(n, n))
    return connected_components(g, directed=False)[1]


# ==========================================================================
# 3. 구조 변수 (라벨을 쓰지 않으므로 학습·테스트 행 모두 같은 방식으로 만든다)
# --------------------------------------------------------------------------
# 왜 효과가 있나: 데이터는 양성을 모두 남기고 음성을 약 22%만 뽑은 1:4 언더샘플링 표본으로 보인다.
#   양성 고객의 신청 건은 모두 남지만, 음성 고객의 신청 건 두 개가 함께 남을 확률은 낮다.
#   그래서 표본 안에서는 '짝이 있으면 양성일 가능성이 높다' (짝 있는 행의 양성률 44~87% vs 평균 20%).
#   이 효과는 표본 설계에서 나온 것이라, 테스트셋도 같은 방식으로 만들어졌다는 가정이 필요하다.
# 효과: 세 폴드 평균 오분류율 0.1700 → 0.1612 (09, 11), 약한 이웃까지 0.1592 (12)
# ==========================================================================
DUP_FEATS = ["DUP_SAME", "DUP_PREV", "DUP_NEXT", "DUP_GAP2", "DUP_MAXSIM"]
WEAK_FEATS = ["WEAK_SAME", "WEAK_ADJ", "WEAK_FAR", "MAXSIM_STABLE"]


def column_dynamics(df, pairs_fs):
    """변수별로 '같은 사람의 한 달 뒤 신청에서 값이 그대로인 비율'(eq0_g1)과 값 종류 수(nun)를 계산한다.

    eq0_g1 > 0.97 인 변수 = 시간이 지나도 거의 안 바뀌는 '안정 변수' → 약한 이웃 찾기에 쓴다.
    (동일인 쌍 중 월 차이 1인 쌍만 보고, 두 값이 모두 있는 경우만 센다. 라벨 미사용)
    """
    num = [c for c in feature_cols(df) if c not in CAT_COLS]
    p = pairs_fs[(pairs_fs.fs > FS_THR) & (pairs_fs.gap == 1)]
    X = df[num].values.astype(np.float64)
    a, b = X[p.i.values], X[p.j.values]
    both = ~(np.isnan(a) | np.isnan(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        eq = np.where(both, a == b, False).sum(0) / both.sum(0)
    return pd.DataFrame(dict(nun=df[num].nunique().values, eq0_g1=eq), index=num)


def add_dup_features(df, pairs):
    """행마다 동일인 짝 개수(같은 달 / 이전 달 / 다음 달 / 2개월 차이)와 최대 매칭 점수.

    '다음 달' 짝도 쓴다: 테스트셋은 6개월치를 한 번에 받으므로, 테스트 행의 다음 달 신청도 테스트셋 안에 있다.
    """
    n = len(df)
    mi = month_index(df.LNMON.values)
    out = {c: np.zeros(n, np.float32) for c in DUP_FEATS}
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:  # 양방향으로 센다
        d = mi[b] - mi[a]
        np.add.at(out["DUP_SAME"], a[d == 0], 1)
        np.add.at(out["DUP_PREV"], a[d == -1], 1)
        np.add.at(out["DUP_NEXT"], a[d == 1], 1)
        np.add.at(out["DUP_GAP2"], a[np.abs(d) == 2], 1)
        np.maximum.at(out["DUP_MAXSIM"], a, pairs["fs"].values.astype(np.float32))
    for c, v in out.items():
        df[c] = v
    return df


def build_structure_features(df, pairs_fs, dynamics):
    """구조 변수 세 묶음을 만든다.

    - dup  (5개) : add_dup_features
    - grp  (39개): 같은 사람 묶음의 크기, 신청한 달 수, 몇 번째 달인지,
                   신청마다 자주 바뀌는 변수 12개 각각의 묶음 내 최대·최소·평균과의 차이
                   (사기 탐지에서 쓰는 '사용자 단위 집계'를 응용)
    - weak (4개) : 안정 변수 일치율이 0.8~0.9인 '같은 사람까지는 아닌 비슷한 사람'의 수(같은 달/인접 달/먼 달)와
                   동일인을 뺀 최대 유사도. 이런 쌍도 무작위 쌍보다 양성 쌍이 4배 많았다 (12)
    반환: 변수가 추가된 df, 동일인 쌍(fs > 300), 행별 그룹 번호, 변수 묶음 이름
    """
    num_feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    mi = month_index(df.LNMON.values)
    pairs = pairs_fs[pairs_fs.fs > FS_THR][["i", "j", "gap", "fs"]].reset_index(drop=True)

    # --- dup ---
    df = add_dup_features(df, pairs)

    # --- grp ---
    comp = components(len(df), pairs)
    g = pd.DataFrame(dict(comp=comp, mi=mi))
    new = {"GRP_SIZE": g.groupby("comp").comp.transform("size"),        # 같은 사람의 신청 건수
           "GRP_NMONTHS": g.groupby("comp").mi.transform("nunique"),    # 신청한 서로 다른 달 수
           "GRP_ORDER": mi - g.groupby("comp").mi.transform("min")}     # 첫 신청 이후 몇 개월째인지
    X = df[num_feats].values
    a, b = X[pairs.i.values], X[pairs.j.values]
    differ = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), num_feats)  # 동일인 쌍에서 값이 다른 비율
    inf50 = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.5, num_feats)                 # 절반 이상이 0/결측인 변수 제외
    changing = differ[inf50].sort_values(ascending=False).index[:N_CHANGING].tolist()
    for c in changing:
        s = df[c].groupby(comp)
        new["GMAX_" + c] = s.transform("max")
        new["GMIN_" + c] = s.transform("min")
        new["GDIFF_" + c] = df[c] - s.transform("mean")
    grp_feats = list(new)

    # --- weak ---
    inf30 = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.3, num_feats)
    keys = [c for c in dynamics[(dynamics.eq0_g1 > 0.97) & (dynamics.nun > 200)].index if inf30.get(c, False)]
    stable = [c for c in dynamics[dynamics.eq0_g1 > 0.97].index if c in df.columns]
    cand = lsh_pairs(df, keys, 150, 2, 5, max_bucket=WEAK_BUCKET)
    gap = np.abs(mi[cand[:, 0]] - mi[cand[:, 1]])
    sim = match_rate(df[stable].values.astype(np.float64), cand)
    weak = (sim > 0.8) & (sim <= 0.9)
    w = {c: np.zeros(len(df), np.float32) for c in WEAK_FEATS}
    for sel, col in [(weak & (gap == 0), "WEAK_SAME"), (weak & (gap == 1), "WEAK_ADJ"), (weak & (gap >= 2), "WEAK_FAR")]:
        for side in (0, 1):
            np.add.at(w[col], cand[sel, side], 1)
    s_nodup = np.where(sim > 0.9, 0, sim).astype(np.float32)  # 일치율 0.9 초과(=동일인)는 빼고 최대값
    for side in (0, 1):
        np.maximum.at(w["MAXSIM_STABLE"], cand[:, side], s_nodup)
    new.update(w)

    df = pd.concat([df, pd.DataFrame({k: np.asarray(v, np.float32) for k, v in new.items()}, index=df.index)], axis=1)
    return df, pairs, comp, dict(dup=DUP_FEATS, grp=grp_feats, weak=WEAK_FEATS)


# ==========================================================================
# 4. kNN 변수 (학습 기간 라벨만 사용)
# --------------------------------------------------------------------------
# 생각: LightGBM은 변수 공간을 축에 평행한 직사각형으로 나눈다. kNN은 '여러 변수가 조금씩 다 비슷한'
#       둥근 이웃을 본다. 서로 다른 것을 보므로 보완된다.
# 방법: 중요 변수 180개를 순위(0~1)로 바꾼 공간에서 가장 가까운 학습 고객 10/50/200명의 TARGET 평균과 거리.
# 누수 방지:
#   - 테스트 행: 학습 기간 전체에서 이웃을 찾는다.
#   - 학습 행  : 자기보다 '이전 달' 학습 행에서만 찾는다 (past 방식). 학습과 테스트가 모두
#                '과거만 본다'는 같은 조건이 되어, 학습 때의 변수 분포가 테스트와 같아진다 (14).
#   - 같은 사람(동일인 그룹)의 다른 신청 건은 이웃에서 뺀다 (자기 정답을 보는 것과 같으므로).
# 효과: 세 폴드 평균 0.1592 → 0.1491 (13, 14, 19)
# ==========================================================================
def knn_distance_space(df, feats, train_mask, n_dim):
    """학습 행으로 LightGBM을 300회 학습해 중요도(정보이득) 상위 n_dim개 수치 변수를 고르고,
    각 변수를 전체 행 기준 순위(0~1)로 바꾼다. 결측은 -0.25 (결측끼리 가깝고, 값이 있는 행과는 멀게).
    순위로 바꾸면 단위와 이상치의 영향이 사라진다.
    """
    sel = df[train_mask]
    m = lgb.train(dict(LGB_BASE, seed=0, learning_rate=0.05), lgb.Dataset(sel[feats], sel.TARGET.values), 300)
    imp = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
    dims = [c for c in imp.index if str(df[c].dtype) != "category"][:n_dim]
    Z = np.column_stack([df[c].rank(pct=True).fillna(-0.25).values for c in dims]).astype(np.float32)
    return Z


def knn_summarize(dist, idx, y_ref, valid):
    """찾은 이웃(거리, 번호)에서 같은 사람이 아닌 이웃만 앞으로 모아 8개 변수를 만든다.

    - KNN_MEAN_k10/50/200 : 가까운 10/50/200명의 TARGET 평균
    - KNN_WMEAN_k50       : 50명의 거리 역수 가중 평균
    - KNN_DIST1, KNN_DIST_k10 : 가장 가까운 이웃까지 거리, 10명 평균 거리 (이웃이 촘촘한 영역인지)
    - KNN_DIST_POS / NEG  : 가장 가까운 양성 / 음성 이웃까지 거리 (없으면 최대 거리 × 1.5)
    """
    kmax = max(KNN_KS)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :kmax]   # 유효한 이웃을 거리 순서 그대로 앞으로
    d = np.take_along_axis(dist, order, 1)
    yy = y_ref[np.take_along_axis(idx, order, 1)].astype(np.float32)
    ok = np.take_along_axis(valid, order, 1)
    yy = np.where(ok, yy, np.nan)
    d = np.where(ok, d, np.nan)
    out = {}
    for k in KNN_KS:
        out["KNN_MEAN_k%d" % k] = np.nanmean(yy[:, :k], 1)
    wts = 1.0 / (d[:, :50] + 1e-3)
    out["KNN_WMEAN_k50"] = np.nansum(yy[:, :50] * wts, 1) / np.nansum(np.where(np.isnan(yy[:, :50]), 0, wts), 1)
    out["KNN_DIST1"] = d[:, 0]
    out["KNN_DIST_k10"] = np.nanmean(d[:, :10], 1)
    big = np.nanmax(d, 1, keepdims=True) * 1.5
    out["KNN_DIST_POS"] = np.nanmin(np.where(yy == 1, d, big), 1)
    out["KNN_DIST_NEG"] = np.nanmin(np.where(yy == 0, d, big), 1)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in out.items()})


def knn_query(Z, y, comp, ref, q):
    """q 행들의 이웃을 ref 행들 중에서 찾는다 (유클리드 거리, 전수 탐색). 같은 그룹(comp) 이웃은 무효 처리."""
    kq = max(KNN_KS) + KNN_EXTRA
    k = min(kq, len(ref))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=8).fit(Z[ref])
    dist, nb = nn.kneighbors(Z[q])
    if k < kq:  # 참조 행이 모자라면 빈 칸을 무효 이웃으로 채운다
        pad = kq - k
        dist = np.hstack([dist, np.full((len(q), pad), np.inf)])
        nb = np.hstack([nb, np.zeros((len(q), pad), int)])
    valid = (comp[ref][nb] != comp[q][:, None]) & np.isfinite(dist)
    return knn_summarize(dist, nb, y[ref], valid)


def knn_features(Z, y, comp, lnmon, tr_idx, te_idx):
    """past 방식 kNN 변수. 학습 기간 첫 달 행은 과거 이웃이 없어 결측으로 남는다 (LightGBM이 결측을 처리)."""
    tr_feat = pd.DataFrame(np.full((len(tr_idx), len(KNN_FEATS)), np.nan, np.float32), columns=KNN_FEATS)
    months = np.sort(np.unique(lnmon[tr_idx]))
    for mth in months[1:]:
        q_pos = np.where(lnmon[tr_idx] == mth)[0]
        ref = tr_idx[lnmon[tr_idx] < mth]
        tr_feat.iloc[q_pos] = knn_query(Z, y, comp, ref, tr_idx[q_pos]).values
    te_feat = knn_query(Z, y, comp, tr_idx, te_idx)
    return tr_feat, te_feat


# ==========================================================================
# 5. 모델 (부스팅 3종 평균)
# --------------------------------------------------------------------------
# - 변수 하나하나의 신호는 약하고(단변량 AUC 최대 0.60) 변수 간 상호작용에서 성능이 나온다 → 나무 부스팅.
# - 세 모델의 단독 성능은 거의 같지만(차이 0.05%p 이내) 오류가 조금씩 달라, 단순 평균이
#   세 폴드 모두에서 나빠지지 않았다 (18, 평균 −0.10%p). CatBoost가 가장 다른 오류를 낸다.
# - 결측은 채우지 않는다. 이 데이터의 결측은 '그 상품·이벤트가 없음'이라는 정보이고, 부스팅이 분할마다
#   결측을 어느 쪽으로 보낼지 학습한다.
# - 학습 행 가중치: 학습 마지막 달에 가까울수록 크게 (반감기 3개월). 시간에 따라 분포가 변하기 때문 (02).
# ==========================================================================
def train_and_predict(tr, te, feats, w, verbose=True):
    """세 모델을 학습하고 테스트 확률을 반환한다 (모델마다 시드 평균 → 모델 간 단순 평균)."""
    preds = {}

    # LightGBM: 시드 3개
    ps = []
    for s in range(3):
        m = lgb.train(dict(LGB_TUNED, seed=s), lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), LGB_ROUNDS)
        ps.append(m.predict(te[feats]))
        log("LightGBM seed %d 완료" % s, verbose)
    preds["lgb"] = np.mean(ps, 0)

    # XGBoost: 시드 2개. 범주형은 XGBoost의 범주형 지원을 그대로 쓴다
    dtr = xgb.DMatrix(tr[feats], tr.TARGET.values, weight=w, enable_categorical=True)
    dte = xgb.DMatrix(te[feats], enable_categorical=True)
    ps = []
    for s in range(2):
        ps.append(xgb.train(dict(XGB_P, seed=s), dtr, XGB_ROUNDS).predict(dte))
        log("XGBoost seed %d 완료" % s, verbose)
    preds["xgb"] = np.mean(ps, 0)

    # CatBoost: 시드 1개. 범주형은 문자열로 바꿔 CatBoost 자체 인코딩을 쓴다
    cats = [c for c in feats if c in CAT_COLS]

    def pool(d, lab=None, wt=None):
        X = d[feats].copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return Pool(X, lab, weight=wt, cat_features=cats)
    cm = CatBoostClassifier(iterations=CAT_ROUNDS, random_seed=0, **CAT_P).fit(pool(tr, tr.TARGET.values, w))
    preds["cat"] = cm.predict_proba(pool(te))[:, 1]
    log("CatBoost seed 0 완료", verbose)

    return np.mean([preds["lgb"], preds["xgb"], preds["cat"]], 0), preds


# ==========================================================================
# 6. 후처리 (같은 사람은 TARGET이 같다는 성질을 직접 쓴다)
# ==========================================================================
def label_transfer(p, te_idx, pairs, y, lnmon, train_end):
    """테스트 행의 동일인 짝이 학습 기간에 있으면, 그 짝들의 실제 TARGET 평균으로 예측을 덮어쓴다.
    (테스트 첫 달 행이 학습 마지막 달의 신청과 이어지는 경우가 대부분. 폴드당 약 150행)
    """
    p = p.copy()
    pos = pd.Series(np.arange(len(te_idx)), index=te_idx)
    s, c = np.zeros(len(te_idx)), np.zeros(len(te_idx))
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:
        k = np.isin(a, te_idx) & (lnmon[b] <= train_end)
        np.add.at(s, pos[a[k]].values, y[b[k]])
        np.add.at(c, pos[a[k]].values, 1)
    hit = c > 0
    p[hit] = s[hit] / c[hit]
    return p, int(hit.sum())


def group_average(p, te_idx, pairs):
    """테스트 기간 안에서 같은 사람으로 연결된 행끼리 예측확률을 평균한다 (한 사람에게 하나의 답)."""
    pos = pd.Series(np.arange(len(te_idx)), index=te_idx)
    k = np.isin(pairs.i.values, te_idx) & np.isin(pairs.j.values, te_idx)
    if not k.any():
        return p, 0
    sub = pd.DataFrame(dict(i=pos[pairs.i.values[k]].values, j=pos[pairs.j.values[k]].values))
    comp = components(len(p), sub)
    avg = pd.Series(p).groupby(comp).transform("mean").values
    return avg, int((np.bincount(comp)[comp] > 1).sum())


# ==========================================================================
# 7. 진단 출력 (예측에는 영향 없음)
# ==========================================================================
def print_diagnostics(df, tr_idx, te_idx, pairs, prob, pred):
    """테스트셋이 학습 데이터와 같은 방식으로 샘플링됐는지 라벨 없이 점검한다.

    학습 데이터 기준: 월 약 12,000~14,700행, 짝 있는 행 약 4.4%, 평균 예측확률 약 0.15~0.2.
    - 월별 행 수가 약 4배(5만 행 안팎) → 언더샘플링되지 않은 데이터일 가능성 (임계값 0.45가 불리)
    - 짝 있는 행이 거의 0% → 사람 단위로 한 건씩 뽑은 데이터 (이 경우는 손해 없음을 확인, 15)
    """
    has = np.zeros(len(df), bool)
    has[np.r_[pairs.i.values, pairs.j.values]] = True
    ln = df.LNMON.values
    te = pd.DataFrame(dict(LNMON=ln[te_idx], prob=prob, pred=pred, has_pair=has[te_idx]))
    diag = te.groupby("LNMON").agg(rows=("pred", "size"), pred_pos_rate=("pred", "mean"),
                                   mean_prob=("prob", "mean"), pair_rate=("has_pair", "mean"))
    tr = pd.DataFrame(dict(LNMON=ln[tr_idx], has_pair=has[tr_idx]))
    print("\n[샘플링 진단] 테스트셋 월별")
    print(diag.round(4).to_string())
    print("비교: 학습 데이터 월평균 행 수 %.0f, 짝 있는 행 비율 %.3f" %
          (tr.groupby("LNMON").size().mean(), tr.has_pair.mean()))
    ratio = diag.rows.mean() / tr.groupby("LNMON").size().mean()
    if ratio > 2:
        print("주의: 테스트셋 월별 행 수가 학습 데이터의 %.1f배입니다. 언더샘플링되지 않은 데이터일 수 있습니다." % ratio)


# ==========================================================================
# 8. 최종 예측 함수
# ==========================================================================
def final_predict(test, train=DEFAULT_TRAIN, threshold=THRESHOLD, return_proba=False, verbose=True):
    """테스트셋의 TARGET(0/1)을 예측한다.

    Parameters
    ----------
    test : str 또는 DataFrame
        테스트셋 CSV 경로 또는 DataFrame. 학습 데이터와 같은 열(TARGET 제외)과 LNMON이 있어야 한다.
    train : str 또는 DataFrame
        학습 데이터(2023-06~2024-05). 기본값은 이 파일과 같은 폴더의 원본 CSV.
    threshold : float
        확률이 이 값보다 크면 1로 판정한다. 기본 0.45.
    return_proba : bool
        True면 (0/1 예측, 확률)을 함께 반환한다.

    Returns
    -------
    numpy.ndarray (테스트 행 순서 그대로의 0/1), return_proba=True면 (예측, 확률) 튜플
    """
    global _T0
    _T0 = time.time()

    # ---- 0) 학습·테스트를 한 표로 합친다 -------------------------------------------------
    # 동일인 매칭과 구조 변수는 라벨을 쓰지 않으므로 학습·테스트 행을 함께 보고 만든다.
    # (테스트셋 안의 같은 사람, 학습-테스트 경계를 넘는 같은 사람을 찾기 위해)
    train_df = load_train(train)
    test_df = prepare_test(test, train_df)
    T = int(train_df.LNMON.max())  # 학습 마지막 달. 테스트는 그 이후
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["TARGET"] = df.TARGET.astype(np.float32)  # 테스트 행은 NaN
    n_tr = len(train_df)
    tr_idx, te_idx = np.arange(n_tr), np.arange(n_tr, len(df))
    base_feats = feature_cols(df)
    log("학습 %d행 (~%d), 테스트 %d행, 테스트 월 %s, 원변수 %d개" %
        (n_tr, T, len(test_df), sorted(test_df.LNMON.unique()), len(base_feats)), verbose)

    # ---- 1) 동일인 매칭 ------------------------------------------------------------------
    pairs_fs = find_fs_pairs(df)
    log("동일인 후보 %d쌍, 그중 점수 > %d: %d쌍" % (len(pairs_fs), FS_THR, (pairs_fs.fs > FS_THR).sum()), verbose)

    # ---- 2) 구조 변수 --------------------------------------------------------------------
    dynamics = column_dynamics(df, pairs_fs)
    df, pairs, comp, groups = build_structure_features(df, pairs_fs, dynamics)
    log("구조 변수 %d개 (dup %d, grp %d, weak %d)" % (sum(map(len, groups.values())), len(groups["dup"]),
                                                     len(groups["grp"]), len(groups["weak"])), verbose)

    # ---- 3) kNN 변수 ---------------------------------------------------------------------
    # y의 테스트 부분(NaN)은 0으로 채우지만, 테스트 행은 이웃의 '참조'로 쓰이지 않으므로 값이 쓰이지 않는다.
    y = np.nan_to_num(df.TARGET.values.astype(np.float64))
    ln = df.LNMON.values
    Z = knn_distance_space(df, base_feats, np.isin(np.arange(len(df)), tr_idx), KNN_DIM)
    trk, tek = knn_features(Z, y, comp, ln, tr_idx, te_idx)
    df = pd.concat([df, pd.concat([trk, tek], ignore_index=True)], axis=1)
    feats = base_feats + groups["dup"] + groups["grp"] + groups["weak"] + KNN_FEATS
    log("kNN 변수 완료, 최종 변수 %d개" % len(feats), verbose)

    # ---- 4) 모델 학습과 예측 ---------------------------------------------------------------
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values  # 학습 마지막 달로부터 개월 수
    w = 0.5 ** (age / HALF_LIFE)
    p_raw, _ = train_and_predict(tr, te, feats, w, verbose)

    # ---- 5) 후처리 -----------------------------------------------------------------------
    p, n_lt = label_transfer(p_raw, te_idx, pairs, y, ln, T)
    p, n_grp = group_average(p, te_idx, pairs)
    pred = (p > threshold).astype(int)
    log("라벨 전이 %d행, 그룹 평균 %d행, 임계값 %.3f, 예측 양성 비율 %.3f" % (n_lt, n_grp, threshold, pred.mean()), verbose)

    if verbose:
        print_diagnostics(df, tr_idx, te_idx, pairs, p, pred)
    return (pred, p) if return_proba else pred


# ==========================================================================
# 9. 명령줄 실행과 자체 검증
# ==========================================================================
def weighted_error(lnmon, y, pred):
    """대회 지표: 테스트 기간을 2개월씩 나눈 오분류율의 60/30/10 가중합 (시험 운행 채점용)."""
    lnmon, y, pred = np.asarray(lnmon), np.asarray(y), np.asarray(pred)
    months = np.sort(np.unique(lnmon))
    errs = [float((pred[np.isin(lnmon, months[2 * i:2 * i + 2])] != y[np.isin(lnmon, months[2 * i:2 * i + 2])]).mean())
            for i in range(3)]
    return float(np.dot([0.6, 0.3, 0.1], errs)), errs


def _arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--%s=" % name)), default)


if __name__ == "__main__":
    train_path = _arg("train", DEFAULT_TRAIN)
    if "--dry-run" in sys.argv:
        # 학습 데이터의 2023-06~2023-11로 학습하고 2023-12~2024-05를 맞혀 본다 (실제 평가와 같은 '학습 직후 6개월').
        # 프로젝트 백테스트 결과(가중 오분류율 0.1431)가 재현되면 이 파일이 올바르게 동작하는 것이다.
        full = load_train(train_path)
        tr_part = full[full.LNMON <= 202311].reset_index(drop=True)
        te_part = full[full.LNMON > 202311].reset_index(drop=True)
        truth = te_part.TARGET.values.copy()
        pred, prob = final_predict(te_part.drop(columns="TARGET"), train=tr_part, return_proba=True)
        err, per = weighted_error(te_part.LNMON.values, truth, pred)
        print("\n[시험 운행] 가중 오분류율 %.4f (기간별 %s), 비교 기준 0.1431" % (err, np.round(per, 4)))
        out = _arg("out")
        if out:
            pd.DataFrame(dict(row=np.arange(len(pred)), LNMON=te_part.LNMON.values, prob=prob, pred=pred)).to_csv(out, index=False)
    else:
        test_path = _arg("test")
        if not test_path:
            sys.exit("사용법: python final_predict.py --test=<테스트 CSV> [--train=<학습 CSV>] [--out=submission.csv]")
        pred, prob = final_predict(test_path, train=train_path, return_proba=True)
        test_ln = pd.read_csv(test_path, usecols=["LNMON"], encoding="utf-8-sig").LNMON.values
        out = _arg("out", os.path.join(HERE, "submission.csv"))
        pd.DataFrame(dict(row=np.arange(len(pred)), LNMON=test_ln, prob=prob, TARGET=pred)).to_csv(out, index=False)
        log("저장: %s (행 %d, 예측 양성 비율 %.3f)" % (out, len(pred), pred.mean()))

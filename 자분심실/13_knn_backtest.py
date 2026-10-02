"""kNN 타깃 인코딩 변수의 효과 백테스트.

거리 공간: 가장 이른 학습 구간(~2023-11)에서 학습한 LightGBM 중요도 상위 N_DIM개 원변수를
           전체 행 기준 순위(0~1)로 바꾼 것. 결측은 -0.25.
누수 방지:
- 학습 행: 동일인 그룹 단위 5조각 OOF (자기 자신과 같은 사람의 다른 신청 건을 이웃으로 쓰지 않음)
- 테스트 행: 학습 기간 행에서만 이웃을 찾고, 같은 동일인 그룹 이웃은 제외(라벨 전이가 따로 처리)
비교: dup+grp+weak vs dup+grp+weak+knn  (라벨 전이 + 그룹 평균 후처리 동일)

사용법: python 13_knn_backtest.py [--smoke]
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average
import features

warnings.filterwarnings("ignore")
SMOKE = "--smoke" in sys.argv
ONLY = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--folds=")), None)  # 예: --folds=C
N_DIM = 40
KS = [10, 50, 200]
K_MAX = max(KS)
EXTRA = 10  # 같은 그룹 이웃 제외용 여유분
HALF_LIFE = 3
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 150 if SMOKE else 1200
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
KNN_FEATS = ["KNN_MEAN_k%d" % k for k in KS] + ["KNN_WMEAN_k50", "KNN_DIST1", "KNN_DIST_k10",
                                                 "KNN_DIST_POS", "KNN_DIST_NEG"]


def knn_features(dist, idx, y_ref, valid):
    """dist/idx: (n, K_MAX+EXTRA) 이웃, valid: 사용할 이웃 마스크. 앞에서부터 유효한 K_MAX개만 쓴다."""
    n = dist.shape[0]
    order = np.argsort(~valid, axis=1, kind="stable")[:, :K_MAX]  # 유효 이웃을 앞으로
    d = np.take_along_axis(dist, order, 1)
    yy = y_ref[np.take_along_axis(idx, order, 1)].astype(np.float32)
    out = {}
    for k in KS:
        out["KNN_MEAN_k%d" % k] = yy[:, :k].mean(1)
    wts = 1.0 / (d[:, :50] + 1e-3)
    out["KNN_WMEAN_k50"] = (yy[:, :50] * wts).sum(1) / wts.sum(1)
    out["KNN_DIST1"] = d[:, 0]
    out["KNN_DIST_k10"] = d[:, :10].mean(1)
    big = d[:, -1:] * 1.5
    out["KNN_DIST_POS"] = np.where(yy == 1, d, big).min(1)
    out["KNN_DIST_NEG"] = np.where(yy == 0, d, big).min(1)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})


def query(Z_ref, Z_q):
    nn = NearestNeighbors(n_neighbors=K_MAX + EXTRA, algorithm="brute", n_jobs=8).fit(Z_ref)
    return nn.kneighbors(Z_q)


def fold_knn(Z, y, comp, tr_idx, te_idx):
    """학습 행 OOF + 테스트 행 kNN 변수."""
    tr_feat = pd.DataFrame(index=range(len(tr_idx)), columns=KNN_FEATS, dtype=np.float32)
    for fit_pos, q_pos in GroupKFold(5).split(tr_idx, groups=comp[tr_idx]):
        ref, q = tr_idx[fit_pos], tr_idx[q_pos]
        dist, nb = query(Z[ref], Z[q])
        tr_feat.iloc[q_pos] = knn_features(dist, nb, y[ref], np.ones_like(nb, bool)).values
    dist, nb = query(Z[tr_idx], Z[te_idx])
    same = comp[tr_idx][nb] == comp[te_idx][:, None]
    te_feat = knn_features(dist, nb, y[tr_idx], ~same)
    return tr_feat.astype(np.float32), te_feat


t0 = time.time()
df = load().reset_index(drop=True)
base_feats = feature_cols(df)
df, pairs, comp, groups = features.build_default(df)
print("구조 변수 생성 %ds" % (time.time() - t0), flush=True)

# 거리 공간 변수 선택: 가장 이른 학습 구간에서만 라벨 사용
early = df[df.LNMON <= FOLDS[0]["train_end"]]
sel_model = lgb.train(dict(PARAMS, learning_rate=0.05), lgb.Dataset(early[base_feats], early.TARGET.values), 300)
imp = pd.Series(sel_model.feature_importance("gain"), base_feats)
dims = [c for c in imp.sort_values(ascending=False).index if str(df[c].dtype) != "category"][:N_DIM]
Z = np.column_stack([df[c].rank(pct=True).fillna(-0.25).values for c in dims]).astype(np.float32)
print("거리 공간 %d차원:" % len(dims), dims[:10], "...", flush=True)

y, lnmon = df.TARGET.values, df.LNMON.values
struct = groups["dup"] + groups["grp"] + groups["weak"]
rows = []
run_folds = FOLDS[:1] if SMOKE else [f for f in FOLDS if ONLY is None or f["name"] in ONLY]
for f in run_folds:
    T = f["train_end"]
    tr_idx = np.where(lnmon <= T)[0]
    te_idx = np.where(lnmon > T)[0]
    if SMOKE:  # 행만 줄여 코드 경로 점검
        rng = np.random.default_rng(0)
        tr_idx = np.sort(rng.choice(tr_idx, 15000, replace=False))
        te_idx = np.sort(rng.choice(te_idx, 5000, replace=False))
    t1 = time.time()
    trk, tek = fold_knn(Z, y, comp, tr_idx, te_idx)
    print(f["name"], "kNN %ds" % (time.time() - t1), flush=True)
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    tr = pd.concat([tr, trk], axis=1)
    te = pd.concat([te, tek], axis=1)
    print("   학습 OOF 대 테스트 KNN_MEAN_k50 평균: %.3f / %.3f, 상관(타깃): %.3f / %.3f" % (
        tr.KNN_MEAN_k50.mean(), te.KNN_MEAN_k50.mean(),
        np.corrcoef(tr.KNN_MEAN_k50, tr.TARGET)[0, 1], np.corrcoef(te.KNN_MEAN_k50, te.TARGET)[0, 1]), flush=True)
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    for name, feats in [("dup+grp+weak", base_feats + struct), ("+knn", base_feats + struct + KNN_FEATS)]:
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        p, _ = label_transfer(p, te_idx, pairs, y, lnmon, T)
        p, _ = group_average(p, te_idx, pairs)
        r = dict(fold=f["name"], exp=name, auc=roc_auc_score(te.TARGET, p))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        rows.append(r)
        extra = ""
        if name == "+knn":
            rk = pd.Series(m.feature_importance("gain"), feats).rank(ascending=False)
            extra = rk[KNN_FEATS].astype(int).to_dict()
        print(f["name"], name, "AUC %.4f err@0.425 %.4f err@0.45 %.4f" % (r["auc"], r["err@0.425"], r["err@0.450"]),
              extra, flush=True)
    print("   ", f["name"], "총 %ds" % (time.time() - t1), flush=True)

res = pd.DataFrame(rows)
if not SMOKE:  # 폴드별로 나눠 실행해도 결과가 쌓이도록 덧붙여 저장
    out = os.path.join(ROOT, "results", "13_knn.csv")
    if os.path.exists(out):
        res = pd.concat([pd.read_csv(out), res]).drop_duplicates(["fold", "exp"], keep="last")
    res.to_csv(out, index=False)
print("\n=== 폴드 평균 ===")
print(res.groupby("exp", sort=False)[["auc"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

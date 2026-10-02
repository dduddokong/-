"""동일인 중복 정보의 효과 백테스트.

- 중복 쌍: 08에서 찾은 쌍 중 안정 변수 일치율 > SIM_THR (변수만 사용, 라벨 미사용).
- dup 변수(행 단위): 같은 달 / 이전 달 / 다음 달 / 2개월 차이 짝의 수, 최대 일치율.
- 라벨 전이: 테스트 행의 짝이 학습 기간(LNMON <= train_end)에 있으면 그 짝들의 타깃 평균으로 예측을 덮어쓴다.
- 그룹 일관성: 라벨 전이 후, 테스트 기간 안에서 연결된 행들의 예측을 평균한다.
"""
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT

warnings.filterwarnings("ignore")
SIM_THR = 0.9
HALF_LIFE = 3
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 1200
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
DUP_FEATS = ["DUP_SAME", "DUP_PREV", "DUP_NEXT", "DUP_GAP2", "DUP_MAXSIM"]


def add_dup_features(df, pairs):
    """행 단위 중복 변수. 쌍 (i, j)는 i < j 인덱스이며 월 차이 부호로 이전/다음을 구분한다."""
    n = len(df)
    m = df.LNMON.values
    mi = (m // 100) * 12 + m % 100
    out = {c: np.zeros(n, np.float32) for c in DUP_FEATS}
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:
        d = mi[b] - mi[a]  # 짝(b)이 a보다 몇 달 뒤인가
        np.add.at(out["DUP_SAME"], a[d == 0], 1)
        np.add.at(out["DUP_PREV"], a[d == -1], 1)
        np.add.at(out["DUP_NEXT"], a[d == 1], 1)
        np.add.at(out["DUP_GAP2"], a[np.abs(d) == 2], 1)
        np.maximum.at(out["DUP_MAXSIM"], a, pairs.sim_stable.values.astype(np.float32))
    for c, v in out.items():
        df[c] = v
    return df


def label_transfer(p, te_idx, pairs, y, lnmon, train_end):
    """테스트 행 중 학습 기간에 짝이 있는 행의 예측을 짝 타깃 평균으로 바꾼다."""
    p = p.copy()
    pos = {r: k for k, r in enumerate(te_idx)}
    s, c = np.zeros(len(te_idx)), np.zeros(len(te_idx))
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:
        for ra, rb in zip(a, b):
            if ra in pos and lnmon[rb] <= train_end:
                s[pos[ra]] += y[rb]
                c[pos[ra]] += 1
    k = c > 0
    p[k] = s[k] / c[k]
    return p, int(k.sum())


def group_average(p, te_idx, pairs):
    """테스트 기간 안에서 짝으로 연결된 행끼리 예측 평균."""
    pos = {r: k for k, r in enumerate(te_idx)}
    e = [(pos[a], pos[b]) for a, b in zip(pairs.i.values, pairs.j.values) if a in pos and b in pos]
    if not e:
        return p, 0
    e = np.array(e)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(p), len(p)))
    _, comp = connected_components(g, directed=False)
    avg = pd.Series(p).groupby(comp).transform("mean").values
    size = np.bincount(comp)[comp]
    return avg, int((size > 1).sum())


df = load().reset_index(drop=True)
pairs = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_all.pkl"))
pairs = pairs[pairs.sim_stable > SIM_THR].reset_index(drop=True)
df = add_dup_features(df, pairs)
print("사용 쌍 %d개 / 짝이 있는 행 %d개" % (len(pairs), (df[DUP_FEATS[:4]].sum(1) > 0).sum()))
print(df.groupby(df[DUP_FEATS[:4]].sum(1).clip(upper=3)).TARGET.agg(["size", "mean"]).round(3).to_string())

base_feats = feature_cols(df.drop(columns=DUP_FEATS))
y, lnmon = df.TARGET.values, df.LNMON.values
rows = []


def record(fold, name, te, p, **extra):
    r = dict(fold=fold, exp=name, auc=roc_auc_score(te.TARGET, p), **extra)
    for t in THRS:
        r["err@%.3f" % t], per = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    rows.append(r)
    print(fold, name, "AUC %.4f err@0.425 %.4f err@0.45 %.4f" % (r["auc"], r["err@0.425"], r["err@0.450"]), extra, flush=True)


for f in FOLDS:
    T = f["train_end"]
    tr, te = df[df.LNMON <= T], df[df.LNMON > T]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    for name, feats in [("base", base_feats), ("+dup", base_feats + DUP_FEATS)]:
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        record(f["name"], name, te, p)
        if name == "+dup":
            imp = pd.Series(m.feature_importance("gain"), feats)
            print("   dup 변수 gain 순위:", {c: int((imp > imp[c]).sum()) + 1 for c in DUP_FEATS})
            p2, n_lt = label_transfer(p, te.index.values, pairs, y, lnmon, T)
            record(f["name"], "+dup+transfer", te, p2, n_transfer=n_lt)
            p3, n_grp = group_average(p2, te.index.values, pairs)
            record(f["name"], "+dup+transfer+group", te, p3, n_group=n_grp)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "09_dup_backtest.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== 폴드 평균 ===")
print(res.groupby("exp", sort=False)[["auc"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

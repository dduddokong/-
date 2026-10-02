"""약한 유사 쌍(안정 변수 일치율 0.8~0.9) 개수 변수의 효과 백테스트.

동일인 쌍(FS>300)과 별개로, 1개월에도 잘 변하지 않는 변수로 LSH 후보를 넓게 모아
같은 달/인접 달/2개월 이상 차이별 '약한 유사 이웃' 수를 센다 (라벨 미사용).
비교: dup+grp (11의 최선) vs dup+grp+weak
"""
import os
import importlib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, CAT_COLS, weighted_error, ROOT
from dup_utils import DUP_FEATS, add_dup_features, components, label_transfer, group_average, month_index

warnings.filterwarnings("ignore")
m08 = importlib.import_module("08_dup_matching")
FS_THR, HALF_LIFE, N_CHANGING = 300, 3, 12
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 1200
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]

df = load().reset_index(drop=True)
base_feats = feature_cols(df)
num_feats = [c for c in base_feats if c not in CAT_COLS]
mi = month_index(df.LNMON.values)

# --- 동일인 쌍 + dup/grp 변수 (11과 동일)
pairs = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))
pairs = pairs[pairs.fs > FS_THR][["i", "j", "gap", "fs"]].reset_index(drop=True)
df = add_dup_features(df, pairs, sim_col="fs")
comp = components(len(df), pairs)
g = pd.DataFrame(dict(comp=comp, mi=mi))
df["GRP_SIZE"] = g.groupby("comp").comp.transform("size").astype(np.float32)
df["GRP_NMONTHS"] = g.groupby("comp").mi.transform("nunique").astype(np.float32)
df["GRP_ORDER"] = (mi - g.groupby("comp").mi.transform("min")).astype(np.float32)
X = df[num_feats].values
a, b = X[pairs.i.values], X[pairs.j.values]
differ = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), num_feats)
informative = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.5, num_feats)
changing = differ[informative].sort_values(ascending=False).index[:N_CHANGING].tolist()
GRP_FEATS = ["GRP_SIZE", "GRP_NMONTHS", "GRP_ORDER"]
for c in changing:
    s = df[c].groupby(comp)
    df["GMAX_" + c] = s.transform("max").astype(np.float32)
    df["GMIN_" + c] = s.transform("min").astype(np.float32)
    df["GDIFF_" + c] = (df[c] - s.transform("mean")).astype(np.float32)
    GRP_FEATS += ["GMAX_" + c, "GMIN_" + c, "GDIFF_" + c]

# --- 약한 유사 이웃
dyn = pd.read_csv(os.path.join(ROOT, "results", "12_column_dynamics.csv"), index_col=0)
inf30 = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.3, num_feats)
keys = [c for c in dyn[(dyn.eq0_g1 > 0.97) & (dyn.nun > 200)].index if inf30.get(c, False)]
stable = list(dyn[dyn.eq0_g1 > 0.97].index)
m08.MAX_BUCKET = 6
cand = m08.lsh_pairs(df, keys, 150, 2, 5)
gap = np.abs(mi[cand[:, 0]] - mi[cand[:, 1]])
sim = m08.match_rate(df[stable].values.astype(np.float64), cand)
weak = (sim > 0.8) & (sim <= 0.9)
print("후보 %d, 약한 유사 쌍 %d (gap0 %d, gap1 %d, gap2+ %d)" %
      (len(cand), weak.sum(), (weak & (gap == 0)).sum(), (weak & (gap == 1)).sum(), (weak & (gap >= 2)).sum()))
WEAK_FEATS = ["WEAK_SAME", "WEAK_ADJ", "WEAK_FAR", "MAXSIM_STABLE"]
for c in WEAK_FEATS:
    df[c] = np.zeros(len(df), np.float32)
for sel, col in [(weak & (gap == 0), "WEAK_SAME"), (weak & (gap == 1), "WEAK_ADJ"), (weak & (gap >= 2), "WEAK_FAR")]:
    for side in (0, 1):
        np.add.at(df[col].values, cand[sel, side], 1)
for side in (0, 1):  # 동일인 쌍을 제외한 이웃 중 최대 유사도
    s = np.where(sim > 0.9, 0, sim).astype(np.float32)
    np.maximum.at(df["MAXSIM_STABLE"].values, cand[:, side], s)
print(df.groupby(df.WEAK_SAME.clip(upper=3) + df.WEAK_ADJ.clip(upper=3)).TARGET.agg(["size", "mean"]).round(3).to_string())

EXPS = {"dup+grp": base_feats + DUP_FEATS + GRP_FEATS,
        "dup+grp+weak": base_feats + DUP_FEATS + GRP_FEATS + WEAK_FEATS}
y, lnmon = df.TARGET.values, df.LNMON.values
rows = []
for f in FOLDS:
    T = f["train_end"]
    tr, te = df[df.LNMON <= T], df[df.LNMON > T]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    for name, feats in EXPS.items():
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        p, _ = label_transfer(p, te.index.values, pairs, y, lnmon, T)
        p, _ = group_average(p, te.index.values, pairs)
        r = dict(fold=f["name"], exp=name, auc=roc_auc_score(te.TARGET, p))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        rows.append(r)
        extra = ""
        if name.endswith("weak"):
            imp = pd.Series(m.feature_importance("gain"), feats).rank(ascending=False)
            extra = imp[WEAK_FEATS].astype(int).to_dict()
        print(f["name"], name, "AUC %.4f err@0.425 %.4f err@0.45 %.4f" % (r["auc"], r["err@0.425"], r["err@0.450"]), extra, flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "12_weak_pairs.csv"), index=False)
print("\n=== 폴드 평균 ===")
print(res.groupby("exp", sort=False)[["auc"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

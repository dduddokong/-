"""② 그룹 집계·짝 차이 변수, 빈도 인코딩  ③ EM 라벨 분포 보정 — 백테스트.

중복 쌍: 10의 Fellegi-Sunter 점수 > FS_THR (점수 분포의 골짜기, 라벨 미사용).
변수(모두 라벨 미사용, 학습+테스트 전체 행으로 계산 → 실제 테스트셋에서도 동일하게 계산 가능):
- dup  : 짝 개수(같은 달/이전/다음/2개월), 최대 FS 점수
- grp  : 그룹 크기·걸친 달 수·그룹 안 시간 순서, 신청 건마다 바뀌는 변수의 그룹 max/min/(값-그룹평균)
- freq : 값 종류가 많은 상위 변수의 동일 값 빈도(전체, 같은 달)
후처리: 라벨 전이 → 그룹 평균 → (고정 임계값 | EM 보정 후 0.5)
"""
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, CAT_COLS, weighted_error, fold_periods, ROOT
from dup_utils import (DUP_FEATS, add_dup_features, components, label_transfer, group_average,
                       em_prior, month_index)

warnings.filterwarnings("ignore")
FS_THR = 300
N_CHANGING, N_FREQ = 12, 20
HALF_LIFE = 3
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 1200
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]

df = load().reset_index(drop=True)
base_feats = feature_cols(df)
num_feats = [c for c in base_feats if c not in CAT_COLS]
pairs = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))
pairs = pairs[pairs.fs > FS_THR][["i", "j", "gap", "fs"]].reset_index(drop=True)
print("FS>%d 쌍 %d개" % (FS_THR, len(pairs)))

# --- dup
df = add_dup_features(df, pairs, sim_col="fs")

# --- grp
comp = components(len(df), pairs)
mi = month_index(df.LNMON.values)
g = pd.DataFrame(dict(comp=comp, mi=mi))
df["GRP_SIZE"] = g.groupby("comp").comp.transform("size").astype(np.float32)
df["GRP_NMONTHS"] = g.groupby("comp").mi.transform("nunique").astype(np.float32)
df["GRP_ORDER"] = (mi - g.groupby("comp").mi.transform("min")).astype(np.float32)
X = df[num_feats].values
a, b = X[pairs.i.values], X[pairs.j.values]
differ = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), num_feats)
informative = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.5, num_feats)
changing = differ[informative].sort_values(ascending=False).index[:N_CHANGING].tolist()
print("신청 건마다 바뀌는 변수:", changing)
GRP_FEATS = ["GRP_SIZE", "GRP_NMONTHS", "GRP_ORDER"]
for c in changing:
    s = df[c].groupby(comp)
    df["GMAX_" + c] = s.transform("max").astype(np.float32)
    df["GMIN_" + c] = s.transform("min").astype(np.float32)
    df["GDIFF_" + c] = (df[c] - s.transform("mean")).astype(np.float32)
    GRP_FEATS += ["GMAX_" + c, "GMIN_" + c, "GDIFF_" + c]

# --- freq: 값 종류가 많은 변수 중 단변량 AUC 상위 (변수 '선택'만 라벨을 쓰고, 값은 라벨 미사용)
nun = df[num_feats].nunique()
hc = [c for c in num_feats if nun[c] > 500 and informative[c]]
sel = df[df.LNMON <= FOLDS[0]["train_end"]]  # 선택은 가장 이른 학습 구간에서만 (누수 방지)
auc = {c: abs(roc_auc_score(sel.TARGET, sel[c].fillna(-1e9)) - 0.5) for c in hc}
freq_cols = sorted(auc, key=auc.get, reverse=True)[:N_FREQ]
FREQ_FEATS = []
for c in freq_cols:
    v = df[c].fillna(-1e18)
    df["CNT_" + c] = v.map(v.value_counts()).astype(np.float32)
    df["CNTM_" + c] = df.groupby([df.LNMON, v])[c].transform("size").astype(np.float32)
    FREQ_FEATS += ["CNT_" + c, "CNTM_" + c]

EXPS = {
    "dup":          base_feats + DUP_FEATS,
    "dup+grp":      base_feats + DUP_FEATS + GRP_FEATS,
    "dup+freq":     base_feats + DUP_FEATS + FREQ_FEATS,
    "dup+grp+freq": base_feats + DUP_FEATS + GRP_FEATS + FREQ_FEATS,
}
y, lnmon = df.TARGET.values, df.LNMON.values
rows = []


def evaluate(fold, name, te, p, em_p=None, **extra):
    r = dict(fold=fold, exp=name, auc=roc_auc_score(te.TARGET, p), **extra)
    for t in THRS:
        r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    if em_p is not None:
        r["err_em@0.5"], _ = weighted_error(te.LNMON, te.TARGET, (em_p > 0.5).astype(int))
    rows.append(r)
    print(fold, name, "AUC %.4f err@0.45 %.4f" % (r["auc"], r["err@0.450"]),
          "em %.4f" % r["err_em@0.5"] if em_p is not None else "", extra, flush=True)


for f in FOLDS:
    T = f["train_end"]
    tr, te = df[df.LNMON <= T], df[df.LNMON > T]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    prior_train = float(np.average(tr.TARGET, weights=w))
    for name, feats in EXPS.items():
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        p, _ = label_transfer(p, te.index.values, pairs, y, lnmon, T)
        p, _ = group_average(p, te.index.values, pairs)
        # EM: 테스트 2개월 기간마다 타깃 비율 추정 후 보정
        em_p, est = p.copy(), {}
        for ms in fold_periods(T):
            k = np.isin(te.LNMON.values, ms)
            pi, em_p[k] = em_prior(p[k], prior_train)
            est["%d-%d" % (ms[0] % 100, ms[1] % 100)] = "%.3f/%.3f" % (pi, te.TARGET.values[k].mean())
        evaluate(f["name"], name, te, p, em_p, prior_train=round(prior_train, 3), em_est_vs_true=est)
        if name == "dup+grp+freq":
            imp = pd.Series(m.feature_importance("gain"), feats).rank(ascending=False)
            print("   중요도 순위 상위 신규 변수:", imp[DUP_FEATS + GRP_FEATS + FREQ_FEATS].sort_values().head(10).astype(int).to_dict())

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "11_group_freq_em.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== 폴드 평균 (라벨 전이 + 그룹 평균 적용 후) ===")
print(res.groupby("exp", sort=False)[["auc"] + ["err@%.3f" % t for t in THRS] + ["err_em@0.5"]].mean().round(4).to_string())

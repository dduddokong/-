"""행 단위 결측/특수값 요약 변수를 LightGBM에 추가했을 때의 효과.

기준: 02에서 채택한 최근 월 가중(반감기 3개월) LightGBM.
- NA_CNT      : 행의 결측 변수 개수
- SV_CNT      : 행의 특수값 플래그(_SV_FLAG_) 합
- RARE_PRESENT: 결측률 50% 초과 변수 중 값이 '있는' 개수 (희귀 정보 보유량)
시드만 바꾼 기준 모델(base_seed1)로 실험 간 차이가 잡음 수준인지 함께 본다.
"""
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT

warnings.filterwarnings("ignore")
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS, HALF_LIFE = 1200, 3
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]

df = load()
base_feats = feature_cols(df)
na_cols = [c for c in base_feats if df[c].isna().any()]
rare_cols = [c for c in na_cols if df[c].isna().mean() > 0.5]
sv_cols = [c for c in base_feats if "_SV_FLAG_" in c]
# 행 단위 집계라 다른 행/타깃 정보를 쓰지 않으므로 전체에서 한 번 계산해도 누수가 없다
df["NA_CNT"] = df[na_cols].isna().sum(1).astype(np.float32)
df["SV_CNT"] = df[sv_cols].sum(1).astype(np.float32)
df["RARE_PRESENT"] = df[rare_cols].notna().sum(1).astype(np.float32)
print("na_cols %d, rare_cols %d, sv_cols %d" % (len(na_cols), len(rare_cols), len(sv_cols)))
print(df.groupby(pd.cut(df.NA_CNT, [0, 15, 20, 25, 30, 40, 80])).TARGET.agg(["size", "mean"]).round(3).to_string())
print(df.groupby("RARE_PRESENT").TARGET.agg(["size", "mean"]).head(8).round(3).to_string())

EXPS = {
    "base":        dict(extra=[]),
    "base_seed1":  dict(extra=[], seed=1),
    "+na":         dict(extra=["NA_CNT"]),
    "+na+sv+rare": dict(extra=["NA_CNT", "SV_CNT", "RARE_PRESENT"]),
}

rows = []
for f in FOLDS:
    tr, te = df[df.LNMON <= f["train_end"]], df[df.LNMON > f["train_end"]]
    age = (f["train_end"] // 100 - tr.LNMON // 100) * 12 + f["train_end"] % 100 - tr.LNMON % 100
    w = 0.5 ** (age.values / HALF_LIFE)
    for name, e in EXPS.items():
        feats = base_feats + e["extra"]
        m = lgb.train(dict(PARAMS, seed=e.get("seed", 0)), lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        r = dict(fold=f["name"], exp=name, auc=roc_auc_score(te.TARGET, p))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        if e["extra"]:
            imp = pd.Series(m.feature_importance("gain"), feats)
            r["gain_rank"] = {c: int((imp > imp[c]).sum()) + 1 for c in e["extra"]}
        rows.append(r)
        print(f["name"], name, "AUC %.4f err@0.45 %.4f err@0.5 %.4f" % (r["auc"], r["err@0.450"], r["err@0.500"]),
              r.get("gain_rank", ""), flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "05_missing_count.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== 폴드별 AUC ===")
print(res.pivot(index="exp", columns="fold", values="auc").round(4).to_string())
print("\n=== 폴드 평균 ===")
print(res.groupby("exp")[["auc"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

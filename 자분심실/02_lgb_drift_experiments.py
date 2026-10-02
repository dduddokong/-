"""LightGBM 실험: 학습 기간 길이, 최근 월 가중, 드리프트 변수 제거가 시간 외 오분류율에 주는 영향."""
import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, add_months, ROOT

warnings.filterwarnings("ignore")

# EDA(PSI, 적대적 검증)에서 앞뒤 기간 분포 차이가 컸던 변수
DRIFT = ["TS_RECENT_APS003", "TS_MOMENTUM_APS003", "TS_VOLATILITY_APS003", "TS_MAX_IN_3M_FLAG_APS003",
         "TS_TOTAL_SLOPE_APS003", "UA1040000", "UA1040003", "D10187D00", "DUN000028", "DA1000104"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 1200

EXPERIMENTS = {
    "base":           dict(),
    "last6m":         dict(window=6),
    "decay_hl6":      dict(half_life=6),
    "decay_hl3":      dict(half_life=3),
    "drop_drift":     dict(drop=DRIFT),
    "drop_drift_hl6": dict(drop=DRIFT, half_life=6),
}
THRS = np.round(np.arange(0.40, 0.601, 0.025), 3)

df = load()
rows, preds = [], []
for f in FOLDS:
    te = df[df.LNMON > f["train_end"]]
    for name, e in EXPERIMENTS.items():
        start = add_months(f["train_end"], -e["window"] + 1) if "window" in e else 0
        tr = df[(df.LNMON <= f["train_end"]) & (df.LNMON >= start)]
        if name == "last6m" and tr.LNMON.nunique() == df[df.LNMON <= f["train_end"]].LNMON.nunique():
            continue  # 폴드 A는 원래 6개월이라 base와 동일
        feats = feature_cols(df, drop=e.get("drop", ()))
        w = None
        if "half_life" in e:
            age = np.array([(f["train_end"] // 100 - m // 100) * 12 + f["train_end"] % 100 - m % 100
                            for m in tr.LNMON.values])
            w = 0.5 ** (age / e["half_life"])
        t0 = time.time()
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
        p = m.predict(te[feats])
        r = dict(fold=f["name"], exp=name, auc=roc_auc_score(te.TARGET, p), mean_p=p.mean(),
                 base_rate=te.TARGET.mean(), sec=round(time.time() - t0))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        rows.append(r)
        preds.append(pd.DataFrame(dict(fold=f["name"], exp=name, LNMON=te.LNMON.values, y=te.TARGET.values, p=p)))
        print(f["name"], name, "AUC %.4f err@0.5 %.4f (%ds)" % (r["auc"], r["err@0.500"], r["sec"]), flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "02_lgb_experiments.csv"), index=False)
pd.concat(preds).to_pickle(os.path.join(ROOT, "cache", "02_lgb_preds.pkl"))
pd.set_option("display.width", 220)
cols = ["auc", "mean_p"] + ["err@%.3f" % t for t in THRS]
print("\n=== 폴드별 ===")
print(res[["fold", "exp", "auc", "mean_p", "base_rate", "err@0.500"]].round(4).to_string(index=False))
print("\n=== 실험별 폴드 평균 (B, C만: last6m 비교용) ===")
print(res[res.fold != "A"].groupby("exp")[cols].mean().round(4).to_string())
print("\n=== 실험별 폴드 평균 (A, B, C) ===")
print(res.groupby("exp")[cols].mean().round(4).to_string())

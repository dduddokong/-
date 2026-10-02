"""상호작용 기여 확인: LightGBM 나무 크기를 바꿔 가법(stump) 모형과 상호작용 모형을 비교.

num_leaves=2 → 각 나무가 변수 하나만 사용(비선형 O, 상호작용 X)
num_leaves=4 → 최대 2~3개 변수 상호작용
num_leaves=31 → 기준선 설정
"""
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT

warnings.filterwarnings("ignore")
BASE = dict(objective="binary", min_child_samples=200, feature_fraction=0.5, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10, verbose=-1, seed=0, num_threads=8)
CONFIGS = {  # 작은 나무는 보정량이 작으므로 반복수를 늘린다
    "stump_L2": dict(num_leaves=2, learning_rate=0.1, rounds=3000),
    "L4": dict(num_leaves=4, learning_rate=0.05, rounds=2000),
    "L31": dict(num_leaves=31, learning_rate=0.03, rounds=1200),
}
THRS = [0.425, 0.45, 0.475, 0.5]

df = load()
feats = feature_cols(df)
rows = []
for f in FOLDS:
    tr, te = df[df.LNMON <= f["train_end"]], df[df.LNMON > f["train_end"]]
    dtr = lgb.Dataset(tr[feats], tr.TARGET.values, free_raw_data=False)
    for name, c in CONFIGS.items():
        params = dict(BASE, num_leaves=c["num_leaves"], learning_rate=c["learning_rate"])
        p = lgb.train(params, dtr, c["rounds"]).predict(te[feats])
        r = dict(fold=f["name"], model=name, auc=roc_auc_score(te.TARGET, p))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        rows.append(r)
        print(f["name"], name, "AUC %.4f err@0.5 %.4f" % (r["auc"], r["err@0.500"]), flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT, "results", "04_interaction_check.csv"), index=False)
print("\n=== 폴드 평균 ===")
print(res.groupby("model")[["auc"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

"""최종 변수(원변수 + dup/grp/weak + kNN past99)로 부스팅 계열 비교와 앙상블 (cache/fold{A,B,C}.pkl 사용).

모델
- lgb_base  : 지금까지의 LightGBM 설정 (캐시에 저장된 시드 3개 예측)
- lgb_tuned : Optuna #36 설정 (잎 35, 잎 최소 50, 변수 0.52, 행 0.957, λ2 11.3, 최소 이득 0.17, 학습률 0.05 × 1100회), 시드 3개
- xgb       : 깊이 6, 학습률 0.05 × 1200회, 시드 2개
- cat       : 깊이 6, 학습률 0.08 × 2000회, 시드 1개
모두 최근 월 가중(반감기 3개월), 후처리(라벨 전이 → 그룹 평균) 동일.

사용법: python 18_boost_ensemble.py [--folds=A,B,C]
결과: results/18_boost_ensemble.csv (폴드별로 덧붙임), 예측: cache/18_preds_fold{X}.pkl
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average

warnings.filterwarnings("ignore")
RUN = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--folds=")), ["A", "B", "C"])
HALF_LIFE = 3
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
OUT = os.path.join(ROOT, "results", "18_boost_ensemble.csv")
LGB_TUNED = dict(objective="binary", learning_rate=0.05, num_leaves=35, min_child_samples=50,
                 feature_fraction=0.52, bagging_fraction=0.957, bagging_freq=1, lambda_l1=0.001,
                 lambda_l2=11.31, min_gain_to_split=0.17, verbose=-1, num_threads=8)
XGB_P = dict(objective="binary:logistic", tree_method="hist", eta=0.05, max_depth=6, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.5, reg_lambda=10, max_bin=256, nthread=8)
CAT_P = dict(loss_function="Logloss", learning_rate=0.08, depth=6, l2_leaf_reg=10, rsm=0.5,
             bootstrap_type="Bernoulli", subsample=0.8, border_count=254, nan_mode="Min",
             thread_count=8, verbose=0)

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["fold"])
for fold in RUN:
    if (done.fold == fold).any():
        print(fold, "이미 완료", flush=True)
        continue
    t0 = time.time()
    T = next(f for f in FOLDS if f["name"] == fold)["train_end"]
    C = pd.read_pickle(os.path.join(ROOT, "cache", "fold%s.pkl" % fold))
    df, feats, pairs, tr_idx, te_idx, cats = C["df"], C["feats"], C["pairs"], C["tr_idx"], C["te_idx"], C["cat_cols"]
    y, ln = df.TARGET.values, df.LNMON.values
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    raw = {"lgb_base": np.mean([C["lgb_raw"][s] for s in sorted(C["lgb_raw"])], 0)}

    ps = []
    for s in range(3):
        m = lgb.train(dict(LGB_TUNED, seed=s), lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), 1100)
        ps.append(m.predict(te[feats]))
    raw["lgb_tuned"] = np.mean(ps, 0)
    print(fold, "lgb_tuned %ds" % (time.time() - t0), flush=True)

    dtr = xgb.DMatrix(tr[feats], tr.TARGET.values, weight=w, enable_categorical=True)
    dte = xgb.DMatrix(te[feats], enable_categorical=True)
    raw["xgb"] = np.mean([xgb.train(dict(XGB_P, seed=s), dtr, 1200).predict(dte) for s in range(2)], 0)
    print(fold, "xgb %ds" % (time.time() - t0), flush=True)

    def pool(d, lab=None, wt=None):
        X = d[feats].copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return Pool(X, lab, weight=wt, cat_features=cats)
    cm = CatBoostClassifier(iterations=2000, random_seed=0, **CAT_P).fit(pool(tr, tr.TARGET.values, w))
    raw["cat"] = cm.predict_proba(pool(te))[:, 1]
    print(fold, "cat %ds" % (time.time() - t0), flush=True)

    combos = {
        "lgb_base": ["lgb_base"], "lgb_tuned": ["lgb_tuned"], "xgb": ["xgb"], "cat": ["cat"],
        "lgb_base+tuned": ["lgb_base", "lgb_tuned"],
        "lgb_base+xgb": ["lgb_base", "xgb"], "lgb_base+cat": ["lgb_base", "cat"],
        "lgb_base+xgb+cat": ["lgb_base", "xgb", "cat"], "lgb_tuned+xgb+cat": ["lgb_tuned", "xgb", "cat"],
        "all4": ["lgb_base", "lgb_tuned", "xgb", "cat"],
    }
    rows = []
    for name, members in combos.items():
        p = np.mean([raw[m] for m in members], 0)
        p, _ = label_transfer(p, te_idx, pairs, y, ln, T)
        p, _ = group_average(p, te_idx, pairs)
        r = dict(fold=fold, model=name, auc=roc_auc_score(te.TARGET, p),
                 logloss=log_loss(te.TARGET, np.clip(p, 1e-6, 1 - 1e-6)))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        rows.append(r)
    corr = pd.DataFrame(raw).rank().corr().round(4)
    print(fold, "예측 순위 상관\n", corr, flush=True)
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False), flush=True)
    pd.to_pickle(raw, os.path.join(ROOT, "cache", "18_preds_fold%s.pkl" % fold))
    done = pd.concat([done, res], ignore_index=True)
    done.to_csv(OUT, index=False)
    print(fold, "완료 %ds" % (time.time() - t0), flush=True)

pd.set_option("display.width", 220)
cols = ["auc", "logloss"] + ["err@%.3f" % t for t in THRS]
print("\n=== 세 폴드 평균 ===")
print(done.groupby("model", sort=False)[cols].mean().round(4).to_string())
print("\n=== 폴드별 err@0.450 ===")
print(done.pivot(index="model", columns="fold", values="err@0.450").round(4).to_string())

"""kNN 변형 비교: past40(과거 이웃만), oof20, oof80.  비교 기준(oof40)은 13의 결과를 쓴다.

사용법: python 14_knn_variants.py [--variants=past40,oof20,oof80] [--smoke]
결과는 results/14_knn_variants.csv에 변형·폴드별로 덧붙여 저장 (중단돼도 이어서 실행 가능).
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average
import features
import knn_utils as K

warnings.filterwarnings("ignore")
SMOKE = "--smoke" in sys.argv
VARIANTS = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--variants=")),
                ["past40", "oof20", "oof80"])
HALF_LIFE = 3
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
ROUNDS = 150 if SMOKE else 1200
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
OUT = os.path.join(ROOT, "results", "14_knn_variants.csv")

t0 = time.time()
df = load().reset_index(drop=True)
base_feats = feature_cols(df)
df, pairs, comp, groups = features.build_default(df)
struct = groups["dup"] + groups["grp"] + groups["weak"]
y, lnmon = df.TARGET.values, df.LNMON.values
early = (lnmon <= FOLDS[0]["train_end"])
print("구조 변수 %ds" % (time.time() - t0), flush=True)

done = pd.read_csv(OUT) if os.path.exists(OUT) and not SMOKE else pd.DataFrame(columns=["variant", "fold"])
spaces = {}
for v in VARIANTS:
    mode, n_dim = v[:-2], int(v[-2:])
    if n_dim not in spaces:
        spaces[n_dim] = K.distance_space(df, base_feats, early, n_dim, PARAMS)[0]
    Z = spaces[n_dim]
    for f in (FOLDS[:1] if SMOKE else FOLDS):
        if ((done.variant == v) & (done.fold == f["name"])).any():
            print(v, f["name"], "이미 완료, 건너뜀", flush=True)
            continue
        T = f["train_end"]
        tr_idx, te_idx = np.where(lnmon <= T)[0], np.where(lnmon > T)[0]
        if SMOKE:
            rng = np.random.default_rng(0)
            tr_idx = np.sort(rng.choice(tr_idx, 15000, replace=False))
            te_idx = np.sort(rng.choice(te_idx, 5000, replace=False))
        t1 = time.time()
        trk, tek = K.fold_features(Z, y, comp, lnmon, tr_idx, te_idx, mode=mode)
        tk = time.time() - t1
        tr = pd.concat([df.iloc[tr_idx].reset_index(drop=True), trk], axis=1)
        te = pd.concat([df.iloc[te_idx].reset_index(drop=True), tek], axis=1)
        c_tr = pd.Series(tr.KNN_MEAN_k50).corr(tr.TARGET)
        c_te = pd.Series(te.KNN_MEAN_k50).corr(te.TARGET)
        age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
        feats = base_feats + struct + K.KNN_FEATS
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.TARGET.values, weight=0.5 ** (age / HALF_LIFE)), ROUNDS)
        p = m.predict(te[feats])
        p, _ = label_transfer(p, te_idx, pairs, y, lnmon, T)
        p, _ = group_average(p, te_idx, pairs)
        r = dict(variant=v, fold=f["name"], auc=roc_auc_score(te.TARGET, p), corr_tr=c_tr, corr_te=c_te,
                 sec_knn=round(tk), sec=round(time.time() - t1))
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        print(v, f["name"], "AUC %.4f err@0.425 %.4f err@0.45 %.4f | 상관 학습 %.3f 테스트 %.3f | %ds" % (
            r["auc"], r["err@0.425"], r["err@0.450"], c_tr, c_te, r["sec"]), flush=True)
        if not SMOKE:
            done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
            done.to_csv(OUT, index=False)

if not SMOKE:
    print("\n=== 변형별 폴드 평균 ===")
    print(done.groupby("variant")[["auc", "corr_tr", "corr_te"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

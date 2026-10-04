"""kNN(past) 거리 공간 차원 수 탐색.

cache/fold{X}.pkl의 원변수·구조 변수를 그대로 쓰고, kNN 변수만 차원 수를 바꿔 다시 계산한다.
차원은 LightGBM 중요도 순서의 앞에서부터 n개 (중요도 계산은 한 번만, 학습 기간 라벨만 사용).
평가: LightGBM 기존 설정 시드 2개 평균 → 라벨 전이 → 그룹 평균.

사용법: python 19_knn_dim_scan.py --fold=A --dims=10,20,...,200
결과: results/19_knn_dim_scan.csv (폴드·차원별로 덧붙임, 중단 후 재실행하면 이어서 진행)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average, components
import knn_utils as K

warnings.filterwarnings("ignore")
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--%s=" % k)), d)
FOLD = arg("fold", "A")
DIMS = [int(x) for x in arg("dims", ",".join(str(d) for d in range(10, 201, 10))).split(",")]
SEEDS = [0, 1]
HALF_LIFE, ROUNDS = 3, 1200
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, num_threads=8)
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
OUT = os.path.join(ROOT, "results", "19_knn_dim_scan.csv")

t0 = time.time()
T = next(f for f in FOLDS if f["name"] == FOLD)["train_end"]
C = pd.read_pickle(os.path.join(ROOT, "cache", "fold%s.pkl" % FOLD))
df, pairs, tr_idx, te_idx = C["df"], C["pairs"], C["tr_idx"], C["te_idx"]
feats_no_knn = [c for c in C["feats"] if c not in K.KNN_FEATS]
base_feats = [c for c in feats_no_knn if not c.startswith(("DUP_", "GRP_", "GMAX_", "GMIN_", "GDIFF_", "WEAK_", "MAXSIM_"))]
y, ln = df.TARGET.values, df.LNMON.values
comp = components(len(df), pairs)

# 중요도 순서는 한 번만 계산하고, 차원 수만큼 앞에서 자른다
Z_all, dims_all = K.distance_space(df, base_feats, ln <= T, max(DIMS), PARAMS)
print("폴드 %s, 원변수 %d개, 중요도 상위 %d개 준비 %ds" % (FOLD, len(base_feats), len(dims_all), time.time() - t0), flush=True)

tr = df.iloc[tr_idx].reset_index(drop=True)
te = df.iloc[te_idx].reset_index(drop=True)
age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
w = 0.5 ** (age / HALF_LIFE)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["fold", "dims"])

for d in DIMS:
    if ((done.fold == FOLD) & (done.dims == d)).any():
        print(d, "이미 완료", flush=True)
        continue
    t1 = time.time()
    Z = np.ascontiguousarray(Z_all[:, :d])
    trk, tek = K.fold_features(Z, y, comp, ln, tr_idx, te_idx, mode="past")
    tk = time.time() - t1
    trd = pd.concat([tr[feats_no_knn], trk], axis=1)
    ted = pd.concat([te[feats_no_knn], tek], axis=1)
    feats = feats_no_knn + K.KNN_FEATS
    p = np.mean([lgb.train(dict(PARAMS, seed=s), lgb.Dataset(trd[feats], tr.TARGET.values, weight=w), ROUNDS)
                 .predict(ted[feats]) for s in SEEDS], 0)
    p, _ = label_transfer(p, te_idx, pairs, y, ln, T)
    p, _ = group_average(p, te_idx, pairs)
    r = dict(fold=FOLD, dims=d, auc=roc_auc_score(te.TARGET, p), logloss=log_loss(te.TARGET, np.clip(p, 1e-6, 1 - 1e-6)),
             corr_tr=pd.Series(trk.KNN_MEAN_k50.values).corr(tr.TARGET), corr_te=pd.Series(tek.KNN_MEAN_k50.values).corr(te.TARGET),
             sec_knn=round(tk), sec=round(time.time() - t1))
    for t in THRS:
        r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    r["err_best"] = min(r["err@%.3f" % t] for t in THRS)
    done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
    done.to_csv(OUT, index=False)
    print("dims %3d | AUC %.4f logloss %.4f err@0.45 %.4f best %.4f | 상관 학습 %.3f 테스트 %.3f | %ds" % (
        d, r["auc"], r["logloss"], r["err@0.450"], r["err_best"], r["corr_tr"], r["corr_te"], r["sec"]), flush=True)

print("총 %ds" % (time.time() - t0))

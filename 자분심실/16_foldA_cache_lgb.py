"""폴드별 최종 변수 캐시 + LightGBM(최종 구성) 시드 3개 예측 저장.  사용법: python 16_foldA_cache_lgb.py [--fold=A|B|C]

저장물 (cache/fold{A|B|C}.pkl):
- df      : 원변수 + dup/grp/weak + kNN(past99), LNMON, TARGET (0..n-1 인덱스)
- feats   : 최종 모델 입력 변수 목록
- pairs   : 동일인 쌍 (라벨 전이·그룹 평균용)
- tr_idx, te_idx
- lgb_raw : 시드별 LightGBM 예측 (후처리 전), {seed: array}
신경망 실험(17)은 이 캐시만 읽어 kNN을 다시 계산하지 않는다.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from kcb_common import load, feature_cols, FOLDS, ROOT
import features
import knn_utils as K

warnings.filterwarnings("ignore")
FOLD = next((a.split("=")[1] for a in sys.argv if a.startswith("--fold=")), "A")
T = next(f for f in FOLDS if f["name"] == FOLD)["train_end"]
HALF_LIFE, ROUNDS, N_DIM, SEEDS = 3, 1200, 99, [0, 1, 2]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)

t0 = time.time()
df = load().reset_index(drop=True)
base_feats = feature_cols(df)
df, pairs, comp, groups = features.build_default(df)
y, ln = df.TARGET.values, df.LNMON.values
Z, _ = K.distance_space(df, base_feats, ln <= T, N_DIM, PARAMS)
tr_idx, te_idx = np.where(ln <= T)[0], np.where(ln > T)[0]
trk, tek = K.fold_features(Z, y, comp, ln, tr_idx, te_idx, mode="past")
knn = pd.concat([trk.set_index(pd.Index(tr_idx)), tek.set_index(pd.Index(te_idx))]).sort_index()
df = pd.concat([df, knn], axis=1)
feats = base_feats + groups["dup"] + groups["grp"] + groups["weak"] + K.KNN_FEATS
print("변수 준비 %ds, 변수 %d개" % (time.time() - t0, len(feats)), flush=True)

tr, te = df.iloc[tr_idx], df.iloc[te_idx]
age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
w = 0.5 ** (age / HALF_LIFE)
lgb_raw = {}
for s in SEEDS:
    m = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
    lgb_raw[s] = m.predict(te[feats])
    print("LightGBM seed %d 완료 (%ds)" % (s, time.time() - t0), flush=True)

pd.to_pickle(dict(df=df[["LNMON", "TARGET"] + feats], feats=feats, pairs=pairs, tr_idx=tr_idx, te_idx=te_idx,
                  lgb_raw=lgb_raw, cat_cols=[c for c in feats if str(df[c].dtype) == "category"]),
             os.path.join(ROOT, "cache", "fold%s.pkl" % FOLD))
print("저장 완료 %ds" % (time.time() - t0), flush=True)

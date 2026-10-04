"""남은 강건성 점검 (cache/fold{X}.pkl 사용, LightGBM 기존 설정 시드 0·1 평균).

④ 동일인 판정 기준 민감도: Fellegi-Sunter 점수 기준을 바꿔 dup/grp/weak 변수와 라벨 전이·그룹 평균을 다시 계산.
   kNN 변수는 캐시 값(점수 300 기준으로 만든 것)을 고정해 동일인 기준의 효과만 분리한다.
⑤ 변수 의존도: 변수 묶음(kNN, 동일인+그룹, 약한 이웃, 원변수 상위 10개, 1위 변수)을 하나씩 빼고 재학습.

사용법: python 22_robustness2.py --fold=A --thresholds=150,200,300,400,500 [--skip-drop]
결과: results/22_robustness2.csv (덧붙임, 이어서 실행 가능)
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
from dup_utils import label_transfer, group_average
import features
import knn_utils as K

warnings.filterwarnings("ignore")
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--%s=" % k)), d)
FOLD = arg("fold", "A")
THRESHOLDS = [int(t) for t in arg("thresholds", "200,300,400").split(",") if t]
SKIP_DROP = "--skip-drop" in sys.argv
SEEDS, HALF_LIFE, ROUNDS = [0, 1], 3, 1200
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, num_threads=8)
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
OUT = os.path.join(ROOT, "results", "22_robustness2.csv")
STRUCT_PREFIX = ("DUP_", "GRP_", "GMAX_", "GMIN_", "GDIFF_", "WEAK_", "MAXSIM_")

t0 = time.time()
T = next(f for f in FOLDS if f["name"] == FOLD)["train_end"]
C = pd.read_pickle(os.path.join(ROOT, "cache", "fold%s.pkl" % FOLD))
cdf, feats, pairs300, tr_idx, te_idx = C["df"], C["feats"], C["pairs"], C["tr_idx"], C["te_idx"]
base_feats = [c for c in feats if not c.startswith(STRUCT_PREFIX) and c not in K.KNN_FEATS]
y, ln = cdf.TARGET.values, cdf.LNMON.values
age = ((T // 100 - ln[tr_idx] // 100) * 12 + T % 100 - ln[tr_idx] % 100)
w = 0.5 ** (age / HALF_LIFE)
pairs_fs = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))
dynamics = pd.read_csv(os.path.join(ROOT, "results", "12_column_dynamics.csv"), index_col=0)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["fold", "test", "variant"])


def is_done(test, variant):
    return ((done.fold == FOLD) & (done.test == test) & (done.variant == str(variant))).any()


def evaluate(test, variant, d, fl, pairs, **extra):
    global done
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    p = np.mean([lgb.train(dict(PARAMS, seed=s), lgb.Dataset(tr[fl], tr.TARGET.values, weight=w), ROUNDS)
                 .predict(te[fl]) for s in SEEDS], 0)
    p, n_lt = label_transfer(p, te_idx, pairs, y, ln, T)
    p, _ = group_average(p, te_idx, pairs)
    r = dict(fold=FOLD, test=test, variant=str(variant), n_feats=len(fl), auc=roc_auc_score(te.TARGET, p),
             logloss=log_loss(te.TARGET, np.clip(p, 1e-6, 1 - 1e-6)), n_transfer=n_lt, **extra)
    for t in THRS:
        r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
    done.to_csv(OUT, index=False)
    print("%s %-9s %-12s 변수 %3d AUC %.4f logloss %.4f err@0.45 %.4f %s (%ds)" % (
        FOLD, test, variant, len(fl), r["auc"], r["logloss"], r["err@0.450"], extra or "", time.time() - t0), flush=True)


# ④ 동일인 판정 기준
knn_cols = cdf[K.KNN_FEATS].reset_index(drop=True)
for thr in THRESHOLDS:
    if is_done("dup_thr", thr):
        continue
    features.FS_THR = thr
    d = cdf[["LNMON", "TARGET"] + base_feats].reset_index(drop=True).copy()
    d, pairs, comp, groups = features.build(d, pairs_fs, dynamics)
    d = pd.concat([d, knn_cols], axis=1)
    fl = base_feats + groups["dup"] + groups["grp"] + groups["weak"] + K.KNN_FEATS
    rows_with_pair = int((d[groups["dup"][:4]].sum(1) > 0).sum())
    evaluate("dup_thr", thr, d, fl, pairs, n_pairs=len(pairs), rows_with_pair=rows_with_pair,
             pos_rate_with_pair=round(float(y[(d[groups["dup"][:4]].sum(1) > 0).values & (ln <= T)].mean()), 3))
features.FS_THR = 300

# ⑤ 변수 의존도 (캐시의 점수 300 기준 변수 그대로)
if not SKIP_DROP:
    imp_model = lgb.train(dict(PARAMS, seed=0, learning_rate=0.05), lgb.Dataset(
        cdf.iloc[tr_idx][feats], y[tr_idx], weight=w), 300)
    imp = pd.Series(imp_model.feature_importance("gain"), feats).sort_values(ascending=False)
    raw_rank = [c for c in imp.index if c in base_feats]
    drops = {
        "none": [],
        "knn": K.KNN_FEATS,
        "dup_grp": [c for c in feats if c.startswith(("DUP_", "GRP_", "GMAX_", "GMIN_", "GDIFF_"))],
        "weak": [c for c in feats if c.startswith(("WEAK_", "MAXSIM_"))],
        "raw_top10": raw_rank[:10],
        "top1": [imp.index[0]],
    }
    print("중요도 1위 %s, 원변수 상위 10개 %s" % (imp.index[0], raw_rank[:10]), flush=True)
    for name, cols in drops.items():
        if is_done("drop", name):
            continue
        evaluate("drop", name, cdf, [c for c in feats if c not in set(cols)], pairs300,
                 dropped=",".join(cols[:10]) if name in ("raw_top10", "top1") else len(cols))

print("총 %ds" % (time.time() - t0))

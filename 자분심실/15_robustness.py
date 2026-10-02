"""강건성 점검 (폴드 A: 실제 평가와 같은 '학습 직후 6개월' 구조).

평가 구간(2023-12~2024-05)을 시나리오별로 바꾸고, 구조 변수를 다시 계산해 세 모델을 비교한다.
- final  : 원변수 + dup/grp/weak + kNN(past99) + 라벨 전이·그룹 평균
- nodup  : 원변수 + kNN(past99)            (동일인 정보 없음)
- base   : 원변수                           (최근 월 가중만)
시나리오
- S0 원래 그대로 (+ final 시드 3개, 예측 기간별 오류)
- S1 중복 없음: 평가 구간에서 동일인 그룹마다 한 행만 남김 (사람 단위 샘플링 가정)
- S2 타깃 비율 25%: 평가 구간의 음성 행을 무작위 제거
- S3 타깃 비율 15%: 평가 구간의 양성 행을 무작위 제거
kNN(past)은 학습 행의 이웃이 학습 기간 안에서만 정해지므로 전체 데이터로 한 번 계산해 재사용한다.

사용법: python 15_robustness.py [--scenarios=S0,S1,S2,S3]
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, weighted_error, fold_periods, ROOT
from dup_utils import label_transfer, group_average, components
import features
import knn_utils as K

warnings.filterwarnings("ignore")
SCEN = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--scenarios=")), ["S0", "S1", "S2", "S3"])
FOLD = FOLDS[0]
T = FOLD["train_end"]
HALF_LIFE, ROUNDS, N_DIM = 3, 1200, 99
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, seed=0, num_threads=8)
THRS = [0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55]
OUT = os.path.join(ROOT, "results", "15_robustness.csv")
rng = np.random.default_rng(42)

t0 = time.time()
full = load().reset_index(drop=True)
base_feats = feature_cols(full)
pairs_all = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))
dynamics = pd.read_csv(os.path.join(ROOT, "results", "12_column_dynamics.csv"), index_col=0)
y_full, ln_full = full.TARGET.values, full.LNMON.values

# --- kNN(past99): 전체 데이터로 한 번
_, pairs0, comp0, _ = features.build(full.copy(), pairs_all, dynamics)
Z, _ = K.distance_space(full, base_feats, ln_full <= T, N_DIM, PARAMS)
tr0, te0 = np.where(ln_full <= T)[0], np.where(ln_full > T)[0]
trk, tek = K.fold_features(Z, y_full, comp0, ln_full, tr0, te0, mode="past")
knn = pd.concat([trk.set_index(pd.Index(tr0)), tek.set_index(pd.Index(te0))]).sort_index()
print("kNN 준비 %ds" % (time.time() - t0), flush=True)


def scenario_rows(name):
    """시나리오별로 남길 원본 행 인덱스."""
    keep = np.ones(len(full), bool)
    te = ln_full > T
    if name == "S1":
        comp_te = pd.Series(comp0[te], index=np.where(te)[0])
        first = comp_te.groupby(comp_te.values).apply(lambda s: rng.choice(s.index.values))
        drop = np.setdiff1d(np.where(te)[0], first.values)
        keep[drop] = False
    elif name in ("S2", "S3"):
        target = 0.25 if name == "S2" else 0.15
        pos, neg = np.where(te & (y_full == 1))[0], np.where(te & (y_full == 0))[0]
        if name == "S2":  # 음성 제거: P / (P + N') = target
            n_keep = int(len(pos) * (1 - target) / target)
            keep[rng.choice(neg, len(neg) - n_keep, replace=False)] = False
        else:              # 양성 제거: P' / (P' + N) = target
            n_keep = int(len(neg) * target / (1 - target))
            keep[rng.choice(pos, len(pos) - n_keep, replace=False)] = False
    return np.where(keep)[0]


def evaluate(name, model, te, p, extra=None):
    r = dict(scenario=name, model=model, n_test=len(te), base_rate=te.TARGET.mean(), auc=roc_auc_score(te.TARGET, p))
    for t in THRS:
        r["err@%.3f" % t], per = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        if t == 0.45:
            for i, e in enumerate(per):
                r["period%d@0.45" % (i + 1)] = e
    r["err_zero"], per0 = weighted_error(te.LNMON, te.TARGET, np.zeros(len(te), int))
    for i, e in enumerate(per0):
        r["period%d_zero" % (i + 1)] = e
    best = min(THRS, key=lambda t: r["err@%.3f" % t])
    r["best_thr"], r["err_best"] = best, r["err@%.3f" % best]
    r.update(extra or {})
    print("  %s %-6s n=%d 양성률 %.3f AUC %.4f err@0.45 %.4f (최적 %.3f @%.3f) 전부0 %.4f" % (
        name, model, len(te), r["base_rate"], r["auc"], r["err@0.450"], r["err_best"], best, r["err_zero"]), flush=True)
    return r


done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["scenario", "model"])
for name in SCEN:
    if (done.scenario == name).any():
        print(name, "이미 완료, 건너뜀", flush=True)
        continue
    t1 = time.time()
    rows_idx = scenario_rows(name)
    df = full.iloc[rows_idx].reset_index(drop=True)
    remap = pd.Series(np.arange(len(rows_idx)), index=rows_idx)
    pa = pairs_all[pairs_all.i.isin(rows_idx) & pairs_all.j.isin(rows_idx)].copy()
    pa["i"], pa["j"] = remap[pa.i.values].values, remap[pa.j.values].values
    df, pairs, comp, groups = features.build(df, pa, dynamics)
    df = pd.concat([df, knn.loc[rows_idx].reset_index(drop=True)], axis=1)
    struct = groups["dup"] + groups["grp"] + groups["weak"]
    y, ln = df.TARGET.values, df.LNMON.values
    tr_idx, te_idx = np.where(ln <= T)[0], np.where(ln > T)[0]
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    print("%s: 평가 구간 %d행 (원래 %d), 동일인 쌍 %d, 준비 %ds" % (name, len(te_idx), len(te0), len(pairs), time.time() - t1), flush=True)
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w = 0.5 ** (age / HALF_LIFE)
    models = {"final": base_feats + struct + K.KNN_FEATS, "nodup": base_feats + K.KNN_FEATS, "base": base_feats}
    rows = []
    for mname, feats in models.items():
        seeds = [0, 1, 2] if (name == "S0" and mname == "final") else [0]
        for s in seeds:
            m = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(tr[feats], tr.TARGET.values, weight=w), ROUNDS)
            p = m.predict(te[feats])
            if mname == "final":
                p, n_lt = label_transfer(p, te_idx, pairs, y, ln, T)
                p, n_g = group_average(p, te_idx, pairs)
            label = mname if s == 0 else "%s_s%d" % (mname, s)
            rows.append(evaluate(name, label, te, p))
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(OUT, index=False)
    print("%s 완료 %ds" % (name, time.time() - t1), flush=True)

pd.set_option("display.width", 250)
cols = ["scenario", "model", "n_test", "base_rate", "auc", "err@0.450", "best_thr", "err_best", "err_zero"]
print("\n=== 요약 ===")
print(done[cols].round(4).to_string(index=False))
s0 = done[(done.scenario == "S0")]
if len(s0):
    print("\n=== S0 예측 기간별 오분류율 @0.45 (전부 0 대비) ===")
    print(s0[["model", "period1@0.45", "period2@0.45", "period3@0.45", "period1_zero", "period2_zero", "period3_zero"]].round(4).to_string(index=False))

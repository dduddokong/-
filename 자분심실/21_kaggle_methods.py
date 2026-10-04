"""캐글 표 데이터 기법 테스트 (cache/fold{A,B,C}.pkl 사용, LightGBM 기존 설정 시드 0·1 평균 기준).

T1 null_imp   : Null importance 변수 선택 (Home Credit). 실제 타깃 중요도 vs 타깃을 섞은 4회의 중요도 →
                실제 gain이 null gain의 75백분위 이하인 변수를 제거. 학습 기간 라벨만 사용.
T2 adv_weight : 적대적 검증 가중치. 학습 행(0) vs 평가 구간 행(1)을 구분하는 분류기의 3겹 OOF 확률로
                w = p/(1-p) (평균 1로 정규화, 0.2~5로 자름)를 최근 월 가중에 곱한다. 라벨 미사용.
T3 pseudo     : 의사 라벨링. 기준 모델 예측이 0.02 미만이면 0, 0.85 초과면 1로 평가 구간 행을 학습에 추가(가중치 0.5).
T4 dart       : LightGBM DART (drop_rate 0.1, skip_drop 0.5, 학습률 0.05 × 800회).
모든 방법은 같은 후처리(라벨 전이 → 그룹 평균)를 거쳐 비교한다.

사용법: python 21_kaggle_methods.py [--folds=A,B,C] [--methods=null_imp,adv_weight,pseudo,dart]
결과: results/21_kaggle_methods.csv (폴드·방법별로 덧붙임)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average

warnings.filterwarnings("ignore")
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--%s=" % k)), d)
RUN = arg("folds", "A,B,C").split(",")
METHODS = arg("methods", "null_imp,adv_weight,pseudo,dart").split(",")
SEEDS = [0, 1]
HALF_LIFE, ROUNDS = 3, 1200
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, num_threads=8)
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
OUT = os.path.join(ROOT, "results", "21_kaggle_methods.csv")


def fit_predict(X, yv, w, Xte, params=PARAMS, rounds=ROUNDS):
    return np.mean([lgb.train(dict(params, seed=s), lgb.Dataset(X, yv, weight=w), rounds).predict(Xte)
                    for s in SEEDS], 0)


done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["fold", "method"])
for fold in RUN:
    t0 = time.time()
    T = next(f for f in FOLDS if f["name"] == fold)["train_end"]
    C = pd.read_pickle(os.path.join(ROOT, "cache", "fold%s.pkl" % fold))
    df, feats, pairs, tr_idx, te_idx = C["df"], C["feats"], C["pairs"], C["tr_idx"], C["te_idx"]
    y, ln = df.TARGET.values, df.LNMON.values
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    age = ((T // 100 - tr.LNMON // 100) * 12 + T % 100 - tr.LNMON % 100).values
    w_rec = 0.5 ** (age / HALF_LIFE)
    rows = []

    def record(method, p_raw, **extra):
        p, _ = label_transfer(p_raw, te_idx, pairs, y, ln, T)
        p, _ = group_average(p, te_idx, pairs)
        r = dict(fold=fold, method=method, auc=roc_auc_score(te.TARGET, p),
                 logloss=log_loss(te.TARGET, np.clip(p, 1e-6, 1 - 1e-6)), **extra)
        for t in THRS:
            r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
        r["sec"] = round(time.time() - t0)
        rows.append(r)
        print("%s %-10s AUC %.4f logloss %.4f err@0.45 %.4f %s (%ds)" % (
            fold, method, r["auc"], r["logloss"], r["err@0.450"], extra or "", r["sec"]), flush=True)

    p_base = np.mean([C["lgb_raw"][s] for s in SEEDS], 0)  # 캐시의 기존 설정 시드 0·1
    if not ((done.fold == fold) & (done.method == "base")).any():
        record("base", p_base)

    for method in METHODS:
        if ((done.fold == fold) & (done.method == method)).any():
            print(fold, method, "이미 완료", flush=True)
            continue
        if method == "null_imp":
            sp = dict(PARAMS, learning_rate=0.05)
            ds = lambda lab: lgb.Dataset(tr[feats], lab, weight=w_rec)
            act = pd.Series(lgb.train(dict(sp, seed=0), ds(tr.TARGET.values), 300).feature_importance("gain"), feats)
            rng = np.random.default_rng(0)
            null = np.column_stack([lgb.train(dict(sp, seed=k), ds(rng.permutation(tr.TARGET.values)), 300)
                                    .feature_importance("gain") for k in range(4)])
            thr = np.percentile(null, 75, axis=1)
            keep = [c for c, a, t in zip(feats, act.values, thr) if a > t]
            dropped = [c for c in feats if c not in keep]
            record(method, fit_predict(tr[keep], tr.TARGET.values, w_rec, te[keep]), n_keep=len(keep),
                   dropped_knn_dup=sum(c.startswith(("KNN", "DUP", "GRP", "WEAK", "GM", "GD", "MAXSIM")) for c in dropped))
        elif method == "adv_weight":
            Xa = pd.concat([tr[feats], te[feats]])
            ya = np.r_[np.zeros(len(tr)), np.ones(len(te))]
            oof = np.zeros(len(Xa))
            ap = dict(PARAMS, learning_rate=0.05, seed=0)
            for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(Xa, ya):
                oof[b] = lgb.train(ap, lgb.Dataset(Xa.iloc[a], ya[a]), 200).predict(Xa.iloc[b])
            pa = np.clip(oof[:len(tr)], 1e-3, 1 - 1e-3)
            w_adv = pa / (1 - pa)
            w_adv = np.clip(w_adv / w_adv.mean(), 0.2, 5)
            record(method, fit_predict(tr[feats], tr.TARGET.values, w_rec * w_adv, te[feats]),
                   adv_auc=round(roc_auc_score(ya, oof), 4))
        elif method == "pseudo":
            sel = (p_base < 0.02) | (p_base > 0.85)
            pl = (p_base[sel] > 0.5).astype(float)
            X2 = pd.concat([tr[feats], te[feats].iloc[np.where(sel)[0]]])
            y2 = np.r_[tr.TARGET.values, pl]
            w2 = np.r_[w_rec, np.full(sel.sum(), 0.5)]
            record(method, fit_predict(X2, y2, w2, te[feats]), n_pseudo=int(sel.sum()),
                   pseudo_acc=round(float((te.TARGET.values[sel] == pl).mean()), 4))
        elif method == "dart":
            dp = dict(PARAMS, boosting="dart", learning_rate=0.05, drop_rate=0.1, skip_drop=0.5)
            record(method, fit_predict(tr[feats], tr.TARGET.values, w_rec, te[feats], dp, 800))
        res_new = pd.DataFrame(rows)
        done = pd.concat([done[~((done.fold == fold) & done.method.isin(res_new.method))], res_new], ignore_index=True)
        done.to_csv(OUT, index=False)
    if rows:
        done = pd.concat([done[~((done.fold == fold) & done.method.isin([r["method"] for r in rows]))],
                          pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(OUT, index=False)

pd.set_option("display.width", 220)
print("\n=== 폴드별 err@0.450 (base 대비 %p) ===")
p = done.pivot_table(index="method", columns="fold", values="err@0.450")
print(((p - p.loc["base"]) * 100).round(2).to_string())
print("\n=== 세 폴드 평균 ===")
print(done.groupby("method")[["auc", "logloss"] + ["err@%.3f" % t for t in THRS]].mean().round(4).to_string())

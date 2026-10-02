"""XGBoost / CatBoost를 LightGBM과 같은 시간 순서 백테스트로 평가하고, 부스팅 앙상블 효과를 확인.

- 공통: 최근 월 가중(반감기 3개월), 결측은 각 라이브러리 기본 처리, 범주형 4개는 범주형으로 사용.
- 반복수: MAX_ROUNDS까지 학습 후 STEP 간격으로 평가해 세 폴드 평균 로그손실이 최소인 지점을 고른다.
- LightGBM 예측은 02 실험의 decay_hl3 결과(cache/02_lgb_preds.pkl)를 재사용한다.

사용법: python 07_xgb_cat_backtest.py [--smoke]   (--smoke: 표본/짧은 학습으로 코드만 점검)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import load, feature_cols, FOLDS, CAT_COLS, weighted_error, ROOT

warnings.filterwarnings("ignore")
SMOKE = "--smoke" in sys.argv
HALF_LIFE = 3
MAX_ROUNDS, STEP = (60, 20) if SMOKE else (2000, 100)
THREADS = 2 if SMOKE else 8
CHECKPOINTS = list(range(STEP, MAX_ROUNDS + 1, STEP))
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", eta=0.05,
                  max_depth=6, min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                  reg_lambda=10, max_bin=256, nthread=THREADS, seed=0)
CAT_PARAMS = dict(loss_function="Logloss", learning_rate=0.05, depth=6, l2_leaf_reg=10,
                  rsm=0.5, bootstrap_type="Bernoulli", subsample=0.8, border_count=254,
                  nan_mode="Min", thread_count=THREADS, random_seed=0, verbose=0)


def weights(lnmon, train_end):
    age = (train_end // 100 - lnmon // 100) * 12 + train_end % 100 - lnmon % 100
    return 0.5 ** (np.asarray(age) / HALF_LIFE)


def run_xgb(tr, te, feats, w):
    dtr = xgb.DMatrix(tr[feats], tr.TARGET.values, weight=w, enable_categorical=True)
    dte = xgb.DMatrix(te[feats], enable_categorical=True)
    b = xgb.train(XGB_PARAMS, dtr, MAX_ROUNDS)
    return [b.predict(dte, iteration_range=(0, k)) for k in CHECKPOINTS]


def run_cat(tr, te, feats, w):
    def pool(d, y=None, w=None):
        X = d[feats].copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        return Pool(X, y, weight=w, cat_features=CAT_COLS)
    m = CatBoostClassifier(iterations=MAX_ROUNDS, **CAT_PARAMS).fit(pool(tr, tr.TARGET.values, w))
    pte = pool(te)
    return [m.predict_proba(pte, ntree_end=k)[:, 1] for k in CHECKPOINTS]


def summarize(name, fold_rows, checkpoints=None):
    """폴드별 체크포인트 예측 → 평균 로그손실 최소 반복수에서의 지표."""
    curve = np.mean([[log_loss(r["y"], p) for p in r["ps"]] for r in fold_rows], axis=0)
    k = int(np.argmin(curve))
    out = dict(model=name, rounds=checkpoints[k] if checkpoints else None, logloss=curve[k])
    out["auc"] = np.mean([roc_auc_score(r["y"], r["ps"][k]) for r in fold_rows])
    for t in THRS:
        out["err@%.3f" % t] = np.mean([weighted_error(r["lnmon"], r["y"], (r["ps"][k] > t).astype(int))[0]
                                       for r in fold_rows])
    return out, {r["fold"]: r["ps"][k] for r in fold_rows}


df = load()
if SMOKE:
    df = df.sample(15000, random_state=0)
feats = feature_cols(df)

results, best_preds = [], {}
for name, runner in [("xgb", run_xgb), ("cat", run_cat)]:
    fold_rows = []
    for f in FOLDS:
        tr, te = df[df.LNMON <= f["train_end"]], df[df.LNMON > f["train_end"]]
        t0 = time.time()
        ps = runner(tr, te, feats, weights(tr.LNMON.values, f["train_end"]))
        fold_rows.append(dict(fold=f["name"], y=te.TARGET.values, lnmon=te.LNMON.values, ps=ps))
        print(name, f["name"], "%ds" % (time.time() - t0),
              "logloss by checkpoint:", np.round([log_loss(te.TARGET, p) for p in ps[::5]], 4), flush=True)
    r, best_preds[name] = summarize(name, fold_rows, CHECKPOINTS)
    results.append(r)
    print(r, flush=True)

# LightGBM(02 decay_hl3) 예측과 결합
lgb_path = os.path.join(ROOT, "cache", "02_lgb_preds.pkl")
if not SMOKE and os.path.exists(lgb_path):
    lp = pd.read_pickle(lgb_path).query("exp == 'decay_hl3'")
    best_preds["lgb"] = {f["name"]: lp[lp.fold == f["name"]].p.values for f in FOLDS}
    ys = {f["name"]: df[df.LNMON > f["train_end"]].TARGET.values for f in FOLDS}
    lns = {f["name"]: df[df.LNMON > f["train_end"]].LNMON.values for f in FOLDS}
    combos = {"lgb": ["lgb"], "lgb+xgb": ["lgb", "xgb"], "lgb+cat": ["lgb", "cat"],
              "xgb+cat": ["xgb", "cat"], "lgb+xgb+cat": ["lgb", "xgb", "cat"]}
    for cname, members in combos.items():
        rows = [dict(fold=fn, y=ys[fn], lnmon=lns[fn], ps=[np.mean([best_preds[m][fn] for m in members], axis=0)])
                for fn in ys]
        r, _ = summarize("ens:" + cname, rows)
        results.append(r)
    print("\n=== 모델 간 예측 상관 (폴드 C, 순위 기준) ===")
    print(pd.DataFrame({m: pd.Series(best_preds[m]["C"]).rank() for m in ["lgb", "xgb", "cat"]}).corr().round(4))
    pd.to_pickle(best_preds, os.path.join(ROOT, "cache", "07_best_preds.pkl"))

res = pd.DataFrame(results)
if not SMOKE:
    res.to_csv(os.path.join(ROOT, "results", "07_xgb_cat.csv"), index=False)
pd.set_option("display.width", 220)
print("\n=== 결과 (세 폴드 평균) ===")
print(res.round(4).to_string(index=False))

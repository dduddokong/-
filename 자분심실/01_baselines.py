"""기준선 비교: 전부0 / L2 로지스틱 / 랜덤포레스트 / LightGBM을 같은 시간 순서 백테스트에서 평가."""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from kcb_common import load, feature_cols, FOLDS, CAT_COLS, weighted_error, ROOT

warnings.filterwarnings("ignore")
THRS = np.round(np.arange(0.35, 0.651, 0.05), 2)

df = load()
feats = feature_cols(df)
num = [c for c in feats if c not in CAT_COLS]


def design_linear(d):
    return pd.concat([d[num], pd.get_dummies(d[CAT_COLS], dtype=np.float32)], axis=1)


def design_tree(d):
    X = d[num].fillna(-999).copy()
    for c in CAT_COLS:
        X[c] = d[c].cat.codes
    return X


MODELS = {
    "logit_l2": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=0.01, max_iter=3000)),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                         max_samples=0.5, n_jobs=8, random_state=0),
    "lgb": None,
}
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                  verbose=-1, seed=0, num_threads=8)
LGB_ROUNDS = 1200

rows, preds = [], []
for f in FOLDS:
    tr = df[df.LNMON <= f["train_end"]]
    te = df[df.LNMON > f["train_end"]]
    ytr, yte = tr.TARGET.values, te.TARGET.values
    for name in ["zero", "logit_l2", "rf", "lgb"]:
        t0 = time.time()
        if name == "zero":
            p = np.zeros(len(te))
        elif name == "logit_l2":
            Xtr = design_linear(tr)
            Xte = design_linear(te).reindex(columns=Xtr.columns, fill_value=0)
            p = MODELS[name]().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        elif name == "rf":
            p = MODELS[name]().fit(design_tree(tr), ytr).predict_proba(design_tree(te))[:, 1]
        else:
            m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[feats], ytr), LGB_ROUNDS)
            p = m.predict(te[feats])
        r = dict(fold=f["name"], model=name, sec=round(time.time() - t0),
                 auc=roc_auc_score(yte, p) if p.std() > 0 else 0.5)
        for t in THRS:
            r["err@%.2f" % t], per = weighted_error(te.LNMON, yte, (p > t).astype(int))
            if t == 0.5:
                r["periods@0.50"] = [round(e, 4) for e in per]
        rows.append(r)
        preds.append(pd.DataFrame(dict(fold=f["name"], model=name, LNMON=te.LNMON.values, y=yte, p=p)))
        print(f["name"], name, "AUC %.4f  err@0.5 %.4f  (%ds)" % (r["auc"], r["err@0.50"], r["sec"]), flush=True)

res = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
res.to_csv(os.path.join(ROOT, "results", "01_baselines.csv"), index=False)
pd.concat(preds).to_pickle(os.path.join(ROOT, "cache", "01_baseline_preds.pkl"))

pd.set_option("display.width", 200)
print("\n=== 폴드별 ===")
print(res[["fold", "model", "auc", "err@0.50", "periods@0.50"]].round(4).to_string(index=False))
print("\n=== 폴드 평균 (임계값별 가중 오분류율) ===")
cols = ["auc"] + ["err@%.2f" % t for t in THRS]
print(res.groupby("model")[cols].mean().round(4).to_string())

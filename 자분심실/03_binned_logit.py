"""구간화 로지스틱(스코어카드 방식): 변수별 얕은 나무로 임계점을 찾고 더미/WoE로 바꾼 뒤 정규화 로지스틱.

- 임계점, WoE는 폴드의 학습 기간에서만 계산한다(정보누수 방지).
- 결측은 별도 구간, 범주형은 범주 자체가 구간.
- C는 학습 기간의 마지막 2개월을 내부 검증으로 써서 고르고, 학습 기간 전체로 재적합한다.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from kcb_common import load, feature_cols, FOLDS, CAT_COLS, weighted_error, add_months, ROOT

warnings.filterwarnings("ignore")
MAX_BINS = 6          # 결측 구간 제외 최대 구간 수
MIN_BIN_FRAC = 0.02   # 구간 최소 표본 비율 (전체 학습 행 기준)
THRS = np.round(np.arange(0.40, 0.601, 0.025), 3)
C_GRID = {"dummy_l2": [0.003, 0.01, 0.03, 0.1], "dummy_l1": [0.01, 0.03, 0.1], "woe_l2": [0.01, 0.1, 1.0]}


class Binner:
    def fit(self, df, y, feats):
        self.feats, self.cuts, self.woe = [], {}, {}
        prior = np.log(y.mean() / (1 - y.mean()))
        for c in feats:
            b = self._raw_bins(df[c], y, c, fit=True)
            if b is None:
                continue
            nb = b.max() + 1
            pos = np.bincount(b, weights=y, minlength=nb)
            cnt = np.bincount(b, minlength=nb)
            self.woe[c] = np.log((pos + 0.5) / (cnt - pos + 0.5)) - prior
            self.feats.append(c)
        return self

    def _raw_bins(self, s, y, c, fit=False):
        if c in CAT_COLS:
            codes = s.cat.codes.values.astype(int)
            return np.where(codes < 0, s.cat.categories.size, codes)
        x = s.values.astype(np.float64)
        miss = np.isnan(x)
        if fit:
            ok = ~miss
            min_leaf = int(MIN_BIN_FRAC * len(x))  # 전체 행 기준: 결측이 많은 변수의 과분할 방지
            th = np.array([])
            if ok.sum() >= 2 * min_leaf:
                t = DecisionTreeClassifier(max_leaf_nodes=MAX_BINS, min_samples_leaf=min_leaf, random_state=0)
                t.fit(x[ok].reshape(-1, 1), y[ok])
                th = np.sort(t.tree_.threshold[t.tree_.feature >= 0])
            if th.size == 0 and min(miss.sum(), (~miss).sum()) < min_leaf:
                return None  # 구간이 하나뿐인 변수는 제외
            self.cuts[c] = th
        th = self.cuts[c]
        return np.where(miss, th.size + 1, np.searchsorted(th, x, side="right"))

    def transform_bins(self, df):
        out = []
        for c in self.feats:
            b = self._raw_bins(df[c], None, c)
            out.append(np.minimum(b, self.woe[c].size - 1))  # 학습 때 없던 범주는 마지막 구간
        return np.column_stack(out)

    def dummies(self, B):
        offs = np.cumsum([0] + [self.woe[c].size for c in self.feats])
        rows = np.repeat(np.arange(B.shape[0]), B.shape[1])
        cols = (B + offs[:-1]).ravel()
        return sp.csr_matrix((np.ones(cols.size, np.float32), (rows, cols)), shape=(B.shape[0], offs[-1]))

    def woe_matrix(self, B):
        return np.column_stack([self.woe[c][B[:, j]] for j, c in enumerate(self.feats)])


def make_model(kind, C):
    if kind == "dummy_l1":
        return LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=500)
    return LogisticRegression(C=C, max_iter=3000)


def design(kind, binner, d):
    B = binner.transform_bins(d)
    return binner.woe_matrix(B) if kind == "woe_l2" else binner.dummies(B)


def fit_predict(kind, tr, te, feats):
    """내부 시간 검증으로 C 선택 → 학습 기간 전체로 재적합 → 테스트 예측."""
    inner_end = add_months(tr.LNMON.max(), -2)
    itr, iva = tr[tr.LNMON <= inner_end], tr[tr.LNMON > inner_end]
    bi = Binner().fit(itr, itr.TARGET.values, feats)
    Xi, Xv = design(kind, bi, itr), design(kind, bi, iva)
    scores = {}
    for C in C_GRID[kind]:
        p = make_model(kind, C).fit(Xi, itr.TARGET.values).predict_proba(Xv)[:, 1]
        scores[C] = ((p > 0.5) != iva.TARGET.values).mean() - 1e-3 * roc_auc_score(iva.TARGET, p)
    best_C = min(scores, key=scores.get)
    b = Binner().fit(tr, tr.TARGET.values, feats)
    m = make_model(kind, best_C).fit(design(kind, b, tr), tr.TARGET.values)
    return m.predict_proba(design(kind, b, te))[:, 1], best_C, len(b.feats), m


df = load()
feats = feature_cols(df)
lgb_path = os.path.join(ROOT, "cache", "01_baseline_preds.pkl")
lgb_preds = pd.read_pickle(lgb_path).query("model == 'lgb'") if os.path.exists(lgb_path) else None

rows, preds = [], []


def record(fold, name, te, p, **extra):
    r = dict(fold=fold, model=name, auc=roc_auc_score(te.TARGET, p), mean_p=p.mean(), **extra)
    for t in THRS:
        r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    rows.append(r)
    preds.append(pd.DataFrame(dict(fold=fold, model=name, LNMON=te.LNMON.values, y=te.TARGET.values, p=p)))
    print(fold, name, "AUC %.4f err@0.5 %.4f" % (r["auc"], r["err@0.500"]), extra, flush=True)


for f in FOLDS:
    tr, te = df[df.LNMON <= f["train_end"]], df[df.LNMON > f["train_end"]]
    for kind in ["woe_l2", "dummy_l2", "dummy_l1"]:
        t0 = time.time()
        p, C, nf, m = fit_predict(kind, tr, te, feats)
        extra = dict(C=C, n_feats=nf, sec=round(time.time() - t0))
        if kind == "dummy_l1":
            extra["nonzero"] = int((m.coef_ != 0).sum())
        record(f["name"], kind, te, p, **extra)
        if lgb_preds is not None:
            pl = lgb_preds[lgb_preds.fold == f["name"]].p.values
            assert len(pl) == len(p)
            record(f["name"], "ens_lgb+" + kind, te, 0.5 * (p + pl))

res = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
res.to_csv(os.path.join(ROOT, "results", "03_binned_logit.csv"), index=False)
pd.concat(preds).to_pickle(os.path.join(ROOT, "cache", "03_binned_preds.pkl"))
pd.set_option("display.width", 220)
cols = ["auc", "mean_p"] + ["err@%.3f" % t for t in THRS]
print("\n=== 모델별 폴드 평균 ===")
print(res.groupby("model")[cols].mean().round(4).to_string())

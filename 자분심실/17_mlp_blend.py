"""MLP 학습과 LightGBM 블렌딩 (--fold=A|B|C, cache/fold{X}.pkl 사용).

전처리 (학습 행으로만 적합):
- 수치 변수: QuantileTransformer(정규분포) → [-5, 5]로 자름 → 결측은 0 (+ 결측 표시 변수)
- 범주형 4개: 원-핫
학습:
- 내부 검증: 학습 기간 마지막 달로 조기 종료 에포크를 정하고, 학습 기간 전체로 그 에포크만큼 재학습
- 손실: 최근 월 가중(반감기 3개월) 이진 교차 엔트로피
블렌딩: p = (1-w)·LightGBM(시드 평균) + w·MLP(시드 평균) → 라벨 전이 → 그룹 평균 → 임계값

사용법: python 17_mlp_blend.py [--seeds=3] [--hidden=512,256,128] [--dropout=0.2] [--lr=1e-3] [--wd=1e-4]
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import FOLDS, weighted_error, ROOT
from dup_utils import label_transfer, group_average

warnings.filterwarnings("ignore")
arg = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith("--%s=" % k)), d)
N_SEEDS = int(arg("seeds", 3))
HIDDEN = [int(h) for h in arg("hidden", "512,256,128").split(",")]
DROPOUT = float(arg("dropout", 0.2))
LR = float(arg("lr", 1e-3))
WD = float(arg("wd", 1e-4))
BATCH, MAX_EPOCHS, PATIENCE = 1024, 60, 5
HALF_LIFE = 3
FOLD = arg("fold", "A")
T = next(f for f in FOLDS if f["name"] == FOLD)["train_end"]
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
torch.set_num_threads(8)

t0 = time.time()
C = pd.read_pickle(os.path.join(ROOT, "cache", "fold%s.pkl" % FOLD))
df, feats, pairs, tr_idx, te_idx = C["df"], C["feats"], C["pairs"], C["tr_idx"], C["te_idx"]
cat_cols = C["cat_cols"]
num_cols = [c for c in feats if c not in cat_cols]
y, ln = df.TARGET.values, df.LNMON.values

# --- 전처리
Xn = df[num_cols].values.astype(np.float32)
has_na = np.isnan(Xn[tr_idx]).any(0)
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0)
qt.fit(Xn[tr_idx])
Xq = np.clip(qt.transform(Xn), -5, 5)
na_ind = np.isnan(Xn[:, has_na]).astype(np.float32)
Xq = np.nan_to_num(Xq, nan=0.0).astype(np.float32)
Xc = pd.get_dummies(df[cat_cols].astype(str), dtype=np.float32).values
X = np.hstack([Xq, na_ind, Xc]).astype(np.float32)
del Xn, Xq
print("입력 %d차원 (수치 %d, 결측 표시 %d, 범주 원-핫 %d), 준비 %ds" %
      (X.shape[1], len(num_cols), na_ind.shape[1], Xc.shape[1], time.time() - t0), flush=True)

age = (T // 100 - ln // 100) * 12 + T % 100 - ln % 100
w_all = (0.5 ** (age / HALF_LIFE)).astype(np.float32)


def make_model(d, seed):
    torch.manual_seed(seed)
    layers, prev = [], d
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(DROPOUT)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def predict(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 8192):
            out.append(torch.sigmoid(model(torch.from_numpy(X[idx[i:i + 8192]]))).squeeze(1).numpy())
    return np.concatenate(out)


def train(idx, seed, epochs=None, val_idx=None):
    """epochs가 None이면 val_idx로 조기 종료하고 (모델, 최적 에포크)를 반환."""
    rng = np.random.default_rng(seed)
    model = make_model(X.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    Xt, yt, wt = torch.from_numpy(X[idx]), torch.from_numpy(y[idx].astype(np.float32)), torch.from_numpy(w_all[idx])
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    for ep in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(idx)))
        for i in range(0, len(idx), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = (lossf(model(Xt[b]).squeeze(1), yt[b]) * wt[b]).sum() / wt[b].sum()
            loss.backward()
            opt.step()
        if epochs is None:
            vl = log_loss(y[val_idx], predict(model, val_idx))
            if vl < best - 1e-5:
                best, best_ep, bad = vl, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if epochs is None:
        model.load_state_dict(best_state)
        return model, best_ep, best
    return model


mlp_raw = []
inner_tr = tr_idx[ln[tr_idx] < T]
inner_va = tr_idx[ln[tr_idx] == T]
for s in range(N_SEEDS):
    t1 = time.time()
    _, ep, vl = train(inner_tr, s, val_idx=inner_va)
    model = train(tr_idx, s, epochs=ep)
    p = predict(model, te_idx)
    mlp_raw.append(p)
    print("MLP seed %d: 최적 에포크 %d (내부 검증 logloss %.4f), 테스트 AUC %.4f, %ds" %
          (s, ep, vl, roc_auc_score(y[te_idx], p), time.time() - t1), flush=True)
mlp = np.mean(mlp_raw, 0)
lgb_seeds = [C["lgb_raw"][s] for s in sorted(C["lgb_raw"])]
lgbm = np.mean(lgb_seeds, 0)

te = df.iloc[te_idx]
rows = []


def evaluate(name, p_raw):
    p, _ = label_transfer(p_raw, te_idx, pairs, y, ln, T)
    p, _ = group_average(p, te_idx, pairs)
    r = dict(model=name, auc=roc_auc_score(te.TARGET, p), logloss=log_loss(te.TARGET, np.clip(p, 1e-6, 1 - 1e-6)))
    for t in THRS:
        r["err@%.3f" % t], _ = weighted_error(te.LNMON, te.TARGET, (p > t).astype(int))
    rows.append(r)


for i, p in enumerate(lgb_seeds):
    evaluate("lgb_seed%d" % i, p)
evaluate("lgb_avg3", lgbm)
for i, p in enumerate(mlp_raw):
    evaluate("mlp_seed%d" % i, p)
evaluate("mlp_avg%d" % N_SEEDS, mlp)
for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    evaluate("blend_w%.1f" % w, (1 - w) * lgbm + w * mlp)

res = pd.DataFrame(rows)
tag = FOLD + "_h%s_d%.1f_lr%g_wd%g_s%d" % ("-".join(map(str, HIDDEN)), DROPOUT, LR, WD, N_SEEDS)
res.to_csv(os.path.join(ROOT, "results", "17_mlp_blend_%s.csv" % tag), index=False)
pd.to_pickle(dict(mlp_raw=mlp_raw), os.path.join(ROOT, "cache", "17_mlp_raw_%s.pkl" % tag))
pd.set_option("display.width", 220)
print("\nLightGBM-MLP 예측 순위 상관: %.4f" % pd.Series(lgbm).corr(pd.Series(mlp), method="spearman"))
print("\n=== 폴드 %s 결과 (후처리 후) ===" % FOLD)
print(res.round(4).to_string(index=False))
print("총 %ds" % (time.time() - t0))

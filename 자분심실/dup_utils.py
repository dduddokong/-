"""동일인 중복 쌍 기반 변수와 후처리 (라벨 전이, 그룹 평균, 라벨 분포 보정)."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DUP_FEATS = ["DUP_SAME", "DUP_PREV", "DUP_NEXT", "DUP_GAP2", "DUP_MAXSIM"]


def month_index(lnmon):
    lnmon = np.asarray(lnmon)
    return (lnmon // 100) * 12 + lnmon % 100


def add_dup_features(df, pairs, sim_col="sim_stable"):
    """행 단위 짝 개수(같은 달/이전/다음/2개월)와 최대 유사도. df는 0..n-1 인덱스여야 한다."""
    n = len(df)
    mi = month_index(df.LNMON.values)
    out = {c: np.zeros(n, np.float32) for c in DUP_FEATS}
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:
        d = mi[b] - mi[a]
        np.add.at(out["DUP_SAME"], a[d == 0], 1)
        np.add.at(out["DUP_PREV"], a[d == -1], 1)
        np.add.at(out["DUP_NEXT"], a[d == 1], 1)
        np.add.at(out["DUP_GAP2"], a[np.abs(d) == 2], 1)
        np.maximum.at(out["DUP_MAXSIM"], a, pairs[sim_col].values.astype(np.float32))
    for c, v in out.items():
        df[c] = v
    return df


def components(n, pairs):
    g = coo_matrix((np.ones(len(pairs)), (pairs.i.values, pairs.j.values)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def label_transfer(p, te_idx, pairs, y, lnmon, train_end):
    """테스트 행의 짝이 학습 기간에 있으면 짝 타깃 평균으로 예측을 덮어쓴다."""
    p = p.copy()
    pos = pd.Series(np.arange(len(te_idx)), index=te_idx)
    s, c = np.zeros(len(te_idx)), np.zeros(len(te_idx))
    for a, b in [(pairs.i.values, pairs.j.values), (pairs.j.values, pairs.i.values)]:
        k = np.isin(a, te_idx) & (lnmon[b] <= train_end)
        np.add.at(s, pos[a[k]].values, y[b[k]])
        np.add.at(c, pos[a[k]].values, 1)
    hit = c > 0
    p[hit] = s[hit] / c[hit]
    return p, int(hit.sum())


def group_average(p, te_idx, pairs):
    """테스트 기간 안에서 연결된 행끼리 예측 평균."""
    pos = pd.Series(np.arange(len(te_idx)), index=te_idx)
    k = np.isin(pairs.i.values, te_idx) & np.isin(pairs.j.values, te_idx)
    if not k.any():
        return p, 0
    sub = pd.DataFrame(dict(i=pos[pairs.i.values[k]].values, j=pos[pairs.j.values[k]].values))
    comp = components(len(p), sub)
    avg = pd.Series(p).groupby(comp).transform("mean").values
    return avg, int((np.bincount(comp)[comp] > 1).sum())


def em_prior(p, prior_train, n_iter=100, tol=1e-6):
    """Saerens et al. (2002): 라벨 없는 예측확률로 새 기간의 타깃 비율을 추정하고 사후확률을 보정."""
    pi = prior_train
    for _ in range(n_iter):
        q = adjust(p, prior_train, pi)
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi, adjust(p, prior_train, pi)


def adjust(p, prior_old, prior_new):
    a = p * prior_new / prior_old
    b = (1 - p) * (1 - prior_new) / (1 - prior_old)
    return a / (a + b)

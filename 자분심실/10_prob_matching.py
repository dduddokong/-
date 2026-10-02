"""Fellegi-Sunter 확률적 매칭으로 동일인 쌍을 다시 찾는다 (라벨 미사용).

점수 = Σ_k [일치: log(m_k / f_k(v)),  불일치: log((1-m_k) / (1-u_k))]
- m_k : 동일인일 때 변수 k가 일치할 확률 (08의 확실한 쌍에서 추정)
- f_k(v): 값 v의 전체 빈도 (흔한 값이 일치하면 증거가 약하다)
- u_k : 무작위 두 행이 변수 k에서 일치할 확률 = Σ_v f_k(v)^2
후보는 LSH(안정 변수 조합)로 모은다. 결과: cache/dup_pairs_fs.pkl
"""
import os
import numpy as np
import pandas as pd
from kcb_common import load, feature_cols, CAT_COLS, ROOT
import importlib

m08 = importlib.import_module("08_dup_matching")

M_CLIP = (0.5, 0.999)
F_FLOOR = 1e-5


def fs_weights(df, feats, sure_pairs):
    X = df[feats].values.astype(np.float64)
    Xn = np.where(np.isnan(X), -1e18, X)  # 결측끼리 일치로 취급
    a, b = Xn[sure_pairs[:, 0]], Xn[sure_pairs[:, 1]]
    m = np.clip((a == b).mean(0), *M_CLIP)
    freq_maps, u = [], np.zeros(len(feats))
    for k in range(len(feats)):
        vc = pd.Series(Xn[:, k]).value_counts(normalize=True)
        freq_maps.append(vc)
        u[k] = float((vc.values ** 2).sum())
    return Xn, m, u, freq_maps


def fs_score(Xn, pairs, m, u, freq_maps):
    a, b = Xn[pairs[:, 0]], Xn[pairs[:, 1]]
    agree = a == b
    score = np.zeros(len(pairs))
    for k in range(Xn.shape[1]):
        f = np.maximum(freq_maps[k].reindex(a[:, k]).values, F_FLOOR)
        w_agree = np.log(m[k] / f)
        w_dis = np.log((1 - m[k]) / max(1 - u[k], 1e-6))
        score += np.where(agree[:, k], w_agree, w_dis)
    return score


def find_fs_pairs(df, verbose=True):
    df = df.reset_index(drop=True)
    base, stable = m08.find_pairs(df, verbose=verbose)
    sure = base[(base.sim_stable > 0.95)][["i", "j"]].values
    feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    nun = df[feats].nunique()
    X = df[feats].values
    informative = ((X == 0) | np.isnan(X)).mean(0) < 0.3
    keys = [c for c, ok in zip(feats, informative) if ok and nun[c] > 100 and c in set(stable)]
    extra = m08.lsh_pairs(df, keys, 200, 3, 2)  # 후보를 넓게
    mi = m08.month_index(df.LNMON.values)
    cand = np.unique(np.vstack([base[["i", "j"]].values, extra]), axis=0)
    gap = np.abs(mi[cand[:, 0]] - mi[cand[:, 1]])
    cand, gap = cand[gap <= m08.MAX_GAP], gap[gap <= m08.MAX_GAP]
    Xn, m, u, fm = fs_weights(df, feats, sure)
    out = pd.DataFrame(dict(i=cand[:, 0], j=cand[:, 1], gap=gap, fs=fs_score(Xn, cand, m, u, fm)))
    out = out.merge(base[["i", "j", "sim_stable"]], on=["i", "j"], how="left")
    if verbose:
        print("후보 %d쌍 (기존 %d + 추가 LSH %d)" % (len(cand), len(base), len(extra)))
    return out


if __name__ == "__main__":
    df = load().reset_index(drop=True)
    pairs = find_fs_pairs(df)
    y = df.TARGET.values
    pairs["agree"] = y[pairs.i] == y[pairs.j]
    qs = [-np.inf, 0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, np.inf]
    pairs["bin"] = pd.cut(pairs.fs, qs)
    print("\nFS 점수 구간별: 쌍 수 / 타깃 일치율 (진단용)")
    print(pairs.groupby("bin").agg(n=("agree", "size"), agree=("agree", "mean"),
                                   gap0=("gap", lambda g: (g == 0).sum()), gap1=("gap", lambda g: (g == 1).sum()),
                                   gap2=("gap", lambda g: (g == 2).sum())).round(3).to_string())
    print("\n점수 히스토그램 (라벨 없이 임계값을 고르기 위한 분포):")
    h, e = np.histogram(pairs.fs.clip(-500, 3000), bins=35)
    for c, lo in zip(h, e[:-1]):
        print("%6d %s" % (lo, "#" * int(60 * np.log1p(c) / np.log1p(h.max()))), c)
    old = pairs.sim_stable > 0.9
    print("\n기존 기준(sim_stable>0.9) %d쌍" % old.sum())
    pairs.drop(columns="bin").to_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))

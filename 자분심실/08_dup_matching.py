"""동일인으로 보이는 거의-중복 행 매칭 (라벨 미사용).

1) 1차 LSH: 값 종류가 많은 변수 3개 조합이 완전히 같은 행 쌍 → 전체 일치율 0.9 초과 쌍을 '확실한 쌍'으로 본다.
2) 확실한 쌍에서 거의 바뀌지 않는(불일치율 < 2%) 변수를 '안정 변수'로 고른다.
3) 2차 LSH: 안정 변수 조합으로 다시 후보를 모으고, 안정 변수 일치율로 판정한다.
결과: cache/dup_pairs.pkl (i, j, 월 차이, 일치율). 타깃은 매칭 품질 진단에만 쓴다.

데이터가 바뀌면(테스트셋 결합) load_fn만 바꿔 같은 절차를 다시 돌리면 된다.
"""
import os
import itertools
import numpy as np
import pandas as pd
from kcb_common import load, feature_cols, CAT_COLS, ROOT

MAX_GAP = 2           # 매칭할 최대 월 차이
MAX_BUCKET = 8        # 버킷이 이보다 크면 흔한 값이라 보고 건너뜀


def month_index(lnmon):
    lnmon = np.asarray(lnmon)
    return (lnmon // 100) * 12 + lnmon % 100


def lsh_pairs(df, cols, n_bands, k, seed):
    rng = np.random.default_rng(seed)
    pairs = set()
    for _ in range(n_bands):
        band = list(rng.choice(cols, k, replace=False))
        for idx in df.groupby(band, sort=False).indices.values():
            if 1 < len(idx) <= MAX_BUCKET:
                pairs.update(itertools.combinations(idx, 2))
    return np.array(sorted(pairs))


def match_rate(X, pairs):
    a, b = X[pairs[:, 0]], X[pairs[:, 1]]
    return ((a == b) | (np.isnan(a) & np.isnan(b))).mean(1)


def find_pairs(df, verbose=True):
    df = df.reset_index(drop=True)
    feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    X = df[feats].values.astype(np.float64)
    mi = month_index(df.LNMON.values)
    nun = df[feats].nunique()
    informative = ((X == 0) | np.isnan(X)).mean(0) < 0.3

    # 1차
    hc = [c for c, ok in zip(feats, informative) if ok and nun[c] > 500]
    p1 = lsh_pairs(df, hc, 40, 3, 0)
    sure = p1[match_rate(X, p1) > 0.9]
    a, b = X[sure[:, 0]], X[sure[:, 1]]
    col_diff = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), feats)
    stable = [c for c in feats if col_diff[c] < 0.02]
    stable_keys = [c for c, ok in zip(feats, informative) if ok and nun[c] > 100 and c in set(stable)]

    # 2차
    p2 = lsh_pairs(df, stable_keys, 80, 4, 1)
    p = np.unique(np.vstack([p1, p2]), axis=0)
    gap = np.abs(mi[p[:, 0]] - mi[p[:, 1]])
    p, gap = p[gap <= MAX_GAP], gap[gap <= MAX_GAP]
    Xs = df[stable].values.astype(np.float64)
    out = pd.DataFrame(dict(i=p[:, 0], j=p[:, 1], gap=gap, sim_stable=match_rate(Xs, p), sim_all=match_rate(X, p)))
    if verbose:
        print("1차 후보 %d, 확실한 쌍 %d, 안정 변수 %d개(키 후보 %d개), 2차 후보 %d" %
              (len(p1), len(sure), len(stable), len(stable_keys), len(p2)))
    return out, stable


if __name__ == "__main__":
    df = load().reset_index(drop=True)
    pairs, stable = find_pairs(df)
    y = df.TARGET.values
    pairs["agree"] = y[pairs.i] == y[pairs.j]
    pairs["bin"] = pd.cut(pairs.sim_stable, [0, .8, .9, .95, .98, .99, 1.0001])
    print("\n안정 변수 일치율 구간별: 쌍 수 / 타깃 일치율 (월 차이별)")
    print(pairs.pivot_table(index="bin", columns="gap", values="agree", aggfunc=["size", "mean"]).round(3).to_string())
    pairs.drop(columns="bin").to_pickle(os.path.join(ROOT, "cache", "dup_pairs_all.pkl"))
    pd.Series(stable).to_csv(os.path.join(ROOT, "results", "08_stable_cols.csv"), index=False)

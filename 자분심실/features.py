"""라벨 없이 계산하는 구조 변수 (학습+테스트 전체 행으로 계산).

- dup  : 동일인 쌍(Fellegi-Sunter 점수 > FS_THR) 개수, 최대 점수
- grp  : 동일인 그룹 크기·기간·순서, 신청 건마다 바뀌는 변수의 그룹 max/min/차이
- weak : 약한 유사 이웃(안정 변수 일치율 0.8~0.9) 개수, 동일인 제외 최대 유사도
"""
import os
import importlib
import numpy as np
import pandas as pd
from kcb_common import feature_cols, CAT_COLS, ROOT
from dup_utils import DUP_FEATS, add_dup_features, components, month_index

m08 = importlib.import_module("08_dup_matching")
FS_THR, N_CHANGING = 300, 12
WEAK_FEATS = ["WEAK_SAME", "WEAK_ADJ", "WEAK_FAR", "MAXSIM_STABLE"]


def build(df, pairs_fs, dynamics):
    """df: 0..n-1 인덱스. pairs_fs: 10_prob_matching 결과, dynamics: 12_column_dynamics.csv.
    반환: (df, pairs, comp, feature_groups)"""
    num_feats = [c for c in feature_cols(df) if c not in CAT_COLS]
    mi = month_index(df.LNMON.values)
    pairs = pairs_fs[pairs_fs.fs > FS_THR][["i", "j", "gap", "fs"]].reset_index(drop=True)

    df = add_dup_features(df, pairs, sim_col="fs")

    comp = components(len(df), pairs)
    g = pd.DataFrame(dict(comp=comp, mi=mi))
    new = {"GRP_SIZE": g.groupby("comp").comp.transform("size"),
           "GRP_NMONTHS": g.groupby("comp").mi.transform("nunique"),
           "GRP_ORDER": mi - g.groupby("comp").mi.transform("min")}
    X = df[num_feats].values
    a, b = X[pairs.i.values], X[pairs.j.values]
    differ = pd.Series((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean(0), num_feats)
    inf50 = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.5, num_feats)
    changing = differ[inf50].sort_values(ascending=False).index[:N_CHANGING].tolist()
    for c in changing:
        s = df[c].groupby(comp)
        new["GMAX_" + c] = s.transform("max")
        new["GMIN_" + c] = s.transform("min")
        new["GDIFF_" + c] = df[c] - s.transform("mean")
    grp_feats = list(new)

    inf30 = pd.Series(((X == 0) | np.isnan(X)).mean(0) < 0.3, num_feats)
    keys = [c for c in dynamics[(dynamics.eq0_g1 > 0.97) & (dynamics.nun > 200)].index if inf30.get(c, False)]
    stable = [c for c in dynamics[dynamics.eq0_g1 > 0.97].index if c in df.columns]
    m08.MAX_BUCKET = 6
    cand = m08.lsh_pairs(df, keys, 150, 2, 5)
    gap = np.abs(mi[cand[:, 0]] - mi[cand[:, 1]])
    sim = m08.match_rate(df[stable].values.astype(np.float64), cand)
    weak = (sim > 0.8) & (sim <= 0.9)
    w = {c: np.zeros(len(df), np.float32) for c in WEAK_FEATS}
    for sel, col in [(weak & (gap == 0), "WEAK_SAME"), (weak & (gap == 1), "WEAK_ADJ"), (weak & (gap >= 2), "WEAK_FAR")]:
        for side in (0, 1):
            np.add.at(w[col], cand[sel, side], 1)
    s_nodup = np.where(sim > 0.9, 0, sim).astype(np.float32)
    for side in (0, 1):
        np.maximum.at(w["MAXSIM_STABLE"], cand[:, side], s_nodup)
    new.update(w)

    df = pd.concat([df, pd.DataFrame({k: np.asarray(v, np.float32) for k, v in new.items()}, index=df.index)], axis=1)
    groups = dict(dup=DUP_FEATS, grp=grp_feats, weak=WEAK_FEATS)
    return df, pairs, comp, groups


def build_default(df):
    pairs_fs = pd.read_pickle(os.path.join(ROOT, "cache", "dup_pairs_fs.pkl"))
    dynamics = pd.read_csv(os.path.join(ROOT, "results", "12_column_dynamics.csv"), index_col=0)
    return build(df, pairs_fs, dynamics)

"""19_knn_dim_scan 결과 그래프: kNN 거리 공간 차원 수별 오분류율 · 로그손실 · AUC (작은 다중 차트, 축 하나씩).

폴드가 여러 개면 폴드별 선을 같은 차트에 그린다 (고정 순서 색: A 파랑, B 주황, C 청록).
출력: results/19_knn_dim_scan.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kcb_common import ROOT

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # 범주형 1~3번 슬롯 (all-pairs 검증 통과 범위)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})

d = pd.read_csv(os.path.join(ROOT, "results", "19_knn_dim_scan.csv")).sort_values(["fold", "dims"])
panels = [("err@0.450", "가중 오분류율 (임계값 0.45)", "min", "{:.4f}"),
          ("logloss", "로그손실", "min", "{:.4f}"),
          ("auc", "AUC", "max", "{:.4f}")]
folds = list(d.fold.unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
for ax, (col, title, best, fmt) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for f in folds:
        s = d[d.fold == f]
        ax.plot(s.dims, s[col], color=SERIES.get(f, TEXT2), lw=2, solid_joinstyle="round", solid_capstyle="round",
                marker="o", ms=5.5, mec=SURFACE, mew=1.5, label="폴드 %s" % f, zorder=3)
        if len(s) < 5:  # 확인용으로 몇 점만 찍은 폴드는 최적점 라벨 생략 (범례와 툴 없이도 선이 구분됨)
            continue
        i = s[col].idxmin() if best == "min" else s[col].idxmax()
        ax.annotate("%d차원\n%s" % (s.dims[i], fmt.format(s[col][i])), (s.dims[i], s[col][i]),
                    textcoords="offset points", xytext=(0, -30 if best == "min" else 12), ha="center",
                    fontsize=9, color=TEXT,
                    arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("kNN 거리 공간 차원 수", color=TEXT2)
    ax.grid(True, color=GRID, lw=1, ls="-")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=TEXT2, length=0)
    ax.margins(y=0.18)
    if len(folds) > 1:
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)

fig.suptitle("kNN 거리 공간 크기에 따른 성능 (LightGBM 시드 2개 평균, 후처리 포함)", x=0.01, ha="left",
             color=TEXT, fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.905, "오분류율·로그손실은 낮을수록, AUC는 높을수록 좋음. 시드 변동 폭은 오분류율 약 0.1%p.",
         color=TEXT2, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.88))
out = os.path.join(ROOT, "results", "19_knn_dim_scan.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("저장:", out)
print(d[["fold", "dims", "auc", "logloss", "err@0.450", "err_best", "corr_tr", "corr_te"]].round(4).to_string(index=False))

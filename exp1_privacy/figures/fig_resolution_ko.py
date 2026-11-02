"""해상도 비교 그림: 공격자 3단계(각 단계 대표 모델) × k, 해상도(60/30/15분)별 곡선.

사용법: python fig_resolution_ko.py DATA OUT [폰트.ttc]
필요 파일: attackers_results{,_30min,_15min}.csv, crosshh_results{,_30min,_15min}.csv
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

D, OUT = sys.argv[1], sys.argv[2]
F = sys.argv[3] if len(sys.argv) > 3 else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(F)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F).get_name(), "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# 해상도는 순서가 있는 크기 → 한 색상(파랑) 밝음→어두움 (높은 해상도일수록 진하게)
RES = [(60, "", "60분", "#86b6ef", "o"), (30, "_30min", "30분", "#2a78d6", "s"),
       (15, "_15min", "15분", "#104281", "D")]
TIERS = [("attackers", "adv_oracle_Pearson", "(a) 오라클 (Pearson)"),
         ("attackers", "adv_RF", "(b) 프로파일링 (RF)"),
         ("crosshh", "LR", "(c) 교차 세대 (LR)")]


def table(kind, col, suf):
    fn = os.path.join(D, f"{kind}_results{suf}.csv")
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn, dtype={"target": str, "combo": str})
    if kind == "crosshh":
        df = df.pivot_table(index=["target", "k", "combo"], columns="model", values="adv").reset_index()
    g = df.groupby("k")[col]
    return g.mean(), 1.96 * g.std().fillna(0) / np.sqrt(g.size())


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
offs = {60: -0.12, 30: 0.0, 15: 0.12}
for ax, (kind, col, title) in zip(axes, TIERS):
    ax.axhline(0, color="#b5b4ad", lw=0.8, zorder=1)
    for res, suf, lab, c, mk in RES:
        t = table(kind, col, suf)
        if t is None:
            continue
        m, ci = t
        ax.errorbar(m.index + offs[res], m.values, yerr=ci.values, color=c, lw=1.6, marker=mk, ms=4,
                    mec="white", mew=0.6, capsize=1.6, elinewidth=0.8, label=lab, zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_xticks(range(1, 6)); ax.set_xlim(0.6, 5.4); ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("집계 규모 k")
    ax.grid(axis="y", color="#e6e5df", lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)
axes[0].set_ylabel("우위 = 2(균형 정확도 − 0.5)")
axes[0].legend(title="해상도", loc="lower left", fontsize=6.5, title_fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.3, w_pad=0.8)
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(OUT, f"fig_resolution_ko.{ext}"), dpi=300)
print("saved")

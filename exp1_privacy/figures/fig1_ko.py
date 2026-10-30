"""Figure 1 (v2): three attacker tiers, strongest model of each tier (chosen by mean advantage)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
D, OUT = sys.argv[1], sys.argv[2]
F = sys.argv[3] if len(sys.argv) > 3 else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(F)
KFONT = fm.FontProperties(fname=F).get_name()
A = pd.read_csv(os.path.join(D, "attackers_results.csv"), dtype={"target": str, "combo": str})
C = pd.read_csv(os.path.join(D, "crosshh_results.csv"), dtype={"target": str, "combo": str})
CW = C.pivot_table(index=["target", "k", "combo"], columns="model", values="adv").reset_index()


def stats(df, col):
    g = df.groupby("k")[col]
    return g.mean(), 1.96 * g.std().fillna(0) / np.sqrt(g.size())


tiers = [  # data, column, label, color, ls, marker
    (A, "adv_oracle_Pearson", "오라클: 대상 프로파일을 앎 (Pearson)", "#3d3d3a", (0, (4, 2)), "^"),
    (A, "adv_RF", "프로파일링: 대상 세대 라벨로 학습 (RF)", "#2a78d6", "-", "o"),
    (CW, "LR", "교차 세대: 다른 세대로 학습 (LR)", "#eb6834", "-", "D"),
]
ks = np.arange(1, 6)
offs = [-0.1, 0.0, 0.1]
plt.rcParams.update({"font.family": KFONT, "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.45))
ax.axhline(0, color="#b5b4ad", lw=0.8, zorder=1)
for (df, col, lab, c, ls, mk), off in zip(tiers, offs):
    m, ci = stats(df, col)
    m, ci = m.reindex(ks).to_numpy(), ci.reindex(ks).to_numpy()
    ax.errorbar(ks + off, m, yerr=ci, color=c, ls=ls, lw=1.6, marker=mk, ms=4.5, mec="white",
                mew=0.6, capsize=1.8, elinewidth=0.8, label=lab, zorder=3)
ax.set_xticks(ks); ax.set_xlim(0.6, 5.4); ax.set_ylim(-0.04, 1.05)
ax.set_xlabel("집계 규모 k (DCU에서 합산되는 세대 수)")
ax.set_ylabel("우위 = 2(균형 정확도 − 0.5)")
ax.grid(axis="y", color="#e6e5df", lw=0.5, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=2.5)
ax.legend(loc="upper right", fontsize=6.2, frameon=False, handlelength=2.4, labelspacing=0.35,
          borderaxespad=0.1)
fig.tight_layout(pad=0.3)
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"fig1_ko.{ext}"), dpi=300)
print("saved")

"""Figure 3: household 02 present vs absent day, alone (k=1) and aggregated with 04+06 (k=3)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

D, OUT, F = sys.argv[1], sys.argv[2], sys.argv[3]
fm.fontManager.addfont(F)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F).get_name(), "font.size": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
sm = pd.read_csv(os.path.join(D, "eco_sm_hourly.csv"), dtype={"household": str})
P = sm.pivot_table(index=["date", "hour"], columns="household", values="power_mean_w")
days = {"재실일 (2012-08-04)": "2012-08-04", "부재일 (2012-07-11)": "2012-07-11"}
style = {"재실일 (2012-08-04)": dict(color="#222220", ls="-", marker="o"),
         "부재일 (2012-07-11)": dict(color="#8a8a85", ls=(0, (3, 1.5)), marker="s")}
h = np.arange(24)
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True)
for ax, (title, members) in zip(axes, [("(a) k=1: 02번 세대 단독", ["02"]),
                                        ("(b) k=3: 02+04+06번 합계 ÷ 3", ["02", "04", "06"])]):
    ax.axvspan(8, 18, color="#eef4fb", zorder=0, lw=0)
    for lab, d in days.items():
        y = P.loc[d, members].sum(axis=1).to_numpy() / len(members)
        ax.plot(h, y, lw=1.3, ms=2.4, mec="white", mew=0.3, label=lab, zorder=3, **style[lab])
    ax.set_title(title, fontsize=7.2)
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("시각 (시)")
    ax.grid(axis="y", color="#e6e5df", lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)
axes[0].set_ylabel("평균 전력 (W)")
axes[1].text(13, axes[1].get_ylim()[1] * 0.70, "주간\n08–18시", ha="center", va="top", fontsize=6, color="#4f4e49")
axes[0].legend(loc="upper left", fontsize=6.0, frameon=False, handlelength=2.2, borderaxespad=0.1)
fig.tight_layout(pad=0.3, w_pad=0.6)
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(OUT, f"fig3_profiles_ko.{ext}"), dpi=300)
print("saved")

"""Figure 1: overview (data -> DCU aggregation -> attacker evaluation), full text width."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, Circle, FancyArrow, Arc

OUT, F = sys.argv[1], sys.argv[2]
fm.fontManager.addfont(F)
KF = fm.FontProperties(fname=F).get_name()
plt.rcParams.update({"font.family": KF, "mathtext.fontset": "dejavusans"})

INK, SUB, LINE = "#222220", "#4f4e49", "#6f6e69"
HL = "#cfe8f7"          # step-label highlight
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#3d3d3a"
W, H = 6.85, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
XM, YM = 137.0, 55.0              # 1 unit = 0.05 in in both axes
ax.set_xlim(0, XM); ax.set_ylim(0, YM); ax.axis("off")


def T(x, y, s, size=6.4, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    return ax.text(x, y, s, fontsize=size, **kw)


def step(x, y, n, s, w=None):
    """numbered step label on a highlight band"""
    t = T(x, y, f"{n}  {s}", 6.6, weight="bold",
          bbox=dict(boxstyle="square,pad=0.35", fc=HL, ec="none"))
    return t


def arrow(x0, y0, x1, y1, lw=1.1, color=LINE):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=8, shrinkA=0, shrinkB=0))


# ---------------- zone A: data ----------------
# cylinder
cx, cy, cw, ch = 13, 22, 20, 15
ax.add_patch(Rectangle((cx - cw / 2, cy - ch / 2), cw, ch, fc="white", ec="none", zorder=2))
ax.add_patch(Ellipse((cx, cy - ch / 2), cw, 4.4, fc="white", ec="none", zorder=1))
ax.plot([cx - cw / 2, cx - cw / 2], [cy - ch / 2, cy + ch / 2], color=INK, lw=0.9, zorder=3)
ax.plot([cx + cw / 2, cx + cw / 2], [cy - ch / 2, cy + ch / 2], color=INK, lw=0.9, zorder=3)
ax.add_patch(Arc((cx, cy - ch / 2), cw, 4.4, theta1=180, theta2=360, ec=INK, lw=0.9, zorder=3))
ax.add_patch(Ellipse((cx, cy + ch / 2), cw, 4.4, fc="#f2f2ee", ec=INK, lw=0.9, zorder=3))
T(cx, cy + 1.6, "ECO 데이터셋", 6.6, weight="bold", zorder=4)
T(cx, cy - 2.6, "6개 세대(01–06번)\n전력·재실 라벨(1초)", 5.6, color=SUB, zorder=4, linespacing=1.3)

# stacked hourly tables
rows = [("시각", r"$e$ (W)", "재실률"), ("10", "337", "1.00"), ("14", "685", "0.75"), (r"$\vdots$", r"$\vdots$", r"$\vdots$"), ("21", "329", "1.00")]
tx, ty, cwid, rh = 30, 13.5, [6.2, 7.4, 6.0], 3.6
for s in range(3, 0, -1):  # back sheets
    ax.add_patch(Rectangle((tx + s * 1.4, ty + s * 1.4), sum(cwid), rh * len(rows), fc="white", ec=LINE, lw=0.6, zorder=1))
x = tx
for j, wcol in enumerate(cwid):
    for i, r in enumerate(rows):
        yy = ty + rh * (len(rows) - 1 - i)
        fc = "#eef6e9" if j == 2 else ("#f2f2ee" if i == 0 else "white")
        ax.add_patch(Rectangle((x, yy), wcol, rh, fc=fc, ec=LINE, lw=0.5, zorder=2))
        T(x + wcol / 2, yy + rh / 2, r[j], 5.4, weight="bold" if i == 0 else "normal", zorder=3)
    x += wcol
T(tx + sum(cwid) / 2, ty - 3.0, "전처리 결과: 세대별 시간당 평균 전력과 재실률", 5.6, color=SUB)
arrow(cx + cw / 2 + 0.5, cy, tx - 0.8, cy)
step(27.0, 51.0, "①", "데이터 전처리 (1초 → 1시간 평균)")

# ---------------- zone B: DCU aggregation ----------------
def house(x, y, c, lab):
    ax.add_patch(Polygon([[x - 3.2, y], [x, y + 2.8], [x + 3.2, y]], closed=True, fc=c, ec=INK, lw=0.7, zorder=3))
    ax.add_patch(Rectangle((x - 2.5, y - 3.6), 5.0, 3.6, fc="white", ec=INK, lw=0.7, zorder=3))
    T(x, y - 1.8, lab, 5.4, zorder=4)

bx = 64
hs = [(bx, 38, "#fbe3d8", r"$e_1$"), (bx, 27.5, "#e6e5df", r"$e_2$"), (bx, 14, "#e6e5df", r"$e_k$")]
for x0, y0, c, lab in hs:
    house(x0, y0, c, lab)
T(bx, 20.4, r"$\vdots$", 7)
T(bx, 44.2, "대상 세대", 5.4, color=ORANGE, weight="bold")
T(bx, 7.2, r"함께 집계되는 $k-1$세대", 5.4, color=SUB)
arrow(tx + sum(cwid) + 4.6, cy, bx - 5.0, cy)

# DCU sum
sx, sy = 79.5, 26
for x0, y0, *_ in hs:
    ax.plot([x0 + 3.4, sx - 3.4], [y0 - 1.0, sy + (y0 - sy) * 0.15], color=LINE, lw=0.8, zorder=1)
ax.add_patch(Circle((sx, sy), 3.4, fc="white", ec=INK, lw=0.9, zorder=3))
T(sx, sy, "Σ", 9, zorder=4)
T(sx, sy + 6.2, "DCU", 6.4, weight="bold")
step(72.0, 51.0, "②", r"집계 생성 ($k$=1–5, 모든 조합)")
T(sx + 4.2, sy - 7.6, r"$a_t=(1/k)\,\Sigma_{i\in S}\,e_{i,t}$", 6.2)
T(sx + 2.4, sy - 13.0, r"$k$=1: 개별값 관측 (현재 AMI)" + "\n" + r"$k$≥2: 합계만 관측 (보안 집계)", 5.4, color=SUB, linespacing=1.35)
arrow(sx + 3.8, sy, 92.5, sy)
T(88.2, sy + 2.4, r"$\{a_t\}$", 6.2)

# ---------------- zone C: attacker evaluation (dashed panel) ----------------
px, py, pw, ph = 93.5, 1.5, 42, 47.5
ax.add_patch(Rectangle((px, py), pw, ph, fc="white", ec=LINE, lw=0.8, ls=(0, (1.5, 1.5)), zorder=0))
T(px + pw / 2, py + ph + 2.6, "공격자 수준별 평가", 7.0, weight="bold")
step(px + pw / 2, 45.0, "③", "공격자 3단계 (사전지식 많음 → 적음)")

# attacker table
cols = [6.0, 13.5, 18.5]
hdr = ("ID", "공격자", "사전지식")
data = [(r"$A_1$", "오라클", r"대상의 $e_0$, $e_1$ 프로파일", GRAY),
        (r"$A_2$", "프로파일링", "대상 세대의 과거 라벨", BLUE),
        (r"$A_3$", "교차 세대", "다른 세대의 라벨만", ORANGE)]
x0t, rh2 = px + 2.0, 4.3
ytop = 40.0
x = x0t
for j, wcol in enumerate(cols):
    ax.add_patch(Rectangle((x, ytop - rh2), wcol, rh2, fc="#dce9f5", ec=LINE, lw=0.5))
    T(x + wcol / 2, ytop - rh2 / 2, hdr[j], 5.8, weight="bold")
    for i, (a, b, c, col) in enumerate(data):
        yy = ytop - rh2 * (i + 2)
        ax.add_patch(Rectangle((x, yy), wcol, rh2, fc="white", ec=LINE, lw=0.5))
        val = (a, b, c)[j]
        T(x + wcol / 2, yy + rh2 / 2, val, 5.6, color=col if j == 1 else INK, weight="bold" if j == 1 else "normal")
    x += wcol
# big down arrow (gradient-like)
ax.add_patch(FancyArrow(px + pw / 2, 21.4, 0, -3.6, width=14, head_width=22, head_length=3.2,
                        length_includes_head=True, fc="#b9d6f2", ec="none"))
step(px + pw / 2, 12.6, "④", "주간 재실/부재 판별 성능 측정")
T(px + pw / 2, 7.2, "주간 재실일(≥0.8) vs 부재일(≤0.5)", 5.5, color=SUB)
T(px + pw / 2, 3.9, "우위 Adv = 2(BA − 0.5)", 5.5, color=SUB)

for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(OUT, f"fig_overview_ko.{ext}"), dpi=300)
print("saved")

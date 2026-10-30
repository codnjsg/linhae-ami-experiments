"""Cross-household (transfer) learned attacker for the day-level task.

The attacker never sees the target's labels. For target t and size k:
  Train: every other labelled household h != t (01-05), aggregated with every combination of
         k-1 background households drawn from HH_ALL minus {h, t}  (target's data never used).
  Test : target t aggregated with every combination of k-1 others (as in sameplay.py).
Scale differences between streams are removed without labels: each day's 24-h vector is divided
by the mean of its own aggregate stream over all observed days (computable by the DCU).
Labels: daytime (08-18) occupancy >= 0.8 -> present (0), <= 0.5 -> absent (1).
Advantage = 2 * (balanced accuracy - 0.5) on the target, per test combo.
"""
import sys, os, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

D = sys.argv[1]
HH_ALL = ["01", "02", "03", "04", "05", "06"]
LABELLED = ["01", "02", "03", "04", "05"]
TARGETS = ["01", "02", "03"]

sm = pd.read_csv(os.path.join(D, "eco_sm_hourly.csv"), dtype={"household": str})
sm["valid"] = (sm.coverage >= 0.9) & sm.power_mean_w.notna()
dayok = sm.groupby(["household", "date"]).valid.sum().ge(22)
okdays = {hh: set(d for (h, d), v in dayok.items() if v and h == hh) for hh in HH_ALL}
P = sm.pivot_table(index=["date", "hour"], columns="household", values="power_mean_w")
P = P.groupby(level=0).transform(lambda s: s.interpolate(limit_direction="both"))
prof = {hh: {d: P.loc[d, hh].to_numpy() for d in okdays[hh]} for hh in HH_ALL}
occ = pd.read_csv(os.path.join(D, "occ_hourly.csv"), dtype={"household": str})
dayocc = occ[(occ.hour >= 8) & (occ.hour < 18)].groupby(["household", "date"]).occ_frac.mean()


def feats(z):
    return np.concatenate([z, [z.mean(), z.std(), z.min(), z.max(), z[8:18].mean() - z[:6].mean(),
                               z[8:18].std()]])


def stream(t, combo):
    """Labelled day-level samples for target t aggregated with combo. Normalization uses all
    valid days of the stream (labelled or not), as the DCU would."""
    members = (t,) + combo
    alld = sorted(set.intersection(*(okdays[m] for m in members)))
    if not alld:
        return None, None
    agg = {d: sum(prof[m][d] for m in members) / len(members) for d in alld}
    scale = np.mean([agg[d].mean() for d in alld])
    dt = dayocc.loc[t]
    X, y = [], []
    for d in alld:
        if d in dt.index and (dt[d] >= 0.8 or dt[d] <= 0.5):
            X.append(feats(agg[d] / scale))
            y.append(0 if dt[d] >= 0.8 else 1)
    return (np.array(X), np.array(y)) if X else (None, None)


def make(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "GB":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, min_samples_leaf=10,
                                              class_weight="balanced", random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))


rows = []
for t in TARGETS:
    for k in range(1, 6):
        Xtr, ytr = [], []
        for h in [x for x in LABELLED if x != t]:
            pool = [x for x in HH_ALL if x not in (h, t)]
            for combo in itertools.combinations(pool, k - 1):
                X, y = stream(h, combo)
                if X is not None:
                    Xtr.append(X); ytr.append(y)
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        tests = []
        for combo in itertools.combinations([x for x in HH_ALL if x != t], k - 1):
            X, y = stream(t, combo)
            if X is not None and (y == 0).sum() >= 8 and (y == 1).sum() >= 8:
                tests.append(("+".join(combo) or "-", X, y))
        for name in ["RF", "GB", "LR"]:
            seeds = [0, 1, 2] if name != "LR" else [0]
            preds = {c: [] for c, _, _ in tests}
            for s in seeds:
                clf = make(name, s).fit(Xtr, ytr)
                for c, X, y in tests:
                    preds[c].append(balanced_accuracy_score(y, clf.predict(X)))
            for c, X, y in tests:
                rows.append(dict(target=t, k=k, combo=c, model=name, n_train=len(ytr),
                                 train_abs_frac=ytr.mean(), n_test=len(y),
                                 adv=2 * (np.mean(preds[c]) - 0.5)))
        print("done", t, k, len(ytr), flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(D, "crosshh_results.csv"), index=False)
W = R.pivot_table(index=["target", "k", "combo"], columns="model", values="adv").reset_index()
g = W.groupby("k")[["RF", "GB", "LR"]]
m, se = g.mean(), g.std() / np.sqrt(g.size().to_numpy()[:, None])
print((m.round(3).astype(str) + " ± " + (1.96 * se).round(3).astype(str)).to_string())
print("\nPer target")
print(W.groupby(["target", "k"])[["RF", "GB", "LR"]].mean().round(3).to_string())

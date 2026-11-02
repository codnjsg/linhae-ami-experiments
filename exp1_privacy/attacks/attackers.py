"""Robustness of the learned-attacker result: same day-level task, same days, same folds
as sameplay.py, with three learned attackers:
  RF  : RandomForest (as in sameplay.py)
  GB  : HistGradientBoosting
  LR  : standardized logistic regression (L2)
Advantage = 2 * (balanced accuracy - 0.5), mean over 3 repeats of stratified 5-fold CV.
"""
import sys, os, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import args, load, suffix, H, HH_ALL, TARGETS

D, RES = args()
okdays, prof, dayocc = load(D, RES)
D8, D18, N6 = H(8, RES), H(18, RES), H(6, RES)


def feats(a):
    return np.concatenate([a, [a.mean(), a.std(), a.min(), a.max(), a[D8:D18].mean() - a[:N6].mean()]])


def make(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "GB":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, min_samples_leaf=5,
                                              class_weight="balanced", random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


rows = []
for t in TARGETS:
    dt = dayocc.loc[t]
    others = [h for h in HH_ALL if h != t]
    for k in range(1, 6):
        for combo in itertools.combinations(others, k - 1):
            members = (t,) + combo
            days = sorted(set.intersection(*(okdays[m] for m in members)) & set(dt.index))
            pres = [d for d in days if dt[d] >= 0.8]
            absn = [d for d in days if dt[d] <= 0.5]
            if len(pres) < 8 or len(absn) < 8:
                continue
            dd = pres + absn
            y = np.array([0] * len(pres) + [1] * len(absn))
            X = np.array([feats(sum(prof[m][d] for m in members) / k) for d in dd])
            row = dict(target=t, k=k, combo="+".join(combo) or "-", n_days=len(dd))
            for name in ["RF", "GB", "LR"]:
                accs = []
                for rep in range(3):
                    pred = np.zeros_like(y)
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                        clf = make(name, rep)
                        clf.fit(X[tr], y[tr])
                        pred[te] = clf.predict(X[te])
                    accs.append(balanced_accuracy_score(y, pred))
                row[f"adv_{name}"] = 2 * (np.mean(accs) - 0.5)
            rows.append(row)
    print("done", t, flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(D, f"attackers_raw{suffix(RES)}.csv"), index=False)
O = pd.read_csv(os.path.join(D, f"sameplay_results{suffix(RES)}.csv"), dtype={"combo": str, "target": str})
R = R.merge(O[["target", "k", "combo", "adv_oracle_Combined", "adv_oracle_Pearson"]],
            on=["target", "k", "combo"], how="left")
R["adv_best_learned"] = R[["adv_RF", "adv_GB", "adv_LR"]].max(axis=1)
R.to_csv(os.path.join(D, f"attackers_results{suffix(RES)}.csv"), index=False)
cols = ["adv_RF", "adv_GB", "adv_LR", "adv_best_learned", "adv_oracle_Combined", "adv_oracle_Pearson"]
g = R.groupby("k")[cols]
m, se = g.mean(), g.std() / np.sqrt(g.size().to_numpy()[:, None])
print((m.round(3).astype(str) + " ± " + (1.96 * se).round(3).astype(str)).to_string())
print("\nPer target")
print(R.groupby(["target", "k"])[cols].mean().round(3).to_string())

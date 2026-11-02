"""Same task, same aggregates, same metric: oracle vs learned attacker.

Task: was target household t PRESENT (daytime 08-18 occ >= 0.8) or ABSENT (<= 0.5) on day d?
Aggregate on day d = target's profile on d + background members' profiles on the SAME day d
(realistic DCU view), divided by k.
  Oracle  : knows e0 (target profile on a present day) and e1 (on an absent day), one of which
            is d itself; decides with Buescher's Combined / Pearson functions.
  Learned : RandomForest trained on other days' aggregates (24 hourly values + daily stats)
            with present/absent labels; never sees the target's own profile on test days.
Advantage = 2 * (balanced accuracy - 0.5) for both.
"""
import sys, os, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import args, load, suffix, H, HH_ALL, TARGETS

D, RES = args()
W = H(2, RES)  # Combined 함수 창 크기: 2시간 (60분: 2구간, 30분: 4구간, 15분: 8구간)
okdays, prof, dayocc = load(D, RES)
D8, D18, N6 = H(8, RES), H(18, RES), H(6, RES)


def peaks(x):
    return {i for i in range(len(x)) if (i == 0 or x[i] > x[i - 1]) and (i == len(x) - 1 or x[i] > x[i + 1])}


def corr(a, b):
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def comb_score(e, ea):
    idx = sorted(peaks(e) | peaks(ea))
    cs = [corr(e[max(0, i - W):i + W + 1], ea[max(0, i - W):i + W + 1]) for i in idx
          if min(len(e), i + W + 1) - max(0, i - W) >= 3]
    return np.mean(cs) if cs else 0.0


def feats(a):
    return np.concatenate([a, [a.mean(), a.std(), a.min(), a.max(), a[D8:D18].mean() - a[:N6].mean()]])


rng = np.random.default_rng(0)
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
            y = np.array([0] * len(pres) + [1] * len(absn))  # 0 = present, 1 = absent
            agg = {d: sum(prof[m][d] for m in members) / k for d in dd}
            # Oracle: for each day d, pair it with a random day of the other class
            ora = {"Combined": [], "Pearson": []}
            for d, lab in zip(dd, y):
                other = absn[rng.integers(len(absn))] if lab == 0 else pres[rng.integers(len(pres))]
                e0, e1 = (prof[t][d], prof[t][other]) if lab == 0 else (prof[t][other], prof[t][d])
                ea = agg[d]
                ora["Combined"].append(0 if comb_score(e0, ea) > comb_score(e1, ea) else 1)
                ora["Pearson"].append(0 if corr(e0, ea) > corr(e1, ea) else 1)
            # Learned attacker: repeated stratified 5-fold CV over days
            X = np.array([feats(agg[d]) for d in dd])
            accs = []
            for rep in range(3):
                pred = np.zeros_like(y)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                 class_weight="balanced", random_state=rep, n_jobs=-1)
                    clf.fit(X[tr], y[tr])
                    pred[te] = clf.predict(X[te])
                accs.append(balanced_accuracy_score(y, pred))
            row = dict(target=t, k=k, combo="+".join(combo) or "-", n_pres=len(pres), n_abs=len(absn),
                       adv_learned=2 * (np.mean(accs) - 0.5))
            for f, p in ora.items():
                row[f"adv_oracle_{f}"] = 2 * (balanced_accuracy_score(y, p) - 0.5)
            rows.append(row)
    print("done", t, flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(D, f"sameplay_results{suffix(RES)}.csv"), index=False)
cols = ["adv_learned", "adv_oracle_Combined", "adv_oracle_Pearson"]
print(R.groupby(["target", "k"])[cols].mean().round(3))
S = R.groupby("k")[cols].agg(["mean", "std", "size"]).round(3)
print("\nPooled over targets and combos\n", S)

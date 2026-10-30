"""Build hourly occupancy labels and an hourly master table; report overlap."""
import sys, os
import numpy as np, pandas as pd

D = sys.argv[1]  # data dir
rows = []
for hh in ["01", "02", "03", "04", "05"]:
    for season in ["summer", "winter"]:
        f = os.path.join(D, "occ", f"{hh}_{season}.csv")
        df = pd.read_csv(f, header=0)
        dates = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip("'"), format="%d-%b-%Y")
        vals = df.iloc[:, 1:86401].to_numpy(dtype=np.int8)
        hourly = vals.reshape(len(df), 24, 3600).mean(axis=2)  # fraction occupied per hour
        for i, d in enumerate(dates):
            for h in range(24):
                rows.append((hh, d.strftime("%Y-%m-%d"), h, hourly[i, h], season))
occ = pd.DataFrame(rows, columns=["household", "date", "hour", "occ_frac", "season"])
occ["occ"] = (occ.occ_frac >= 0.5).astype(int)  # majority vote per hour
occ.to_csv(os.path.join(D, "occ_hourly.csv"), index=False)

sm = pd.read_csv(os.path.join(D, "eco_sm_hourly.csv"), dtype={"household": str})
sm["valid"] = (sm.coverage >= 0.9) & sm.power_mean_w.notna()
print("sm rows", len(sm), "valid frac", sm.valid.mean().round(3))
print(sm.groupby("household").date.agg(["min", "max", "nunique"]))

# Days with valid full meter data (>=22 of 24 hours valid)
dayok = sm.groupby(["household", "date"]).valid.sum().ge(22).rename("dayok").reset_index()
okset = {hh: set(g[g.dayok].date) for hh, g in dayok.groupby("household")}

print("\nLabel days per household and overlap with other households' valid meter days")
for hh, g in occ.groupby("household"):
    ldays = sorted(set(g.date) & okset.get(hh, set()))
    cnt = [sum(d in okset[o] for o in okset if o != hh) for d in ldays]
    c = pd.Series(cnt)
    print(f"{hh}: label days {g.date.nunique()}, with own meter ok {len(ldays)}, "
          f"occ rate {g.occ.mean():.2f}; days with >=k-1 others: "
          + ", ".join(f"k={k}:{(c >= k - 1).sum()}" for k in range(2, 7)))

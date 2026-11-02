"""실험 1 공격 스크립트 공통 부분: 인자 처리, 해상도별 데이터 로딩, 시각 → 구간 인덱스 변환.

해상도(--res)는 60 / 30 / 15분. 60분일 때는 단편논문 코드와 같은 파일·같은 결과를 냄.
  - 계량 데이터: eco_sm_hourly.csv (60) / eco_sm_30min.csv / eco_sm_15min.csv
  - 유효일 기준: 하루 중 유효 구간이 22시간분 이상 (60분: 22개, 30분: 44개, 15분: 88개)
  - 재실 라벨(일 단위 재실일/부재일)은 해상도와 무관하게 occ_hourly.csv의 08–18시 재실률로 정함
  - 결과 파일 이름: 60분은 기존과 같고, 그 외는 _30min / _15min 접미사
"""
import argparse
import os

import pandas as pd

HH_ALL = ["01", "02", "03", "04", "05", "06"]
LABELLED = ["01", "02", "03", "04", "05"]
TARGETS = ["01", "02", "03"]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--res", type=int, default=60, choices=[15, 30, 60], help="해상도(분), 기본 60")
    a = ap.parse_args()
    return a.data_dir, a.res


def suffix(res):
    return "" if res == 60 else f"_{res}min"


def slots_per_day(res):
    return 1440 // res


def H(h, res):
    """시각(시) → 그 시각이 시작되는 구간 인덱스"""
    return h * slots_per_day(res) // 24


def load(D, res):
    """returns okdays {hh: set(date)}, prof {hh: {date: array(S)}}, dayocc Series[(hh, date)]"""
    S = slots_per_day(res)
    fn, col = ("eco_sm_hourly.csv", "hour") if res == 60 else (f"eco_sm_{res}min.csv", "slot")
    sm = pd.read_csv(os.path.join(D, fn), dtype={"household": str})
    sm["valid"] = (sm.coverage >= 0.9) & sm.power_mean_w.notna()
    dayok = sm.groupby(["household", "date"]).valid.sum().ge(22 * S // 24)
    okdays = {hh: set(d for (h, d), v in dayok.items() if v and h == hh) for hh in HH_ALL}
    P = sm.pivot_table(index=["date", col], columns="household", values="power_mean_w")
    P = P.groupby(level=0).transform(lambda s: s.interpolate(limit_direction="both"))
    prof = {hh: {d: P.loc[d, hh].to_numpy() for d in okdays[hh]} for hh in HH_ALL}
    occ = pd.read_csv(os.path.join(D, "occ_hourly.csv"), dtype={"household": str})
    dayocc = occ[(occ.hour >= 8) & (occ.hour < 18)].groupby(["household", "date"]).occ_frac.mean()
    return okdays, prof, dayocc

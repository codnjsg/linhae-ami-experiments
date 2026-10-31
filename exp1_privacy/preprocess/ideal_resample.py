#!/usr/bin/env python3
"""IDEAL mains 전력(1Hz)을 ECO와 같은 형식의 15분 / 30분 / 60분 평균으로 줄이는 스크립트.

IDEAL 센서 파일 형식 (Pullinger et al., Scientific Data 2021):
  - gzip 압축 CSV, 헤더 없음, 2열: timestamp(UTC, 'YYYY-MM-DD hh:mm:ss'), value(W)
  - 파일 이름에 가구(home), 방, 센서 ID와 센서 종류가 들어 있음

사용법 (압축 푼 폴더, 또는 IDEAL 배포 zip 그대로 모두 가능):
    python ideal_resample.py <폴더 또는 zip> [<폴더 또는 zip> ...]
옵션:
    --match electric-mains,electric-combined   파일 이름에 모두 포함돼야 하는 문자열 (쉼표 구분, 기본값)
    --homes 61,62,63                            처리할 가구 번호만 지정 (기본: 전부)
    --list                                      처리하지 않고 찾은 파일 목록만 출력 (먼저 확인용)
    --workers 4                                 동시에 처리할 파일 수 (기본 1)

결과 (현재 폴더): ideal_sm_15min.csv, ideal_sm_30min.csv (slot), ideal_sm_hourly.csv (hour)
  컬럼: household, date, slot|hour, power_mean_w, coverage   ← ECO 출력과 같음
  - household    : 'h' + IDEAL home 번호 (예: h61)
  - date, slot   : 영국 현지 시각(GMT/BST) 기준. UTC를 영국 서머타임 규칙으로 직접 변환 (tzdata 불필요)
  - coverage     : 구간 초 중 측정값이 있는 비율 (0~1)
표준 라이브러리만 사용합니다.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

HOME_RE = re.compile(r"home(\d+)", re.IGNORECASE)
BASE = 15 * 60
NB = 86400 // BASE
SPECS = [(15, "ideal_sm_15min.csv", "slot"), (30, "ideal_sm_30min.csv", "slot"),
         (60, "ideal_sm_hourly.csv", "hour")]


def last_sunday(year, month):
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


_BST = {}


def uk_offset_hours(t):
    """영국 서머타임: 3월 마지막 일요일 01:00 UTC ~ 10월 마지막 일요일 01:00 UTC 동안 +1시간"""
    y = t.year
    if y not in _BST:
        s, e = last_sunday(y, 3), last_sunday(y, 10)
        _BST[y] = (dt.datetime(s.year, s.month, s.day, 1), dt.datetime(e.year, e.month, e.day, 1))
    s, e = _BST[y]
    return 1 if s <= t < e else 0


def find_files(srcs, needles):
    """(원본 위치, zip 내부 이름 또는 None, 표시용 이름) 목록"""
    out = []
    for src in srcs:
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as z:
                for name in z.namelist():
                    base = os.path.basename(name)
                    if base and all(n in base for n in needles):
                        out.append((src, name, base))
        else:
            for root, _, files in os.walk(src):
                for fn in files:
                    if all(n in fn for n in needles):
                        out.append((os.path.join(root, fn), None, fn))
    return sorted(out, key=lambda x: x[2])


def open_lines(path, member):
    if member is None:
        raw = open(path, "rb")
    else:
        raw = zipfile.ZipFile(path).open(member)
    head = raw.read(2)
    rest = raw  # 이어서 읽기
    stream = io.BufferedReader(_Prefixed(head, rest))
    if head == b"\x1f\x8b":
        stream = gzip.GzipFile(fileobj=stream)
    return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")


class _Prefixed(io.RawIOBase):
    """앞에서 읽은 2바이트(gzip 판별용)를 되돌려 붙여 주는 래퍼"""
    def __init__(self, prefix, f):
        self.p, self.f = prefix, f

    def readable(self):
        return True

    def readinto(self, b):
        if self.p:
            n = min(len(b), len(self.p))
            b[:n], self.p = self.p[:n], self.p[n:]
            return n
        data = self.f.read(len(b))
        b[:len(data)] = data
        return len(data)


def process(item):
    """파일 하나 → {date: (sums[96], counts[96])}. 같은 초의 중복 측정은 하나만 셈."""
    path, member, base = item
    m = HOME_RE.search(base)
    hh = "h" + (m.group(1) if m else base)
    days = {}
    last = None
    hour_cache = {}  # 'YYYY-MM-DD hh' (UTC) -> (현지 날짜 문자열, 현지 시각의 하루 중 초 오프셋)
    with open_lines(path, member) as f:
        for line in f:
            ts, _, val = line.partition(",")
            ts = ts.strip()
            if len(ts) < 19 or ts == last:
                continue
            try:
                v = float(val)
            except ValueError:
                continue
            if v != v or v < 0:
                continue
            last = ts
            hk = ts[:13]
            hc = hour_cache.get(hk)
            if hc is None:
                try:
                    t = dt.datetime(int(ts[0:4]), int(ts[5:7]), int(ts[8:10]), int(ts[11:13]))
                except ValueError:
                    continue
                lt = t + dt.timedelta(hours=uk_offset_hours(t))
                hc = hour_cache[hk] = (lt.strftime("%Y-%m-%d"), lt.hour * 3600)
            key, hsec = hc
            try:
                b = (hsec + int(ts[14:16]) * 60 + int(ts[17:19])) // BASE
            except ValueError:
                continue
            d = days.get(key)
            if d is None:
                d = days[key] = ([0.0] * NB, [0] * NB)
            d[0][b] += v
            d[1][b] += 1
    return hh, base, days


def merge(sums, counts, m):
    f = m // 15
    out = []
    for s in range(NB // f):
        S = sum(sums[s * f:(s + 1) * f])
        C = sum(counts[s * f:(s + 1) * f])
        out.append((s, (S / C) if C else "", min(1.0, C / (m * 60))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--match", default="electric-mains,electric-combined")
    ap.add_argument("--homes", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    needles = [s for s in a.match.split(",") if s]
    items = find_files(a.srcs, needles)
    if a.homes:
        want = {"home" + h.strip() for h in a.homes.split(",")}
        items = [x for x in items if (HOME_RE.search(x[2]) and HOME_RE.search(x[2]).group(0).lower() in want)]
    print(f"찾은 파일 {len(items)}개 (조건: {' & '.join(needles)})")
    if a.list or not items:
        for _, _, b in items[:50]:
            print("  ", b)
        if len(items) > 50:
            print(f"   ... 외 {len(items) - 50}개")
        if not items:
            print("파일 이름 조건(--match)을 확인하세요. 예: --match electric-mains")
        return
    files = {m: open(fn, "w", newline="") for m, fn, _ in SPECS}
    w = {m: csv.writer(files[m]) for m in files}
    for m, _, col in SPECS:
        w[m].writerow(["household", "date", col, "power_mean_w", "coverage"])
    seen = set()
    try:
        pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
        results = pool.map(process, items) if pool else map(process, items)
        for i, (hh, base, days) in enumerate(results, 1):
            if hh in seen:
                print(f"  경고: {hh} 파일이 여러 개 → {base} 건너뜀 (--match로 하나만 고르세요)")
                continue
            seen.add(hh)
            for d in sorted(days):
                s, c = days[d]
                for m in w:
                    for slot, p, cov in merge(s, c, m):
                        w[m].writerow([hh, d, slot, f"{p:.2f}" if p != "" else "", f"{cov:.3f}"])
            print(f"[{i}/{len(items)}] {hh}: {len(days)}일 처리 ({base})", flush=True)
        if pool:
            pool.shutdown()
    finally:
        for f in files.values():
            f.close()
    print("완료 ->", ", ".join(fn for _, fn, _ in SPECS))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ECO 스마트미터(sm_csv) 1초 데이터를 15분 / 30분 / 60분 평균으로 한 번에 줄이는 스크립트.
(eco_sm_to_hourly.py 확장판: 원본 데이터를 한 번만 읽고 세 해상도를 모두 만듦)

사용법 (zip 그대로 또는 압축 푼 폴더 모두 가능):
    python eco_sm_resample.py 01_sm_csv.zip 02_sm_csv.zip ... 06_sm_csv.zip
    python eco_sm_resample.py            # 현재 폴더에서 01~06_sm_csv(.zip/폴더) 자동 탐색

결과 (현재 폴더):
    eco_sm_15min.csv, eco_sm_30min.csv : household, date, slot, power_mean_w, coverage
    eco_sm_hourly.csv                  : household, date, hour, power_mean_w, coverage
                                         (기존 eco_sm_to_hourly.py와 같은 형식 → 기존 코드 그대로 사용 가능)
  - slot         : 하루 안의 구간 번호 (15분이면 0~95, 30분이면 0~47)
  - power_mean_w : 해당 구간 powerallphases(전 상 유효전력 합, W)의 평균. 결측(-1)은 제외
  - coverage     : 해당 구간 초 중 유효값 비율 (0~1)
표준 라이브러리만 사용합니다. 세 파일 모두 작으니(수 MB) 이 파일들만 올려주세요.
"""
import csv
import io
import os
import re
import sys
import zipfile

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")
HH_RE = re.compile(r"(0[1-6])_sm")
BASE = 15 * 60          # 기본 구간 900초
NB = 86400 // BASE      # 96개


def base_bins(lines):
    """하루 파일 → 15분 구간별 (합, 유효개수)"""
    sums = [0.0] * NB
    counts = [0] * NB
    sec = 0
    for line in lines:
        if sec >= 86400:
            break
        field = line.split(",", 1)[0].strip()
        try:
            v = float(field)
        except ValueError:  # 헤더 행 등은 건너뜀
            continue
        if v > -1:
            b = sec // BASE
            sums[b] += v
            counts[b] += 1
        sec += 1
    return sums, counts


def merge(sums, counts, m):
    """15분 구간을 m분 구간으로 합침 (합과 개수를 합치므로 정확한 평균)"""
    f = m // 15
    out = []
    for s in range(NB // f):
        S = sum(sums[s * f:(s + 1) * f])
        C = sum(counts[s * f:(s + 1) * f])
        out.append((s, (S / C) if C else "", C / (m * 60)))
    return out


def iter_day_files(src):
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for name in sorted(z.namelist()):
                if DATE_RE.search(name):
                    with z.open(name) as f:
                        yield name, io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
    else:
        for root, _, files in os.walk(src):
            for fn in sorted(files):
                if DATE_RE.search(fn):
                    path = os.path.join(root, fn)
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        yield path, f


def main(srcs):
    specs = [(15, "eco_sm_15min.csv", "slot"), (30, "eco_sm_30min.csv", "slot"),
             (60, "eco_sm_hourly.csv", "hour")]
    files = {m: open(fn, "w", newline="") for m, fn, _ in specs}
    writers = {m: csv.writer(files[m]) for m in files}
    for m, _, col in specs:
        writers[m].writerow(["household", "date", col, "power_mean_w", "coverage"])
    try:
        for src in srcs:
            mm = HH_RE.search(os.path.basename(src.rstrip("/\\")))
            hh = mm.group(1) if mm else os.path.basename(src)
            n = 0
            for name, lines in iter_day_files(src):
                date = DATE_RE.search(name).group(1)
                sums, counts = base_bins(lines)
                for m in writers:
                    for s, p, cov in merge(sums, counts, m):
                        writers[m].writerow([hh, date, s, f"{p:.2f}" if p != "" else "", f"{cov:.3f}"])
                n += 1
            print(f"[{hh}] {src}: {n}일 처리", flush=True)
    finally:
        for f in files.values():
            f.close()
    print("완료 ->", ", ".join(fn for _, fn, _ in specs))


def find_sources(folder="."):
    found = {}
    for name in sorted(os.listdir(folder)):
        m = re.fullmatch(r"(0[1-6])_sm_csv(\.zip)?", name, flags=re.IGNORECASE)
        if m and (m.group(1) not in found or not m.group(2)):
            found[m.group(1)] = os.path.join(folder, name)
    return [found[k] for k in sorted(found)]


if __name__ == "__main__":
    srcs = sys.argv[1:] or find_sources()
    if not srcs:
        print(__doc__)
        print("현재 폴더에서 01_sm_csv.zip ~ 06_sm_csv.zip(또는 폴더)을 찾지 못했습니다.")
        print("현재 폴더:", os.path.abspath("."))
        sys.exit(1)
    print("처리 대상:", ", ".join(srcs))
    main(srcs)

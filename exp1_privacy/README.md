# 실험 1. 애그리게이터 프라이버시 노출 검증 (RQ1)

DCU가 관측하는 데이터로 대상 가구의 재실일/부재일을 구분할 수 있는지 측정합니다.

- 시나리오 A (AES-GCM 경로): DCU가 개별 가구 시계열을 평문으로 관측 (k = 1)
- 시나리오 B (LinHAE 경로): DCU는 k가구 합산값만 관측 (k ≥ 2)

현재 코드는 단편논문(ECO 단독, k = 1~5) 버전입니다. IDEAL 혼합 가구 풀, 15/30분 해상도 확장은 이후 커밋에서 추가합니다.

## 데이터 준비 (`data/` 폴더 하나에 모음)

- `data/occ/01_summer.csv`, `01_winter.csv`, … `05_winter.csv` : ECO 재실 라벨 (01–05번)
- `data/eco_sm_hourly.csv` (+ `eco_sm_15min.csv`, `eco_sm_30min.csv`) : 아래 1단계 결과

## 실행 순서 (저장소 루트에서)

| 단계 | 명령 | 하는 일 | 결과 파일 |
|---|---|---|---|
| 1 | `python exp1_privacy/preprocess/eco_sm_resample.py 01_sm_csv.zip … 06_sm_csv.zip` | 1초 계량값 → 15/30/60분 평균 전력 (현재 폴더에 출력, `data/`로 이동) | `eco_sm_15min.csv`, `eco_sm_30min.csv`, `eco_sm_hourly.csv` |
| 2 | `python exp1_privacy/preprocess/prep.py data` | 1초 재실 라벨 → 시간별 재실률 | `occ_hourly.csv` |
| 3 | `python exp1_privacy/attacks/sameplay.py data` | 오라클(Pearson·Combined) + 프로파일링 RF | `sameplay_results.csv` |
| 4 | `python exp1_privacy/attacks/attackers.py data` | 프로파일링 RF·GB·LR (3단계 결과 필요) | `attackers_raw.csv`, `attackers_results.csv` |
| 5 | `python exp1_privacy/attacks/crosshh.py data` | 교차 세대 RF·GB·LR | `crosshh_results.csv` |
| 6 | `python exp1_privacy/figures/fig1_ko.py data results [폰트.ttc]` | k별 우위 그래프 | `results/fig1_ko.png` |
| – | `python exp1_privacy/figures/overview_ko.py results 폰트.ttc` | 실험 개요 그림 | `results/fig_overview_ko.png` |
| – | `python exp1_privacy/figures/fig_profiles_ko.py data results 폰트.ttc` | 02번 세대 재실일·부재일 예시 | `results/fig3_profiles_ko.png` |

4·5단계는 오래 걸립니다(2코어 기준 각 10분 이상). 그림 스크립트는 한글 폰트(Noto Sans CJK 등) 경로가 필요합니다.

`eco_sm_resample.py`는 기존 `eco_sm_to_hourly.py`의 확장판이며, `eco_sm_hourly.csv`는 기존 스크립트와 같은 형식·값으로 출력됩니다.

## IDEAL 전처리 (혼합 가구 풀)

```
python exp1_privacy/preprocess/ideal_resample.py <IDEAL 폴더 또는 zip> --list      # 먼저 대상 파일 확인
python exp1_privacy/preprocess/ideal_resample.py <IDEAL 폴더 또는 zip> --workers 4
```

- 기본으로 파일 이름에 `electric-mains`, `electric-combined`가 모두 들어간 파일(가구별 mains 피상전력, 1Hz)을 처리합니다. 다르면 `--match`로 바꾸세요.
- UTC 시각을 영국 현지 시각(GMT/BST)으로 변환하고, ECO와 같은 형식의 `ideal_sm_15min.csv`, `ideal_sm_30min.csv`, `ideal_sm_hourly.csv`를 만듭니다. 가구 이름은 `h<번호>`입니다.

## 단편논문 표 1 각 열의 출처

| 표 1 열 | 파일 | 컬럼 |
|---|---|---|
| 오라클 Comb. / Pear. | `sameplay_results.csv` | `adv_oracle_Combined`, `adv_oracle_Pearson` |
| 프로파일링 RF / GB / LR | `attackers_results.csv` | `adv_RF`, `adv_GB`, `adv_LR` |
| 교차 세대 RF / GB / LR | `crosshh_results.csv` | `model`별 `adv` |

각 값은 대상 세대(01–03번)와 배경 조합별 우위를 k마다 평균한 것입니다.

## 공통 설정

- 유효 시간: 1시간 중 유효 측정 90% 이상, 유효일: 하루 22시간 이상
- 재실일: 주간(08–18시) 재실률 ≥ 0.8, 부재일: ≤ 0.5, 그 사이는 제외
- 대상 세대 01–03번, 배경은 대상 외 5개 세대 중 k−1개의 모든 조합
- 우위 = 2 × (균형 정확도 − 0.5)

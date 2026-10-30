# LinHAE AMI 실험

협대역 PLC 기반 AMI에서 LinHAE를 활용한 애그리게이터 프라이버시 보존 연구의 실험 코드입니다.

| 폴더 | 실험 | 연구 질문 | 상태 |
|---|---|---|---|
| `exp1_privacy/` | 실험 1. 애그리게이터 프라이버시 노출 검증 | RQ1: DCU가 복호화한 개별 계량값으로 재실 추론이 가능한가 | 단편논문 코드 이관, IDEAL 확장 예정 |
| `exp2_channel/` | 실험 2. 채널 실현가능성 분석 | RQ2: Middleton Class-A 잡음에서 PER, SNR\* | 예정 |
| `exp3_selection/` | 실험 3. 요구사항 기반 스킴 선택 vs 고정 스킴 | RQ3 | 예정 |

진행 순서: 실험 1 → 실험 2 → 실험 3

## 환경

```
pip install -r requirements.txt
```

Python 3.13 기준. 전처리 스크립트(`eco_sm_resample.py`)는 표준 라이브러리만 사용합니다.

## 데이터

데이터는 저장소에 포함하지 않습니다(`.gitignore`). 각 데이터셋을 직접 받아 `data/` 아래에 두세요.

- **ECO** (Electricity Consumption and Occupancy): 스위스 6가구, 1Hz 스마트미터 + 재실 라벨. ETH Zürich 배포
- **IDEAL**: 영국 255가구, 1초 mains 전력. University of Edinburgh DataShare 배포 (실험 1 확장에 사용 예정)

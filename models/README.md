# 공식 Stage 1 모델 (고정본)

실측 파이프라인(`pipeline/v4_core.py`, `pipeline/live.py`, `app.py`)이 읽는 학습 산출물의 고정 사본이다. 원시 텔레메트리나 세션 ID는 들어 있지 않고, 실측 학습 분할에서 계산한 통계값만 들어 있다.

| 파일 | 내용 |
|---|---|
| `round5b/stage1_v2.yaml` | Stage 1 설정. 코호트 키(작업군·신고 전력 대역·GPU), α=0.05, 증거 정의 |
| `round5b/fold_00/cohort_baseline_v2.json` | 코호트별 robust median/MAD |
| `round5b/fold_00/stage1_v2_calibration.json` | conformal 보정 점수, 증거 임계값, 정상 희소도 가중치 |
| `round5b/fold_00/stage1_alpha_sweep.json` | 보정 분할에서 고른 α |
| `session_gate.json` | 세션 경보 게이트: 후보 5창 연속, 교차검증 정상 세션 경보 0.052 |

## 읽는 순서

로컬에 다시 학습한 결과(`dataset/real/pipeline/refit/`, git 제외)가 있으면 그것을 먼저 읽고, 없으면 이 폴더를 읽는다(`pipeline/v4_core.py`의 `model_root`). 그래서 저장소를 새로 받아도 실측 데이터 없이 콘솔과 판정 경로가 동작한다.

## 다시 만들기 (실측 캡처가 있을 때)

1. `pipeline/fit_stage1_v2.py`로 round5b를 적합한다.
2. `pipeline/session_calibration.py`로 세션 게이트를 정한다.
3. 결과 파일을 이 폴더로 복사한다.

---
threat_id: B-NORMAL-001
category: benign
mitre_technique: null
evidence:
  - name: progress_log_available
    necessity: required
    description: "정상 워크로드는 progress/request 로그가 있는 경우가 많음"
  - name: log_period_mismatch
    necessity: exclusion
    description: "비교 가능한 느린 스텝이 전력 주기와 어긋나면 정상 문서 배제"
benign_lookalikes: []
grid_relevance:
  mechanism: none
  tier: unsupported
  notes: "DRAFT pending human approval. 오탐 방지용 정상 프로파일 집합"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/evidence_thresholds.json
attack_ids: []
profile: {declared: consistent, inst: 'no', util: coupled}
---
# 정상 워크로드 프로파일 (오탐 방지용 기준)

## 목적
이상 신호가 감지됐을 때, 그것이 아래의 "정상적으로 전력이 출렁이는 상황"과 일치하는지
확인하기 위한 기준 문서. 아래 패턴과 유사하면 공격이 아니라 정상일 가능성이 높다.

## 정상적으로 변동폭이 큰 워크로드들
- **대규모 분산 학습(정상)**: compute-heavy 위상과 communication-heavy(all-reduce) 위상이
  번갈아 나타나 전력이 출렁임. 단, 주기가 배치 크기·모델·네트워크에 종속되어 불규칙하고,
  그 사용자의 과거 학습 패턴과 일치함.
- **하이퍼파라미터 탐색**: 짧은 학습을 여러 번 반복 → 전력이 오르내림. 하지만 각 구간이 짧고
  job_type/사용자 이력으로 설명 가능.
- **체크포인트 저장**: 주기적으로 GPU가 잠깐 유휴(저장 중) → 전력 급락 후 복귀. 일회성 스파이크.
- **데이터 로딩 병목**: I/O 대기로 GPU가 잠깐 놀았다가 재개 → 불규칙한 전력 변동.
- **학습-평가(eval) 교차**: 학습 구간과 평가 구간의 부하가 달라 전력 프로파일이 두 레벨로 나뉨.

## 공격과의 구분 포인트
- 조서 `pattern`에 사각파(규칙적 duty·기계적 전환)가 없으면 정상 쪽으로 본다.
- serving/batch처럼 `step_end`가 없어 Stage1 `period_match`가 항상 false인 경우는
  그 칸만으로 정상을 배제하지 않는다. 도달 가능한 칸은 `progress_log_available`이다.
- `period_comparable`이고 `log_period_mismatch`이면 이 정상 문서는 쓰지 않는다.

## 정상 판정의 핵심 근거
- 변동의 원인이 맥락으로 설명됨 (예정된 작업, 알려진 사용자 패턴, 표준적 학습 동역학).
- 이 사용자가 과거에도 유사한 패턴을 보였음.
- job_type이 명확하고, 시간대·GPU 조합이 이 사용자에게 통상적임.

## 유지보수·운영성 변동 (정상)
- 드라이버 업데이트, 정기 유지보수, 벤치마크 실행 등은 일시적 전력 변동을 유발하나
  운영 일정과 대조하면 정상으로 확인 가능.

## Evidence 근거 (초안)
오탐 방지용 정상 프로파일 집합

## 관측 시그니처
(DRAFT, 사람 승인 전)
- 학습·추론·HPO 등 신고 작업과 맞는 전력 수준, 진행 로그 존재.
- 작은 전력 진폭 또는 불규칙한 요청 기반 변동, 뚜렷한 이상 증거 없음.
- 저전력 (low power) 대기·소형 서비스. 정상(normal) 판정 근거.
- 신고 작업 유형과 전력 패턴 일치 (consistent with declared job).

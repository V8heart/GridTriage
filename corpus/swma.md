---
threat_id: T-SWMA-001
category: cyber_physical_attack
mitre_technique: null
evidence:
  - name: log_period_mismatch
    necessity: required
    description: "비교 가능한 느린 스텝에서 전력 주기와 로그 스텝이 1×/2×로 맞지 않음"
  - name: period_comparable
    necessity: supporting
    description: "스텝이 빨라 비교 불가면 주기 불일치를 주장하지 않음"
  - name: flat_power
    necessity: exclusion
    description: "거의 평탄한 고부하는 SWMA 사각파가 아님"
  - name: cross_job_sync
    necessity: exclusion
    description: "여러 작업이 맞춰 움직이면 coordinated 문서다"
benign_lookalikes: []
grid_relevance:
  mechanism: electromechanical_oscillation
  tier: B
  notes: "DRAFT pending human approval. Bit2Watt SWMA: 별도 CUDA 커널의 규칙적 active/passive 전환 → 저주파 전기기계 외란 후보"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/evidence_thresholds.json
attack_ids: ["T1496.001"]
profile: {form: [square], level: [mid, high], log: [mismatch, na], multi_gpu: single, timescale: [
    seconds]}
---
# SWMA (Synthetic Workload Modulation Attack)

## 출처
Bit2Watt (Ji, Pan, Xu, arXiv:2607.05993, 2026), 4.2.1절.

## 공격 개요
공격자가 별도 CUDA 커널(active/passive 두 모드)을 올려, 호스트 컨트롤러가
unified-memory control flag로 두 모드를 정밀한 스케줄에 따라 전환한다. 권한 상승 불필요.

## 관측 가능한 정성적 특성 (탐지 근거로 사용)
- **평균 전력**: 정상 워크로드 범위를 벗어날 수 있음 (active 모드가 SM을 포화시켜 순간 전력이 TDP 근처까지 도달).
- **변동폭(swing)**: 매우 큼. active↔passive 전이가 뚜렷.
- **duty cycle 규칙성**: 매우 규칙적. 호스트가 정해진 스케줄로 전환하므로 주기가 일정하고 예측 가능.
- **주파수 제어 가능성**: 있음. 공격자가 명시적으로 조정 가능 (Table 1 기준 1.5~6kHz 대역).
- **워크로드 위장성**: 낮음. 정상 학습/추론과 무관한 별도 합성 커널이라, 프로파일링 시 정체불명의 커널로 보임.
- **지속성**: 공격이 실행되는 동안 지속적으로 반복.

## 저해상도 텔레메트리에서 남는 흔적 (우리가 실제로 관측하는 것)
- 원 신호(kHz)는 직접 관측 불가하나, 다운샘플링된 신호에서도 변동폭(분산)이 정상 대비 크게 증가.
- SM 사용률이 규칙적으로 100%↔0% 부근을 오가는 패턴.
- 정상 학습의 data-loading/compute 위상 전환보다 훨씬 규칙적이고 기계적임.

## 정상 워크로드와의 구분 포인트
- 조서 `pattern`이 창 중앙값 기준으로 사각파(규칙적 duty, 뚜렷한 swing, 분당 전환)를 보일 때만 SWMA를 주장한다.
- `period_comparable`이 참일 때만 로그 스텝과 전력 주기를 비교한다. 서브초 스텝(정상 학습, host-inherited SWMA)은 비교 불가다.
- 비교 가능하면 `log_period_mismatch`만 주기 불일치로 쓴다. Stage1 `period_mismatch`는 `strong_peak` 뒤에 가려져 있어 이 문서의 필수가 아니다.
- decoy 스텝이 우연히 맞을 수 있으므로 `period_match`만으로 배제하지 않는다.
- `cross_job_sync`가 켜지면 이 문서가 아니라 coordinated 문서다.

## Evidence 근거 (초안)
Bit2Watt SWMA: 별도 CUDA 커널의 규칙적 active/passive 전환

## 관측 시그니처
(DRAFT, 사람 승인 전)
- 전력 사각파 반복 (power square wave), 규칙적인 high/low 전환 (regular duty cycle).
- 주기 0.5–4초 (period 0.5–4 s), 큰 전력 진폭 swing 수백 W (large power swing).
- 뚜렷한 스펙트럼 첨두 (strong spectral peak), 진행 로그 스텝과 전력 주기 불일치 (log period mismatch).
- 평탄한 고부하가 아니다. 여러 GPU 동시 동기화면 coordinated 문서를 본다.

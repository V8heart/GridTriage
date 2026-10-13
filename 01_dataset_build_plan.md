# GridTriage(KDN repo) 1단계 — 데이터셋 구축 계획

> 대상 저장소: https://github.com/V8heart/KDN
> 목적: 이 문서를 그대로 Cursor 프롬프트로 사용해 데이터셋 구축 단계를 진행한다.
> 근거: 작품소개서(Cyber-Physical Threat Triage Framework) 4~7장의 요구사항 + 현재 리포 코드(`pipeline/`, `corpus/`, `data.csv`) 실측 분석.

---

## 0. 현재 상태 진단 (왜 이 데이터셋들이 필요한가)

리포를 직접 읽고 확인한 사실:

| 파일 | 현재 상태 | 문제/갭 |
|---|---|---|
| `data.csv` | 1,800행, 컬럼은 `timestamp, power_w, util_gpu_pct, label`뿐. `label`은 `normal`(1200) / `swma`(600) 2종류만 존재 | `corpus/`엔 `ltma.md`, `cryptojacking.md`가 이미 있는데 대응하는 라벨 데이터가 전혀 없음 → RAG 매칭·파이프라인 검증이 이 두 클래스에서는 아예 테스트되지 않음 |
| `pipeline/run_pipeline.py` | `context`에 `id_user`, `job_type`, `gres_req` 컬럼을 참조하는 코드가 이미 있음 | `data.csv`에 이 컬럼들이 없어 맥락 결합(작품소개서 §4.1 "Workload coupling", `ltma.md`가 강조하는 "맥락적 불일치" 판단)이 현재 코드에서 동작 불가 |
| `pipeline/features.py` | 단일 시계열(`power`, 선택적 `util`)만 입력받음. Magnitude/Ramp/Periodicity/Persistence 4개 특성군만 구현됨 | 작품소개서 §4.1의 **Synchronization**(다중 GPU coherence) 특성군이 코드·데이터 어디에도 없음. `data.csv`도 단일 GPU 기준이라 다중 GPU 동기화를 검증할 수 없음 |
| README 언급 `bit2watt_impl/` | 리포에 실제로 존재하지 않음(계획만 문서에 기술) | SWMA/LTMA를 "합성 신호"가 아니라 **실제 GPU에서 재현한 telemetry**로 검증하는 절차가 아직 없음 |
| 5장 Observability 지표(§5.2: timestamp cadence, value refresh ratio 등) | 코드/데이터 어디에도 없음 | DCGM의 실제 유효 갱신 간격을 측정하는 절차·데이터가 없어 "polling 간격 ≠ 물리 측정률" 주장을 검증할 방법이 없음 |

**결론**: 1단계 데이터셋 구축의 목적은 "그럴듯한 합성 데이터 늘리기"가 아니라, **작품소개서가 이미 주장한 내용과 현재 코드/데이터 사이의 간극을 메우는 것**이다. 아래 D0~D7은 이 간극 하나하나에 대응한다.

### 0.1 실측 서버 제약 (2026-09-06 확인)

- Ubuntu 22.04.5, NVIDIA Driver 580.173.02, 드라이버 보고 CUDA 13.0
- RTX 4090 24GB x2, DCGM 4.6.1, Python 3.10.12
- `.venv`의 PyTorch 2.14.0+cu130은 CUDA 사용 및 GPU 2장 인식 가능
- 시스템 `nvcc`는 CUDA Toolkit 11.5이므로 Ada(`sm_89`)를 직접 컴파일할 수 없음. SWMA 실측은 우선 PyTorch 연산 토글로 구현하고, 논문과 동일한 CUDA 커널 재현은 CUDA Toolkit 업그레이드 후 별도 트랙으로 둔다.
- DCGM `DCGM_FI_PROF_*`는 GeForce에서 지원되지 않는다. 또한 1~10ms 폴링은 호출 주기일 뿐 센서의 물리 갱신률을 뜻하지 않는다.
- 루트 파일시스템 여유 공간은 약 169GB이므로 장기 고주파 원시 로그는 세션 단위로 제한하고 압축하며 Git에 올리지 않는다.

---

## 1. 데이터셋 목록 (D0~D7)

### D0. 스키마 통일 — `data.csv` 확장
- **목적**: `run_pipeline.py`가 이미 기대하는 컬럼을 실제로 채워서 맥락 결합 로직이 동작하게 한다.
- **스키마(확정, DCGM 필드 실사 반영판 — 근거는 부록 A)**:
  ```
  timestamp, collection_timestamp, gpu_id, power_w, util_gpu_pct, mem_copy_util_pct,
  sm_clock_mhz, temp_c, fb_used_mb, pid, process_name,
  id_user, job_type, gres_req, label
  ```
  > 주의: 이전 초안의 `sm_active_pct`는 DCGM의 `DCGM_FI_PROF_SM_ACTIVE` 등 DCP(Datacenter Profiling) 계열로만 얻을 수 있는데, 이 계열은 **우리 실측 서버의 RTX 4090에서 지원되지 않는다** (부록 A 참조). 따라서 스키마에서 제외하고 `util_gpu_pct`(DCGM field 203 `gpu_utilization`, 4090에서 지원됨)로 대체한다. 합성 데이터도 이 제약에 맞춰 `sm_active_pct` 없이 생성한다.
  > `fb_used_mb`는 RTX 4090에서 DCGM `FB_USED`가 실제 NVML 사용량과 다르게 0을 반환한 실측 사례가 있으므로, **NVML memory info를 우선 사용**하고 실패 시 `FB_TOTAL - FB_FREE`로 계산한다.
  > `pid`/`process_name`은 accounting mode에 의존하지 않는다. NVML의 현재 실행 중인 compute-process API로 PID를 얻고 `/proc/<pid>/comm` 또는 `/proc/<pid>/cmdline`에서 이름을 결합한다. 프로세스가 없는 행은 null로 둔다.
  > `id_user`/`job_type`/`gres_req`는 DCGM이 주는 값이 아니라 **실험 하네스가 실행 시점에 직접 기록하는 그라운드트루스 메타데이터**다(부록 A §A.3). 합성 데이터에서는 임의 배정, 실측 데이터에서는 실행한 스크립트 정보를 그대로 기록한다.
- **구현 방식**: 기존 `data.csv` 생성 로직(현재 리포에 생성 스크립트가 없으므로 역산 필요 — `features.py`의 `__main__` 자체 테스트에 쓰인 파형 생성식과 동일한 패턴)을 `dataset/synth_common.py`로 분리하고, 위 컬럼을 채우는 방식으로 재작성한다.
- **저장 위치**: `dataset/synthetic/v2_schema/*.csv`
- **검증 기준**: `run_pipeline.py --telemetry <new.csv>`가 에러 없이 `context` 딕셔너리에 3개 컬럼을 모두 채워서 출력하는지 확인.

### D1. 정상 워크로드 데이터셋 — `corpus/normal_workloads.md`의 5개 패턴 각각
- **목적**: `normal_workloads.md`가 명시한 5가지 "정상인데 변동폭이 큰" 상황을 실제 라벨로 만들어 오탐 억제 성능을 검증한다. Stage1은 재현율 중심의 후보 생성기이므로 일부 정상 변동을 후보로 올려도 실패가 아니다. 최종 성공 기준은 **Stage1 후보율과 Stage3 최종 정상 판정률을 분리해 보고**, 후단에서 `normal_workloads`로 올바르게 기각하는 것이다.
- **하위 클래스 5개**:
  1. `normal_distributed_training` — compute/all-reduce 위상 전환, 주기 불규칙
  2. `normal_hpo_search` — 짧은 학습 반복, 각 구간 짧고 사용자 이력으로 설명 가능
  3. `normal_checkpoint` — 주기적 짧은 유휴 스파이크
  4. `normal_dataloader_stall` — 불규칙 I/O 대기 전력 변동
  5. `normal_eval_train_switch` — 두 레벨 부하 전환
- **구현 방식(합성)**: 각 패턴을 `dataset/synth_normal_patterns.py`에 함수로 구현. `features.py`의 `duty_regularity`, `periodicity_strength` 계산식으로 역검증 — 각 패턴이 SWMA(매우 규칙적)와 통계적으로 구분되는지 자체 테스트 포함.
- **구현 방식(실측, 우선순위 2)**: README의 실측 서버(4090 x2)에서 실제 분산학습/HPO 스윕/체크포인트 저장 스크립트를 돌리며 DCGM 폴링으로 캡처 (§2 D6과 동일 수집기 재사용).
- **저장 위치**: `dataset/synthetic/normal/*.csv` (1차), `dataset/real/normal/*.csv` (2차)

### D2/D3/D4. 공격 데이터셋 — SWMA / LTMA / Cryptojacking
공통 원칙: **corpus 문서에 이미 적힌 정성적 특성을 정량 신호로 역산해서 만든다.** (문서 → 신호가 아니라, 신호 → 문서 설명이 다시 나오는지가 검증 기준)

| 클래스 | 합성 신호 생성 로직 | 실측 재현 방법 | 우선순위 |
|---|---|---|---|
| **SWMA** (`swma.md`) | 이미 `features.py __main__`에 예시 있음(사각파, duty 매우 규칙적) → 이를 `dataset/synth_attacks.py`의 정식 함수로 승격, on/off 주기·진폭을 파라미터화 | 우선 `.venv`의 PyTorch CUDA 연산(큰 matmul 등)과 sleep을 일정 duty로 토글하는 `bit2watt_impl/swma_workload.py`를 구현해 캡처한다. 시스템 nvcc 11.5는 RTX 4090의 `sm_89`를 지원하지 않으므로 커스텀 CUDA 커널은 현재 기본 경로로 삼지 않는다. Bit2Watt의 1.5~6kHz 재현이 아니라 **소프트웨어 telemetry에서 관측되는 규칙적 변동** 검증이 목표다. | 1차: 합성 / 2차: 실측 |
| **LTMA** (`ltma.md`) | 평균은 정상 범위 유지, 변동폭은 있으나 불규칙(주기·진폭이 학습 동역학에 종속) — 실제 학습 루프의 배치 크기/옵티마이저 스텝 이벤트에 결합된 노이즈로 시뮬레이션 | 실제 PyTorch 학습 스크립트 안에 변조 컨트롤러를 삽입해(하이퍼파라미터 동적 조작) 실제 학습 중 DCGM 캡처 (`bit2watt_impl/ltma_inject.py`) | 1차: 합성 / 2차: 실측 |
| **Cryptojacking** (`cryptojacking.md`) | 평균 지속적으로 높음, 변동폭 작음(평평), 장시간 지속 — 낮은 분산의 고정 고부하 신호 | 실제 GPU 해시 연산 벤치마크(합법적 마이닝 벤치마크 툴, 실제 채굴 네트워크 접속 없이 로컬 연산만) 실행 후 캡처 | 1차: 합성 / 2차: 실측(선택) |

- **저장 위치**: `dataset/synthetic/attacks/{swma,ltma,cryptojacking}.csv`, `dataset/real/attacks/{swma,ltma,cryptojacking}.csv`
- **검증 기준**: 각 클래스의 `features_to_description()` 출력이 해당 corpus 문서의 "관측 가능한 정성적 특성" 문장과 사람이 봤을 때 일치하는지 체크리스트로 확인 (자동화하려면 D7의 RAG 평가셋과 연결).

### D5. 멀티 GPU 동기화 데이터셋
- **목적**: 작품소개서 §4.1 "Coordination(Synchronization)" 특성군을 실제로 구현·검증하려면 최소 2개 이상 GPU의 동시 시계열이 필요. 현재 `data.csv`는 단일 GPU라 이 특성군 자체가 코드에 없다.
- **구현 방식**: 실측 서버가 4090 x2이므로, (a) 정상 분산학습(all-reduce로 자연 동기화), (b) SWMA를 2개 GPU에 동시 주입(공격적 동기화)을 각각 캡처해서 `gpu_id`별로 롱포맷 CSV 저장.
- **저장 위치**: `dataset/real/multi_gpu/*.csv`
- **후속 코드 작업(데이터셋 자체는 아니지만 함께 계획)**: `features.py`에 `cross_correlation`, `coherence`, `synchronization_index` 함수 추가 필요 — 이 데이터로 즉시 단위테스트 가능.

### D6. Observability / Data Quality 메타데이터셋
- **목적**: 작품소개서 §5.1~5.2가 주장하는 "DCGM timestamp는 수집 시점이며 새 하드웨어 측정을 강제하지 않는다"는 논지를 실측으로 뒷받침.
- **구현 방식**: `bit2watt_impl/collect_telemetry.py`(신규)는 **pynvml을 주 수집 경로**로 사용하고 DCGM은 교차검증 옵션으로 둔다. 10ms/50ms/100ms/1000ms 호출 주기에서 collection timestamp, API 반환값, 이전 값 대비 변경 여부를 기록한다. API가 하드웨어 원시 측정 timestamp를 제공하지 않는 필드는 `raw_timestamp`를 null로 두고 collection timestamp를 하드웨어 측정 시각처럼 표기하지 않는다.
- **해석 원칙**: 요청 주기와 센서 갱신 주기는 다르다. 10ms 폴링 결과를 100Hz 측정으로, 1ms 폴링 결과를 1kHz 물리 측정으로 주장하지 않는다. 동일 값 반복 비율과 값 변경 간격으로 **실효 갱신률**을 별도 보고한다.
- **산출 지표**: collection cadence 분포, 호출 지연, value refresh ratio, 고유값 비율, 값 변경 간격 분포, 필드별 결측/N/A 비율.
- **저장 위치**: `dataset/real/observability/*.csv` + `dataset/real/observability/summary.md`(측정 결과 요약)

### D7. RAG 평가셋 (retrieval quality + open-set)
- **목적**: `rag_analyzer.py`의 `SignatureRetriever.search()`가 실제로 옳은 문서를 상위에 매칭하는지 정량 평가할 라벨셋이 현재 없음 (self-test는 2개 케이스뿐).
- **구성**:
  1. **Positive set**: D1~D4에서 만든 각 케이스의 `features_to_description()` 출력 → 정답 `threat_id`(또는 `normal`) 쌍. JSONL로 저장.
  2. **Open-set/Unknown 케이스**: corpus에 없는 새로운 패턴(예: 임의의 새 합성 변조 패턴)을 만들어 "어느 문서와도 강하게 매칭되지 않아야 정상"임을 검증 — 8.1절 E5(Generalization) 실험의 최소 버전.
- **저장 위치**: `dataset/eval/retrieval_eval.jsonl`
- **검증 기준(자동화 가능)**: top-1 정확도, unknown 케이스의 최대 유사도가 임계치 이하인지.

---

## 2. 구축 우선순위 (Cursor 작업 순서)

이 순서대로 브랜치/커밋을 나눠 진행할 것을 권장.

1. **Phase 1 (합성, GPU 불필요, 즉시 가능)**
   - D0 스키마 확장 + D1(정상 5패턴, 합성) + D2/D3/D4(공격 3클래스, 합성)를 한 번에 진행.
   - 결과물: `dataset/synthetic/` 하위에 클래스별 CSV, 통합본 `dataset/synthetic/all_v2.csv` (현재 `data.csv`를 대체/보완).
   - 완료 기준: `run_pipeline.py --telemetry dataset/synthetic/all_v2.csv --rag-backend tfidf --llm-backend stub` 실행 시 3개 공격 클래스 + 5개 정상 패턴을 처리한다. Stage1 후보율과 최종 판정을 따로 평가하며, 정상 변동이 Stage1 후보가 되는 것은 허용하되 후단에서 `normal_workloads`로 기각되는지 확인한다.

2. **Phase 2 (실측 인프라, GPU 필요, 공격 재현 이전)**
   - D6 Observability 수집기(`bit2watt_impl/collect_telemetry.py`) 먼저 구현 — pynvml을 기본으로 하고 DCGM을 교차검증 옵션으로 제공한다. 공격이 없어도 idle 상태로 바로 실행 가능하고 이후 모든 실측 수집에 재사용한다.
   - 완료 기준: §5.2 표의 7개 지표 중 최소 timestamp cadence, value refresh ratio 2개는 실측값으로 채워짐.

3. **Phase 3 (실측 정상 워크로드)**
   - D1의 5패턴을 실제 서버에서 캡처(2차 우선순위였던 실측 버전).
   - 완료 기준: 합성 버전과 실측 버전의 `features_to_description()` 출력이 질적으로 유사한지 비교 로그 남김.

4. **Phase 4 (실측 공격 재현)**
   - `bit2watt_impl/swma_workload.py`(PyTorch 연산/sleep 토글), `ltma_inject.py` 구현 후 D2/D3 실측 캡처.
   - cryptojacking 실측은 선택(로컬 벤치마크로 대체 가능, 실제 채굴 네트워크 접속 금지).
   - 완료 기준: 실측 SWMA 신호가 합성 SWMA와 통계적으로 유사한 `swing_ratio`/`duty_regularity`를 보이는지 확인.

5. **Phase 5 (멀티 GPU + 평가셋)**
   - D5(멀티 GPU 동기화) 캡처 + `features.py`에 synchronization 피처 함수 추가.
   - D7(RAG 평가셋) 구성, `pipeline/eval_retrieval.py`(신규) 작성해 top-1 정확도 자동 계산.
   - 완료 기준: 8장 실험 설계(E3/E4/E5)의 최소 버전이 이 데이터셋만으로 재현 가능.

---

## 3. 저장소 구조 제안 (신규 폴더)

```
KDN/
├── dataset/
│   ├── synth_common.py          # 공통 파형/노이즈 생성 유틸
│   ├── synth_normal_patterns.py # D1 (5패턴)
│   ├── synth_attacks.py         # D2/D3/D4 (SWMA/LTMA/Cryptojacking)
│   ├── synthetic/
│   │   ├── normal/*.csv
│   │   ├── attacks/*.csv
│   │   └── all_v2.csv
│   ├── real/
│   │   ├── normal/*.csv
│   │   ├── attacks/*.csv
│   │   ├── multi_gpu/*.csv
│   │   └── observability/*.csv + summary.md
│   └── eval/
│       └── retrieval_eval.jsonl
├── bit2watt_impl/
│   ├── collect_telemetry.py     # D6 공통 수집기 (pynvml 기본, DCGM 교차검증)
│   ├── swma_workload.py         # D2 PyTorch 연산/sleep 토글 실측
│   └── ltma_inject.py           # D3 실측 재현
└── pipeline/
    └── eval_retrieval.py        # D7 자동 평가 스크립트 (신규)
```

---

## 4. Cursor에 그대로 붙여넣을 프롬프트

> 아래 블록만 복사해서 Cursor에 붙여넣으면 됨.

```
이 리포(V8heart/KDN, GridTriage 파이프라인)의 1단계 데이터셋 구축을 진행해줘.

컨텍스트:
- 현재 data.csv는 timestamp, power_w, util_gpu_pct, label(normal/swma) 뿐이고,
  corpus/에는 swma.md, ltma.md, cryptojacking.md, normal_workloads.md 4개 문서가 있는데
  ltma/cryptojacking 라벨 데이터가 전혀 없어. run_pipeline.py는 id_user/job_type/gres_req
  컬럼을 이미 참조하는데 data.csv엔 없어서 맥락 결합 로직이 동작하지 않는 상태야.
  features.py에는 다중 GPU synchronization(coherence) 피처가 아예 구현돼 있지 않아.

목표: 아래 순서로 dataset/ 폴더를 신설하고 데이터셋을 구축해줘.

Phase 1 (합성, 지금 바로):
1. dataset/synth_common.py에 공통 파형/노이즈 생성 유틸 작성
2. dataset/synth_normal_patterns.py에 정상 5패턴 구현:
   normal_distributed_training, normal_hpo_search, normal_checkpoint,
   normal_dataloader_stall, normal_eval_train_switch
   (각각 normal_workloads.md의 설명과 부합하는 통계적 특성을 갖도록)
3. dataset/synth_attacks.py에 swma, ltma, cryptojacking 합성 신호 구현
   (각각 swma.md/ltma.md/cryptojacking.md의 "관측 가능한 정성적 특성" 절과
   부합하는 mean/swing_ratio/duty_regularity/periodicity_strength가 나오도록,
   features.py의 compute_window_features로 직접 검증하면서 파라미터 튜닝)
4. 스키마는 다음으로 통일 (RTX 4090에서 DCGM으로 실제 지원되는 필드만 사용,
   근거는 부록 A):
   timestamp, collection_timestamp, gpu_id, power_w, util_gpu_pct,
   mem_copy_util_pct, sm_clock_mhz, temp_c, fb_used_mb, pid, process_name,
   id_user, job_type, gres_req, label
   (id_user/job_type/gres_req는 DCGM이 주는 값이 아니라 실험 하네스가 직접
   기록하는 메타데이터임 — 부록 A §A.3 참고)
5. dataset/synthetic/all_v2.csv로 통합 (정상 5클래스 + 공격 3클래스 혼합,
   기존 data.csv 대비 클래스 다양성 확보)
6. python pipeline/run_pipeline.py --telemetry dataset/synthetic/all_v2.csv
   --rag-backend tfidf --llm-backend stub 로 실행. Stage1 후보율과 최종 판정을
   분리해 평가하고, 정상 변동은 Stage1 후보가 될 수 있으나 최종적으로
   normal_workloads로 기각되는지, 공격은 올바른 corpus 문서와 매칭되는지 확인

Phase 2 (실측 인프라 — GPU 서버 접속 가능해지면):
7. bit2watt_impl/collect_telemetry.py 작성: pynvml을 기본 수집 경로로, DCGM을
   교차검증 옵션으로 사용. 다양한 polling 주기(10ms/50ms/100ms/1000ms)에서
   collection timestamp, API 반환값, 값 변경 여부를 기록한다. API가 하드웨어
   원시 timestamp를 제공하지 않으면 raw_timestamp는 null로 저장한다.
   polling 주기를 물리 측정률로 간주하지 말고 value refresh ratio·값 변경 간격으로
   필드별 실효 갱신률을 계산한다. fb_used_mb는 NVML memory info를 우선 사용하고
   DCGM을 쓸 경우 FB_TOTAL-FB_FREE로 계산한다.
   수집할 필드는 반드시 계획 문서(01_dataset_build_plan.md) 부록 A.3의 Tier 1
   목록만 사용할 것 (DCGM_FI_PROF_* 계열은 RTX 4090에서 지원되지 않으므로
   시도하지 말 것 — 부록 A.2 근거 참고). id_user/job_type/gres_req는 DCGM
   응답이 아니라 실행 스크립트의 커맨드라인 인자에서 가져와 같은 행에 join.
   pid/process_name은 accounting mode를 전제로 하지 말고 NVML running-process API와
   /proc/<pid>/comm 또는 cmdline을 조합해 기록할 것.
8. 이후 모든 실측 데이터 캡처(정상/공격/멀티GPU)는 이 수집기를 재사용.
   각 캡처가 무엇에 대응하는지는 부록 A.5(학습 프로세스 → 데이터셋 매핑)
   표를 그대로 따를 것

Phase 3~5는 계획 문서(01_dataset_build_plan.md) 2절의 Phase 3/4/5를 그대로 따라줘.

각 Phase가 끝나면 결과(생성된 파일 목록, run_pipeline.py 실행 로그, 통계 검증 결과)를
요약해서 알려줘. 코드는 기존 리포 스타일(주석 위주 한국어 docstring, numpy/pandas 기반,
무거운 딥러닝 의존성 최소화)을 그대로 따라줘.
```

---

## 부록 A. DCGM 필드 선정 근거

> D0 스키마와 Phase 1/2의 컬럼 목록이 왜 이렇게 정해졌는지에 대한 상세 근거. Cursor가 `collect_telemetry.py`나 `synth_common.py`를 작성할 때 이 부록을 그대로 참고하면 됨.

### A.1 결론: "지원되는 필드를 다 모으기"가 아니라 두 조건의 교집합만 쓴다

1. **우리 실제 GPU(RTX 4090 x2)에서 데이터가 실제로 나오는 필드인가**
2. **작품소개서 §4.1의 7개 특성군(Magnitude/Ramp/Periodicity/Persistence/Synchronization/Workload coupling/Data quality) 중 하나에 대응되는가**

이 두 조건의 교집합만 필수(Tier 1)로 잡는다. 이유는 (a) 데이터센터 GPU 전용 필드는 애초에 우리 하드웨어에서 값 자체가 안 나오고, (b) 나머지 절반(ECC 세부, inforom, NVSwitch, MIG, vGPU 등)은 우리 위협모델과 무관해 차원만 늘리고 해석성을 낮추기 때문이다.

### A.2 RTX 4090(GeForce, 비-데이터센터 GPU)의 DCGM 지원 범위

| 카테고리 | 대표 필드 | 4090 지원 여부 |
|---|---|---|
| 전력 | `BOARD_POWER_WATTS`, `TOTAL_ENERGY_CONSUMPTION`, `POWER_VIOLATION` | ✅ |
| 기본 사용률 | `GPU_UTIL_RATIO`, `MEM_COPY_UTIL`, `ENC_UTIL`, `DEC_UTIL` | ✅ |
| 클럭 | `SM_CLOCK`, `MEM_CLOCK`, `MAX_SM_CLOCK` | ✅ |
| 온도 | `GPU_TEMP_CELSIUS` 및 임계온도 | ✅ |
| 메모리 | `FB_FREE`/`FB_TOTAL` | ✅. 단, 실측에서 DCGM `FB_USED`가 0을 반환하여 `fb_used_mb`는 NVML 우선 또는 `FB_TOTAL-FB_FREE`로 계산 |
| 실행 중 프로세스 | NVML compute-running-process API + `/proc/<pid>` | ✅. 현재 실행 중 PID/이름 수집 가능 |
| DCGM accounting 통계 | `PROCESS_ACCOUNTING_STATS` | ⚠️ 현재 비활성이고 활성화에 sudo 필요. 필수 수집 경로에서 제외 |
| PCIe 링크 상태 | `PCIE_LINK_GEN`/`LINK_WIDTH` | ✅ |
| PCIe 처리량 | `PCIE_TX_THROUGHPUT`/`PCIE_RX_THROUGHPUT` | ❌ 실측에서 `N/A`; Tier 1/2에서 제외 |
| **프로파일링(DCP)**: SM/Tensor/DRAM active, PCIe·NVLink 처리량 bytes | `DCGM_FI_PROF_*` 전체 | ❌ 데이터센터 GPU(A100/H100/T4 등) 전용, GeForce 미지원 |
| NVLink/NVSwitch | `NVLINK_*`, `NVSWITCH_*` | ❌ 4090은 NVLink 자체가 없음(Ada 세대 GeForce부터 제거) |
| MIG / vGPU / Fabric Manager / C2C | 관련 필드 전부 | ❌ 해당 하드웨어 자체가 없음 |
| ECC 세부(SBE/DBE, row-remap 등) | `ECC_SBE_*`, `ROW_REMAP_*` | 대부분 ❌ (GeForce는 ECC 메모리 아님) |

**실무적 함의**: "SM active/Tensor active 같은 고해상도 워크로드 지문"은 DCGM으로는 4090에서 얻을 수 없다. 이게 꼭 필요하면 CUPTI/Nsight 같은 별도 프로파일링 경로가 필요한데, 이는 작품소개서 §9.2가 이미 규정한 "Server-only telemetry = DCGM/NVML 범위"를 벗어나므로 1차 구현에서는 시도하지 않는다. 대신 `util_gpu_pct`/`mem_copy_util_pct`/`power_w`/`sm_clock_mhz`/`fb_used_mb`만으로 구분 가능함을 보이는 것이 "저해상도 telemetry의 한계 안에서 어디까지 가능한가"라는 우리 논지에 더 부합한다. `memory_temp`는 실측에서 0, PCIe TX/RX throughput은 `N/A`였으므로 수집 스키마에서 제외한다.

### A.3 Tier 1 / Tier 2 / 제외 컬럼

| Tier | 컬럼 | 근거 |
|---|---|---|
| Tier 1 (필수, 시계열 특성) | `collection_timestamp`, `gpu_id`, `power_w`, `util_gpu_pct`, `mem_copy_util_pct`, `sm_clock_mhz`, `temp_c`, `fb_used_mb` | Magnitude/Ramp/Periodicity 계산(`features.py`)의 직접 입력. `fb_used_mb`는 NVML 우선 또는 `FB_TOTAL-FB_FREE` |
| Tier 1 (품질 메타데이터) | `raw_timestamp`(API가 제공할 때만; 아니면 null), `requested_interval_ms`, `actual_interval_ms`, `value_changed` | polling 주기와 실효 센서 갱신률을 분리하기 위한 값 |
| Tier 1 (필수, 프로세스 맥락) | `pid`, `process_name` | NVML running-process API + `/proc`로 수집. 프로세스가 없으면 null |
| Tier 2 (선택) | `fan_speed_pct`, `pstate`, `pcie_link_gen/width`, `power_limit_enforced_w` | 위협 신호는 아니지만 스로틀링 등 혼입 요인을 배제하는 데 유용 |
| 제외 | `DCGM_FI_PROF_*` 전체, `memory_temp`, PCIe TX/RX throughput, NVLink/NVSwitch/MIG/vGPU/C2C 전체, ECC 세부, inforom, diagnostic 결과 | A.2 참조 — 4090에서 미지원/N/A이거나 위협모델과 무관 |

### A.4 `id_user` / `job_type` / `gres_req`는 DCGM 필드가 아니다

NVML은 현재 실행 중인 compute process의 PID와 GPU 메모리 사용량을 제공하며, 프로세스명은 `/proc/<pid>/comm` 또는 `cmdline`로 보완한다. DCGM accounting mode는 sudo가 필요하고 현재 비활성이라 필수 경로로 사용하지 않는다. `id_user`, `job_type`, `gres_req`는 **실험 하네스가 실행 시점에 직접 기록해야 하는 그라운드트루스 메타데이터**다 (예: "이 시간창은 job_type=distributed_training, id_user=sim_userA로 내가 직접 실행시킨 것"이라고 수집 스크립트가 별도로 로그에 남김). `collect_telemetry.py`를 구현할 때 DCGM/NVML 응답에서 이 컬럼들을 찾으려 하지 말고, 실행 스크립트의 커맨드라인 인자나 설정 파일에서 가져와 같은 행에 join해야 한다.

### A.5 학습 프로세스 → 데이터셋 매핑

D1/D2/D3/D4/D5를 실제로 "무엇을 실행해서 얻는가"로 연결하면:

| # | 실행할 프로세스 | 대응 데이터셋 | GPU 구성 |
|---|---|---|---|
| 1 | 소형 모델 단일 GPU 학습 반복 | 정상 대조군 | 1장 |
| 2 | 동일 모델 `torch.distributed`(DDP) 2-GPU 분산학습 | D1 `normal_distributed_training` + D5 정상 동기화 케이스 | 2장 |
| 3 | 배치크기/러닝레이트를 바꿔가며 짧은 학습 반복(HPO 스윕) | D1 `normal_hpo_search` | 1장 |
| 4 | N step마다 체크포인트 저장 로직 삽입 | D1 `normal_checkpoint` | 1장 |
| 5 | 데이터로더에 인위적 지연 삽입 | D1 `normal_dataloader_stall` | 1장 |
| 6 | 학습 N epoch / 평가 M epoch 교차 반복 | D1 `normal_eval_train_switch` | 1장 |
| 7 | PyTorch CUDA 연산/sleep 토글 워크로드 실행 | D2 `swma` | 1장 |
| 8 | 동일 토글 워크로드 2-GPU 동시 실행 | D5 공격적 동기화 케이스(#2와 대조) | 2장 |
| 9 | LTMA 컨트롤러를 #1 학습 루프에 삽입 | D3 `ltma` | 1장 |
| 10 | 로컬 GPU 해시 벤치마크(실제 채굴 네트워크 접속 없음) 장시간 실행 | D4 `cryptojacking` | 1장 |

### A.6 구현해야 하는 공격 기법 (corpus 근거 기반, 3개로 충분)

| 공격 | 구현 방법 |
|---|---|
| **SWMA-like** | 현재 시스템 nvcc 11.5는 RTX 4090의 `sm_89`를 컴파일하지 못하므로, `.venv`의 PyTorch 2.14.0+cu130에서 큰 CUDA matmul(active)과 sleep(passive)을 일정 duty로 토글한다. `bit2watt_impl/swma_workload.py`. 이는 논문의 unified-memory persistent kernel을 동일 재현하는 것이 아니라 **소프트웨어 telemetry에서 관측 가능한 규칙적 변동의 검증용 근사**임을 결과에 명시한다. |
| **LTMA** | 별도 커널 없이 실제 PyTorch 학습 스크립트(A.5의 #1) 안에 변조 컨트롤러를 삽입해 특정 iteration마다 batch size/보조 연산을 동적으로 조정. `bit2watt_impl/ltma_inject.py` — 반드시 "기존 학습 스크립트를 감싸는 wrapper" 형태로 만들어야 "평균은 정상 유지"라는 `ltma.md`의 설명이 실제로 재현됨 |
| **Cryptojacking** | 새로 코드를 짜기보다 **기존에 검증된 오픈소스 GPU 해시레이트 벤치마크**(로컬 연산만, 실제 마이닝 풀 접속 없음)를 그대로 실행해 캡처 |

corpus에 없는 새 공격 카테고리를 추가로 구현할 필요는 없음(근거 없는 위협을 추가하면 인용 체계가 깨짐). D7의 open-set/unknown 케이스는 별도 공격이 아니라 순수 평가용 합성 신호로 충분.

---

## 5. 참고: 이번 계획이 반영하지 않은 것

- 계통 시뮬레이션(Physics Validation Layer), 실제 PDU/UPS/PCC 전력계 데이터 — 작품소개서 9.3절에서 이미 "2차 확장 과제"로 분리했으므로 이번 데이터셋 구축 범위에 포함하지 않음.
- kHz 급 원신호 재현 — README에도 명시된 대로 "물리 계측 영역, 우선순위 제외" 원칙을 그대로 따름. 모든 합성/실측 데이터는 DCGM/NVML이 실제로 관측 가능한 해상도 기준으로만 만든다.

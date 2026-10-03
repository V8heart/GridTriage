# 실측 실패 진단과 단계별 수정안 (v3)

작성일: 2026-10-02
참고: KDN_작품소개서_v13 전문, `dataset/real/pipeline/failure_modes_report.md`,
`refit/round3/stage1_report.md`, `refit/round3/stage2_full_report.md`,
`stage2_period_sync_measure.md`, 실측 78세션 원본 `telemetry.csv`, Bit2Watt 원문(arXiv 2607.05993)
상태: **제안 — 사람 승인 전 구현하지 않음**

---

## 0. 한 줄 요약

실측이 안 된 이유는 하나가 아니라 **네 종류**이고, 각각 처방이 다르다.

| 종류 | 예 | 처방 |
|---|---|---|
| **A. 코드 버그** (고치면 실측으로 얻을 수 있음) | `power_instant_w` 파싱 버그, SWMA 비동기 큐 적체 | 수리 + **기존 78세션 복원** (재수집 최소화) |
| **B. 캡처 설계상 의도적으로 안 보이게 만든 것** | piggyback 0.05–0.11 s, LTMA 30 s 위상 + 10 ms aux, 소형 serving | 설계 재조정 후 재캡처, 또는 "관측 불가 대조군"으로 재분류 |
| **C. NVML 물리 한계** (원리적으로 관측 불가) | Bit2Watt 원형 SWMA 1.5–6 kHz, LTMA 1.2–3 kHz, >2 GPU 동기화 | **보정된 관측 연산자로 합성** + "판정 불가" 출력이 정답 |
| **D. 파이프라인 로직** | Stage2 bool 조서, corpus 신분증, RAG 저유사도, normal 경로 부재 | 단계별 수정 (§5) |

**가장 중요한 새 발견**: 수집기가 `power_instant_w`를 잘못 읽어 쓰레기 값(1e-319)으로 저장했지만,
**비트를 재해석하면 78세션 전부 복원 가능**하다. 복원한 순시 전력은 0.5–1 s SWMA와 LTMA를 그대로 보여 준다.
즉 "NVML이라 못 본다"고 결론 낸 것 중 상당 부분은 **물리 한계가 아니라 버그**였다.

---

## 1. 파이프라인 큰 그림 (현재)

```
NVML 수집 (10 Hz 요청)  ──►  Stage1 v2 (30s 창/15s stride, 코호트 robust-z → Mondrian conformal, idle 게이트)
                                │ 후보만
                                ▼
                         Stage2 v2: 증거 bool 번들 + pattern 카드
                                │
                         RAG (TF-IDF, 설명문 ↔ corpus 본문) → top-3 문서
                                │
                         judge (stub 체크리스트 / Ollama gemma3:12b) → known|partial|unknown|normal
```

v13 소개서의 서술 구조(Stage 0~5, Mode A/B, 관측 연산자 y[k]=Q{(h*x)(kTs)}+ε)는 맞다.
다만 실측에서는 **Stage 0(관측 경계)이 결선되지 않은 것**이 모든 층의 실패를 키운다.

---

## 2. 실측 실패 진단 (원인별)

### 2.1 [A] 코드 버그

#### A-1. `power_instant_w` 파싱 버그 — **확정**

`bit2watt_impl/collect_telemetry.py:190-202`

```python
for attr in ("dVal", "ullVal", "uiVal", "ull_val"):
    if hasattr(raw, attr):
        return float(val) / 1000.0 if attr != "dVal" else float(val)
```

NVML field value는 union이라 `dVal`이 **항상 존재**한다. `valueType`을 보지 않고 첫 속성 `dVal`을 읽으므로,
mW 정수 비트를 double로 해석해 1e-319 대 denormal이 저장됐다.

복원 검증 (`s-0adfee7f7507b344`, GPU0):

| power_w (평균) | power_instant_w (저장값) | 비트 재해석 → W |
|---|---|---|
| 294.4 | 1.64e-318 | **331.6** |
| 291.5 | 1.53e-318 | **310.6** |
| 293.9 | 1.33e-318 | **268.8** |

CSV에 6자리 유효숫자로 저장돼 mW 해상도가 유지된다 → `np.float64.view(uint64)/1000`으로 전량 복원 가능.

**78세션 타깃 GPU, 평균 전력 vs 복원 순시 전력 (p90−p10 swing, W, 세션 중앙값)**

| 워크로드 | swing 평균 | swing 순시 | 해석 |
|---|---:|---:|---|
| finetune (숙주) | 1.6 | 11.4 | 기준 |
| **LTMA** | 9.8 | **93.8** | 순시에서 숙주 대비 8배 → **관측 가능** |
| swma_piggyback | 2.8 | 17.0 | 여전히 숙주와 비슷 (0.05–0.11 s, 진짜 한계) |
| swma_mimicry | 3.9 | 17.9 | 동일 |
| swma_basic | 120 | **412** | |
| crypto_flat | 8.0 | 10.9 | 평탄 그대로 (물리적으로 맞음) |
| FSDP / dataloader / HPO | 8.6 / 17 / 30 | 64 / 72 / 119 | 정상도 순시 swing이 큼 → 코호트 기준선 필요 |

**SWMA 주기별 (같은 세션, swing W)**

| 주기 | swing 평균 | swing 순시 | 지배 주파수(순시) |
|---|---:|---:|---|
| 4 s | 335–358 | 403–415 | 0.25 Hz ✓ |
| 2 s | 107–296 | 133–418 | 0.50 Hz ✓ |
| 1 s | 18–49 | **194–409** | **1.00 Hz ✓** (평균에선 0.06 Hz로 소실) |
| 0.5 s | 1–39 | 6–408 | 0.06 Hz (aliasing) |

해석:
- `power_w`는 약 **1 s boxcar 평균**이다. 1 Hz에서 sinc 영점, 0.5 Hz에서 ~0.64배로 실측과 정확히 맞는다.
- 두 필드 모두 값 변경률이 0.20이다. 100 ms 폴링에서 **갱신이 약 500 ms마다** 일어난다는 뜻이고, Nyquist는 약 1 Hz다.
- 그래서 순시 전력으로는 **≤1 Hz까지 관측 가능**하고, 0.5 s 주기(2 Hz)는 순시에서도 접혀 사라진다. 이는 진짜 한계(C)다.
- failure_modes_report의 "0.5–1 s는 430 W에 붙어 주기가 사라진다"는 **1 s 칸에서는 틀린 결론**이다. 버그 때문에 평균 필드만 봤기 때문이다.

#### A-2. SWMA active 루프의 비동기 큐 적체 — **부분 확인, 검증 필요**

`bit2watt_impl/swma_workload.py:84-89`. active 구간에서 `torch.mm`을 동기화 없이 호스트 시간 기준으로 계속 enqueue하고,
루프가 끝난 뒤 한 번만 `synchronize()`한다. 4096² fp16 mm은 4090에서 약 1 ms라
enqueue가 실행보다 빠르면 적체가 쌓이고, 그 drain 시간만큼 passive 구간이 잠식된다.

순시 전력에서도 0.5 s 칸 일부(jitter 0.5 s: 9.8 W, coordinated 0.5 s: 5.8 W / 11.2 W)가 평탄하다.
같은 0.5 s의 다른 세션은 400 W swing이므로, **세션에 따라 적체가 passive를 지운 것**으로 보인다.
처방: mm 블록마다 `torch.cuda.Event` 기반 시간 예산을 두거나 N회마다 동기화한다.
`attack_intervals`의 실제 active 길이를 요청 duty와 비교하면 바로 확인된다.

#### A-3. 합성기의 NVML 모델이 실측과 다름 — **확정**

`dataset/synth_attacks.py`: `swma()`, `ltma()`, `cryptojacking()`의 기본값이 `nvml_avg_window_s=0.0`이라 평활이 없다.
갱신 주기(500 ms sample-and-hold)는 어디에도 모델링되지 않았다.
합성 기본 SWMA는 실측보다 낙관적이다. 이것이 합성–실측 격차(v13 §4.2.3 (5))의 직접 원인 중 하나다.

#### A-4. 창 정답 라벨링이 희소한 공격 구간을 놓침 — **확정, 수정함 (2026-10-03)**

`dataset/window_truth.py`의 `window_is_attack`은 "30 s 창의 50% 이상이 공격 구간과 겹치면 공격"으로 판정한다.
- LTMA는 기록된 구간이 보조 연산 1회(약 0.6 ms)씩이라 겹침이 0.15%다. 그래서 **LTMA 창 전부가 숙주 정상(finetune)으로 라벨링**됐다.
  test의 LTMA 209창이 정상으로 채점됐으므로, LTMA를 잡으면 오히려 오탐으로 세어지는 구조였다.
- SWMA shallow는 duty ≤ 0.5라 43%가 `normal_idle`로 라벨링됐다.
- train/cal에서 공격 창 약 100개가 정상 기준선에 섞였다.

수정: 5 s 이내 간격의 공격 구간을 하나로 이은 뒤 겹침을 판정한다(`ATTACK_GAP_BRIDGE_S`).
LTMA의 30 s 무주입 구간은 실제 공격 연산이 없으므로 숙주 정상으로 남는다. 회귀 테스트도 추가했다.

#### A-5. tail-min 점수의 해상도 포화 — **확정**

`empirical_tail_p`는 참조값 n개 기준으로 p가 1/(n+1)보다 작아지지 않는다.
- 정상 창도 10개 피처 중 하나쯤은 이 하한에 닿는다. 그래서 극단값 영역에서 공격과 정상이 구분되지 않는다.
- 예: LTMA 창은 순시 z=6.3이었지만 u=6.33으로, cal 기준선 7.9에 못 미쳤다.

지수 꼬리 외삽(`tail_extrapolation: exponential`)을 옵션으로 추가했다.
그러나 test 정상 오탐이 13% → 29–40%로 늘어 **채택하지 않았다**. 세션 수가 적은 코호트에서 cal과 test의 분포 차이가 증폭되기 때문이다.

### 2.2 [B] 캡처 설계상 의도적 비관측

| 항목 | 코드 근거 | 왜 안 보였나 | 제안 |
|---|---|---|---|
| piggyback / mimicry | `attack_variants.hosted_scale` 0.3–0.7× / 0.9–1.1× 숙주 step(0.37 s) → 0.05–0.15 s 주기 | 2 Hz 갱신 + 1 s 평균 아래 | **"NVML 대역 밖 실측 대조군"으로 재분류**. C 합성 보정의 실측 앵커로 사용 (§3.3) |
| LTMA | `ltma_inject.py` 30 s 위상 + step당 aux mm, 평균 유지 | 30 s 위상은 창(30 s)과 같아 창 안에서 평탄. aux는 step(0.39 s) 단위 | 순시 채널로 이미 보임. 위상 주기를 창보다 짧게 둔 변종 추가 |
| LLM serving / batch | 1–4 RPS 소형 모델, 62 W / 93 W | 요청 버스트가 보드 전력에 안 뜸 | 7B급 + 버스트형 부하 재캡처, 또는 "저부하 정상" 코호트로만 사용 |
| mixed tenants | 매트릭스에 없음 | `cross_job_sync` 정상 기준 부재 | 2 GPU 서로 다른 정상 잡 동시 캡처 추가 (4–6세션) |
| coordinated 정상 오탐 | 8/8이 타깃 ↔ **companion idle** 쌍 | idle 24 W 평탄과의 상관이 sync로 계산됨 | idle 게이트 GPU는 sync 계산에서 제외 (일부 `_noidle`로 진행 중) |

### 2.3 [C] NVML 물리 한계 (Bit2Watt 원문 기준)

| 원문 공격 | 원문 파라미터 | NVML에서 남는 것 |
|---|---|---|
| SWMA | 별도 커널 active/passive, CPU가 unified memory flag로 전환. **1.5–6 kHz**, 진폭 103–248 W (3090: 5 kHz, 204 W) | kHz 성분은 전부 소실. 남는 것: 평균 상승(duty×진폭), util 높음 대비 전력 낮음(decoupling), 클럭/온도의 저주파 반응 |
| LTMA | 학습 파이프라인 안에서 하이퍼파라미터 조작 / aux 연산 삽입. **1.2–3 kHz**, 진폭 더 큼, 주파수는 학습 동역학에 종속 | 동일하게 kHz 소실. step 단위 envelope만 남을 수 있음 |
| 규모 효과 | 1 MW, DER 90%, GPU 1,000장에서 THD 46.8%, 감쇠비 −0.27 | 2-GPU로는 관측 불가 |

원문 측정은 NI USB-6421 DAQ, **10 MSa/s**, +12 V 보조 전원선이다. 원문 Table 4는 NVML을 "450 Hz"로 적었다.
우리 실측 갱신은 약 2 Hz이므로 이 수치는 원문 확인 후 인용한다.

**원칙 (v13 §4.2.3과 동일하게 유지)**: kHz 원형 공격을 NVML로 "탐지했다"고 합성해서 주장하면 안 된다.
C 계열 합성의 목적은 두 가지다.
1. NVML 관측 연산자를 통과시킨 뒤 무엇이 남는지 정량화한다. 정답은 대부분 "판정 불가 / Mode B 필요"다.
2. 가상 Mode B 채널(고속 계측)을 함께 생성해, 외부 계측이 붙으면 판정이 어떻게 달라지는지 보인다. 이 채널에는 **합성 표기를 강제**한다.

### 2.4 [D] 파이프라인 로직 (이미 진단된 것 + 추가)

| 위치 | 증상 (실측) | 원인 |
|---|---|---|
| Stage1 | 1차 실행에서 idle 2533창 100% 후보 | 합성 코호트. **round3에서 수정됨** (실측 정상 refit, expected_band, α=0.05) |
| Stage1 | LTMA / piggyback 후보 0 | 평균 전력 채널만 사용. LTMA는 순시 채널 추가로 회복 가능 |
| Stage2 번들 | `strong_peak` 1/42, `period_mismatch` 1/42 | `tau_peak`≈5.69가 합성에 맞춰졌는데 실측 사각파의 로그 prominence는 3.6–4.5 |
| Stage2 번들 | `progress_log_available` 42/42 | 위장 로그가 항상 있어 변별력 0. 그런데 LLM은 이를 hidden_ml 근거로 사용 |
| corpus | SWMA 필수 = 꺼진 칸들 | 문서 신분증이 실측에서 켜지는 증거와 반대 |
| judge | 정상 오탐 17 중 12에 공격명 | `normal` / "Stage1 오탐" 선택지가 사실상 없음. normal은 RAG top-1이 benign일 때만 자동 |
| RAG | 유사도 0.07–0.18, 거의 동점. LLM이 88% 그대로 따름 | §5 Step 5 |

---

## 3. 관측할 수 없는 데이터의 합성 설계

### 3.1 공격 × 관측성 등급표 (이걸 먼저 확정)

| 등급 | 정의 | 해당 | 데이터 출처 | 평가 지표 |
|---|---|---|---|---|
| **O** (observed) | NVML 대역 안, 실측 성공 | SWMA ≥1 s(순시), crypto, LTMA(순시) | **실측** | recall / FP |
| **R** (repairable) | 대역 안이나 버그로 놓침 | SWMA 0.5–1 s 적체 세션 | 수리 후 **재캡처** (합성 대체 금지) | recall |
| **S** (scale) | 대역 안이나 2×4090으로 못 만듦 | >2 GPU 동기화, mixed tenants 다수, 대규모 coordinated | **실측 트레이스 조합 합성** | recall, 합성 표기 |
| **U** (unobservable) | NVML 대역 밖 | 원형 SWMA 1.5–6 kHz, LTMA 1.2–3 kHz, piggyback 실측 | **물리 파형 합성 → 관측 연산자** + 가상 Mode B | **abstain 정확도** (판정 불가를 맞게 냈나) |

### 3.2 관측 연산자 적합 (합성의 핵심, 실측으로 보정)

v13 §4.1.5의 y[k] = Q{(h*x)(kTs)} + ε를 **실측으로 적합**한다.

- 정답 x(t): SWMA 세션의 `attack_intervals`(실제 active 구간, 이미 기록됨) × 진폭.
- 관측 y: `power_w`와 복원한 `power_instant_w`.
- 적합 대상:
  - `h_avg`: 약 1 s boxcar 예상. §2.1 표의 sinc 감쇠와 일치하는지 검증한다.
  - `h_inst`: 순시 필드의 평활 폭.
  - `Ts`: 실효 갱신 주기 (≈500 ms, 위상 지터 포함).
  - ε: idle·finetune의 잔차 분포.
  - 상승/하강 비대칭 (전력 관리 지연).
- 산출: `dataset/eval/nvml_observation_operator.json` (GPU 모델별) + period_retention 곡선의 **실측 vs 연산자 예측** 비교표.
- 이 연산자를 모든 합성(S, U, 기존 v3 재생성)에 적용한다. 그러면 합성–실측 격차의 1차 원인(A-3)이 사라진다.

### 3.3 U 등급 합성 절차 (논문 의도 공격)

1. **물리 파형 x(t)** — 원문 파라미터로 생성한다. 10 kHz 이상 샘플.
   - SWMA: f ∈ [1.5, 6] kHz, 진폭 ∈ [103, 248] W, duty, 기저 전력. 공격 지속 on/off envelope는 시나리오 파라미터로 둔다.
   - LTMA: 실제 학습 step 트레이스(실측 finetune 순시 전력)에 f ∈ [1.2, 3] kHz 변조를 high-load 위상에서만 곱한다. 원문 "주파수는 학습 동역학에 종속"을 반영한다.
2. **NVML 뷰 y** = 연산자(x). util / clock / 온도는 실측 SWMA·crypto에서 적합한 회귀로 생성한다.
3. **실측 앵커로 검증**: piggyback·mimicry 실측(0.05–0.15 s, 이미 대역 밖)의 NVML 통계와
   같은 조건 합성의 통계를 KS 검정 등으로 비교한다. 맞아야 U 합성을 믿을 수 있다.
4. **가상 Mode B 채널**: x를 PDU/DAQ 대역(예: 1 kHz, 10 kHz)으로 다운샘플한 `modeb_power_w`.
   - 컬럼명과 매니페스트에 `synthetic_channel: true`를 넣는다.
   - Stage1/2 기본 경로에서는 사용 금지하고 `--mode b` 실험에서만 읽는다.
5. **라벨**: `gt_observability = U`. 평가는 recall이 아니라
   "NVML-only 경로가 `unobservable`/`판정 불가`를 냈는가"와 "Mode B 경로가 맞췄는가"를 따로 본다.

### 3.4 S 등급 합성 (규모)

- 실측 단일 GPU 트레이스(SWMA, DDP, finetune)를 N장으로 복제한다. 위상 오프셋, 참여율, 지터를 파라미터로 둔다.
- 합성 GPU를 만들지 않고 **실측 트레이스의 재조합**만 쓴다. 노이즈 모델은 실측이고, 연산자는 이미 적용된 상태다.
- v13 §4.2.4 원칙을 그대로 따른다. 대규모 동기화 성능은 "합성 규모 스윕"으로만 보고하고 실측 주장과 섞지 않는다.

---

## 4. 이미 진행 중인 것 (미커밋 diff, 참고)

- `log_period_mismatch` / `period_comparable` (Stage2 전용, 1 s 이상 step만 비교)
- pattern 카드(주기, 규칙성, swing p10/p90)를 번들에 실음
- `cross_job_sync_index_noidle`
- corpus 7건 수정 (SWMA 필수를 `log_period_mismatch`로 교체 등)
- judge 스키마에 `normal` 추가

아래 단계는 이 diff를 **먼저 커밋한 뒤** 그 위에 쌓는 것을 전제로 한다.

---

## 5. 단계별 수정안

### Step 0 — 계측 수리와 복원 (1–2일, 최우선)

| 작업 | 파일 | 완료 기준 |
|---|---|---|
| 0-1 `power_instant_w`를 `valueType` 분기로 읽기 (uiVal/ullVal/dVal) | `collect_telemetry.py` | 단위 테스트: mock union으로 mW → W |
| 0-2 기존 78세션 복원 스크립트. 원본 불변, `power_instant_w_recovered` 컬럼을 별도 산출 | `dataset/recover_instant_power.py` (신규) | 0 < 값 < 600 W, 평균 필드와 상관 > 0.9 |
| 0-3 SWMA active 루프 시간 예산화 (Event 동기화) | `swma_workload.py` | 0.5 / 1 s 칸에서 `attack_intervals` 실제 duty가 요청 ±10% |
| 0-4 `value_changed`를 필드별로 기록 (평균 / 순시 따로) | `collect_telemetry.py` | |
| 0-5 R 등급 재캡처: SWMA 0.5 / 1 s 칸만 (8–10세션, 약 2 h) | `capture_matrix.yaml` | |

### Step 1 — 관측 연산자 적합 + Stage 0 결선 (2–3일)

- §3.2 적합 스크립트 `dataset/fit_observation_operator.py`를 작성한다.
- 창마다 `observable_band_hz`(실효 갱신에서 계산)를 넣고, 관측 품질이 낮으면 `observability: limited`를 붙인다.
- 새 증거 `beyond_observable_band`: 창에 의심은 있으나 대역 밖이면 켠다. 이것이 **"판정 불가(Mode B 필요)"** 출력의 근거다.
- v13 G2 "부분 달성 → 달성"으로 갱신할 수 있는 항목이다.

### Step 2 — 합성 v4 생성 (3–4일)

- 모든 합성기를 연산자 통과로 통일한다 (`nvml_avg_window_s=0` 기본값 제거).
- 등급별 생성기: O/R은 생성하지 않음(실측 사용), S는 트레이스 재조합, U는 §3.3.
- `gt_observability` 컬럼, split manifest, holdout(U 공격은 일부 파라미터 군집을 holdout).
- 산출: `all_v4.csv`, `synth_vs_real_gap_v4.md`. 같은 조건의 실측 vs 합성 피처 분포 비교로 격차가 줄었음을 보인다.

### Step 3 — Stage1 순시 채널 (2일)

- 피처 추가: `inst_swing`, `inst_over_avg_swing`(순시/평균 비), `inst_band_0p3_1hz`, `inst_regularity`.
- 실측 정상으로 코호트 refit (round4). α 규칙은 round3 그대로 (cal만, ≤0.10 가중 FP).
- 기대: LTMA 후보 0 → 회복. 단 FSDP / dataloader / HPO의 순시 swing이 커서 FP 증가 위험이 있으므로, 코호트별 FP를 표로 반드시 보고한다.

### Step 4 — Stage2 증거 재설계 (2–3일)

- bool 임계 대신 **등급 증거**(none / weak / strong)를 쓴다. 임계는 실측 train/cal로 재적합한다 (`tau_peak` 등).
- `square_wave_pattern` 증거 신설: 규칙성 ≥ r, swing ≥ s, 주기가 관측 대역 안. pattern 카드 수치에서 계산한다.
  SWMA 필수를 이것으로 교체한다. 사람 눈의 "사각파"와 기계 판정을 같은 척도로 맞추는 것이다.
- `progress_log_available`은 번들에서 제외하거나 evidence weight 0 + LLM에 "변별력 없음" 명시.
- sync 계산에서 idle 게이트 GPU를 제외한다 (coordinated 정상 오탐 8/8 해소).
- `inst_util_decoupled`(순시 전력 대비 util 괴리)를 LTMA 필수로 둔다.

### Step 5 — RAG 재설계: 확장 지식베이스 + 하이브리드 검색 (4–6일)

**전제 (2026-10-02 결정)**: RAG는 유지한다. 지금은 문서가 10건이라 전부 넣어도 되지만,
실제 데이터센터 배포에서는 ATT&CK, ATLAS, 논문, 운영 문서, 사내 런북이 수백~수천 청크로 늘어난다.
이 규모에서 "LLM이 맞는 문서를 받게 하는 계층"이 필요하고, 그것이 공모전의 신기술 항목이기도 하다.
목표는 **문서가 늘어날수록 오히려 RAG가 일을 하는 구조**다.

#### 5.0 지금 유사도가 낮은 이유 (코드 근거)

| # | 원인 | 위치 |
|---|---|---|
| 1 | **비대칭 텍스트**: 쿼리는 `features_to_description()`이 만든 짧은 5문장 템플릿, 문서는 출처·개요·MITRE가 섞인 긴 본문 | `features.py:103`, `corpus/*.md` |
| 2 | char 2–4gram TF-IDF에 **문서 10건**: IDF가 변별력을 못 만들고 "전력·변동·주기" 같은 공용어만 겹침 → 0.07–0.18 동점 | `rag_analyzer.py:150` |
| 3 | **검색 표현 ≠ 판정 표현**: 판정은 증거 bool + pattern, 검색은 별도 자연어. 둘이 다른 문서를 가리킴 | `run_pipeline.py:568-570` |
| 4 | sbert 기본 `all-MiniLM-L6-v2`는 **영어 전용** | `rag_analyzer.py:95` |
| 5 | soft range 페널티(0.1×위반 수)가 원 점수 폭과 같은 크기 + 합성 train 수치 | `rag_analyzer.py:236` |
| 6 | 문서 단위 인덱싱(청크 없음): 긴 문서에서 관련 문단 하나가 전체 평균에 묻힘 | `_load_corpus` |

핵심: **유사도 숫자를 올리는 게 목적이 아니라, 맞는 문서와 틀린 문서의 점수 차(margin)가 벌어지게 하는 것**이다.
아래 설계는 쿼리와 문서를 같은 형식으로 맞추고, 보정된 reranker 점수를 쓴다.
그래서 관련 문서는 높게, 무관 문서는 낮게 나오는 것이 정상적으로 기대된다.

#### 5.1 지식베이스 확장 (무엇을 넣나)

| 계층 | 출처 | 규모(예상) | 역할 |
|---|---|---|---|
| **L1 위협 카드** | 우리 `corpus/*.md` 10건 (+ 확장) | 10–30 | 판정 단위. 증거 체크리스트 보유 |
| **L2 공격 기법** | MITRE ATT&CK Enterprise (T1496 Resource Hijacking, T1496.001 Compute Hijacking 등), ATT&CK for ICS (T0831 Manipulation of Control, T0879 Damage to Property 등), **MITRE ATLAS** (AML.T0034 Cost Harvesting, AML.T0029 Denial of ML Service 등). 공식 STIX/YAML 데이터에서 자동 변환 | 전체 1,000+ 기법 (필터 없이 넣어 잡음 견딤을 보임) | 공격 일반 지식, 완화책, 탐지 데이터 컴포넌트 |
| **L3 연구 근거** | Bit2Watt, LAA, BlackIoT, Chaudhary(공진 기준), Choukse(AI 전력 안정화), NERC 사고 리뷰 | 논문 10–20편 → 섹션 청크 | 물리 메커니즘, 수치 근거 인용 |
| **L4 관측성** | NVIDIA NVML/DCGM 필드 문서, 우리 관측 연산자 결과(§3.2) | 수십 청크 | "이 대역은 못 본다"의 근거 |
| **L5 운영** | 대응 런북(조사 절차, 스케줄러 조치), 정상 워크로드 프로파일 | 수십 | recommended_action 근거 |

모든 문서를 **공통 스키마**로 변환한다:
`doc_id, layer, source, url, license, lang, title, section, text, attack_ids[], threat_ids[], observability_tier, updated`.
ATT&CK/ATLAS는 공식 저장소(attack-stix-data, atlas-data)의 버전 태그를 고정해 재현성을 확보한다.

#### 5.2 인덱싱

- **섹션 단위 청크** (200–400 토큰, 제목 경로 접두: `SWMA > 관측 시그니처 > ...`).
- **L1 카드에 `## 관측 시그니처` 섹션 신설**: Stage2 증거·pattern 카드와 **같은 통제 어휘**로 쓴다 (예: "전력 사각파, 주기 2–4 s, swing 300 W 이상, 규칙성 0.9 이상, 로그 step과 비교 불가"). 쿼리 렌더러(5.3)와 같은 문장 틀을 쓰므로 **대칭 검색**이 된다.
- 임베딩: **다국어 모델**. 한국어 카드와 영어 ATT&CK을 교차 검색해야 한다.
  - 1안 `BAAI/bge-m3` (dense + sparse + multi-vector 한 모델, 오프라인 가능, 약 2.2 GB)
  - 2안 `intfloat/multilingual-e5-large`
- 저장: FAISS (이미 설치됨) + sparse 인덱스. 인덱스에 KB 해시·모델 버전을 기록한다.
- **문서 추가 = 해당 문서만 임베딩 후 append**. 재학습 0회라는 v13 주장(§4.1.6)을 그대로 유지·시연한다.

#### 5.3 쿼리 생성 (증거 → 질의)

- **결정적 쿼리 렌더러**: evidence bundle + pattern 카드 → "관측 시그니처" 문장 틀로 변환한다. 문서 쪽과 같은 어휘다.
  LLM을 쓰지 않으므로 같은 입력에 같은 쿼리가 나와 결정성 1.0을 유지한다.
- **다중 쿼리**: ① 시그니처 질의 (L1 대상), ② 메커니즘 질의 ("periodic GPU load modulation affecting grid oscillation", L2·L3 대상), ③ 관측성 질의 (`beyond_observable_band`일 때 L4 대상).
- 선언 컨텍스트(`declared_*`)는 쿼리에 넣지 않는다 (v13 §4.4.3: 넣으면 0.092로 하락, 거짓일 수 있음).

#### 5.4 검색 파이프라인

```
쿼리(1–3개)
  ├─ sparse (BM25 / bge-m3 sparse) ─┐
  ├─ dense  (bge-m3)               ─┼─ RRF 융합 → top-50
  └─ 메타데이터 필터 (layer, observability_tier)
                                    ↓
                     cross-encoder reranker (bge-reranker-v2-m3) → 0–1 보정 점수
                                    ↓
           L1 카드 순위 = 0.5·reranker + 0.5·증거 체크리스트 점수
                                    ↓
           카드에 링크된 attack_ids로 L2·L3 청크 확장 (지식 그래프 1-hop)
```

- **증거 체크리스트 점수**: required 충족률, exclusion 위반 시 탈락, supporting 가산. 0–1로 정규화한다.
  판정 근거와 검색 근거를 묶는 장치다.
- **reranker 점수를 "유사도"로 보고**한다. cosine보다 해석 가능하고 cal split에서 임계 보정이 가능하다.
- **unknown 기각**: 최고 L1 점수 < τ_unknown이면 Unknown. τ는 cal에서 적합한다 (기존 0.28 TF-IDF 임계 대체).
- 1-hop 확장 덕분에 LLM은 "카드 체크리스트 + ATT&CK 기법 설명·완화책 + 논문 근거 문단"을 함께 받는다.
  ATT&CK을 넣었을 때 판정 문장에 기법 ID와 완화책이 인용되는 것이 보이는 개선점이다.

#### 5.5 LLM 컨텍스트 구성

- 토큰 예산(gemma3:12b `num_ctx` 16k)에 맞춰 넣는다: L1 후보 카드 전부(체크리스트만), L2·L3 청크 상위 N개(중복 제거).
- 각 청크에 인용 ID `[doc_id#chunk]`를 붙이고, 출력 `explanation`의 인용이 실제 제공 청크인지 스키마 검증한다.
  없는 인용은 fallback 처리한다.
- 판정권은 지금처럼 체크리스트·게이트에 있다. RAG는 **후보 순위와 근거 공급**만 한다 (v13 §4.1.3 원칙 유지).

#### 5.6 평가 (공모전 보고용)

- **라벨된 질의 세트**: 실측 test 창 + 합성 v4 창 → 정답 카드, 정답 ATT&CK ID. 정답은 사람이 승인한 매핑표로만 만든다.
- 지표:
  - L1 Recall@1/@3, MRR, nDCG@5
  - **점수 분리도**: 관련 vs 무관 문서 점수 AUC, 평균 margin
  - Unknown 기각률 (미등록 3건 + 신규 holdout)
  - L2 인용 정확도: 판정 문장에 인용된 ATT&CK ID가 정답 매핑과 일치하는 비율
- **Ablation 표**:

| 구성 | 기대 |
|---|---|
| TF-IDF 문서 단위 (현재) | 기준선, 0.07–0.18 동점 |
| + 청크 + 대칭 시그니처 섹션 | 분리도 상승 |
| + bge-m3 dense / hybrid RRF | 교차언어(ATT&CK) 검색 가능 |
| + reranker | 보정 점수, margin 확대 |
| + 증거 체크리스트 융합 | 판정–검색 일치 |
| KB 10건 vs 1,000+건 | **문서가 늘어도 Recall@3 유지 = 확장성 실증** |

- 누설 통제: 쿼리에 gt 없음. 카드의 수치 범위·임계는 train/cal 산출물만 쓴다. ATT&CK은 공개 자료라 누설 대상이 아니다.

#### 5.7 구현 순서

1. KB 수집기 `pipeline/kb/ingest_attack.py` (STIX/ATLAS → 공통 스키마 JSONL)와 논문 섹션 청커
2. L1 카드 `## 관측 시그니처` 섹션 작성 → 사람 승인 게이트 (기존 corpus 승인 절차와 동일)
3. `pipeline/kb/index.py` (bge-m3 + FAISS + sparse, 증분 추가)
4. 쿼리 렌더러 + 하이브리드 + reranker + 체크리스트 융합 → `SignatureRetriever` 교체 (인터페이스 유지, `--rag-backend hybrid`)
5. 1-hop 확장과 인용 검증을 `stage2_llm.judge`에 연결
6. 평가 스크립트 `pipeline/eval_retrieval_v2.py` + ablation 표

필요 자원: bge-m3 + reranker 모델 다운로드 약 4.5 GB (디스크 여유 150 GB). 추론은 CPU로도 가능하다.
**모델 다운로드는 승인 후 진행한다.**

### Step 6 — judge 출력 공간 정리 (1일)

- 선택지: `known | partial | unknown | normal(stage1_fp) | unobservable`.
- 프롬프트에 "Stage1 후보여도 정상일 수 있다"와 "대역 밖 증거면 unobservable"을 명시한다.
- 데이터셋에 GT가 없는 문서(llmjacking, burst, hidden_ml, sponge)는 `eval_scope: out_of_dataset`으로 표시한다. 오답 집계를 따로 낸다.

### Step 7 — 평가와 소개서 반영 (2일)

- 지표를 **관측성 등급별**로 분리한다. O/R은 recall·FP, S는 합성 표기, U는 abstain 정확도와 Mode B 대비 향상폭.
- v13 수정 포인트:
  - §4.2.3 표에 "실측 78세션 + 순시 채널 복원"과 등급표를 추가한다.
  - §4.1.5 관측 연산자를 "실측 적합값(boxcar ≈1 s, 갱신 ≈500 ms)"으로 바꾼다. 서술이 아니라 측정값이 된다.
  - §4.4.1 실측 열을 추가한다 (round3: 정상 후보율 0.062, detectable recall 0.657).
  - §4.4.3 RAG 수치를 갱신한다.
  - α 표기(합성 0.10 / 실측 0.05)를 구분한다. pytest 수를 갱신한다 (현재 207).
  - 신규 서사: "버그로 잃었던 순시 채널을 복원해 NVML 관측 한계를 1 Hz까지 실측으로 확정"은 v13 §3.2 (2) 관측 경계 주장의 **실증 근거**가 된다.

---

## 6. 일정 (대회 일정 기준)

| 기간 | 작업 | 마감 근거 |
|---|---|---|
| 10/2–10/4 | 미커밋 diff 커밋, Step 0 (복원 + 수리), Step 1 연산자 적합 | 근거 수치 확보 |
| 10/5–10/9 | Step 3 (Stage1 순시), Step 5 (RAG), Step 4 일부 | |
| 10/10–10/11 | 소개서 v14 수치 반영, 기능 동결 | **10/11 접수** |
| 10/12–10/20 | Step 2 합성 v4 (S/U), Step 6, R 재캡처 | |
| 10/21–22 | | **예선** |
| ~11/10 | Step 7 평가, 콘솔 연결 | **11/11 최종** |

접수 전에는 **Step 0·1·3·5가 가장 효과가 크다.** 재수집 없이 기존 데이터로 실측 수치를 바꿀 수 있기 때문이다.
U 합성은 접수 후에 해도 서술에 영향이 적다.

---

## 7. 결정 현황 (2026-10-02 갱신)

| # | 항목 | 상태 |
|---|---|---|
| 1 | kHz SWMA를 4090에서 실행 | **미결**. 실행해도 NVML이 kHz를 보게 되지는 않는다. 목적은 "kHz 공격이 돌 때 NVML에 남는 저주파 흔적" 기준 데이터 확보뿐이다. 권장: 실행하지 않고 piggyback 실측을 기준으로 사용 |
| 2 | 순시 전력의 사용 위치 | **진행 (2026-10-03)**. Stage1 증거 `inst_power_unsteady`로 투입(round5b). fold_00 LTMA 0%→38%, serving·ResNet 오탐 불변. 점수 피처로 직접 넣으면(round4) serving 오탐 2.5%→66%라 기각. LTMA가 fold_00에만 있어 6세션 추가 캡처 중(`scripts/queues/ltma_v2.txt`) |
| 3 | LLM serving 재캡처 | **완료** (6/6 clean, 중앙값 140–190 W). serve_1b(1.3B, fp16) + bursty Poisson 트래픽 6세션, `scripts/capture_serving_v2.sh`. 로그 `dataset/real/logs/capture_serving_v2.log` |
| 4 | 가상 Mode B 채널 | **보류** (나중에 결정) |
| 5 | RAG 유지·확장 | **결정: 유지**. ATT&CK·ATLAS·논문·관측성·런북으로 KB 확장, Step 5 참조. bge-m3·reranker 다운로드 완료(CPU 동작 확인) |

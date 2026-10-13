# v5 계획 — 정확도 60%의 원인 진단과 open-set(처음 보는 공격) 판정·연구 개선

기존 v4 코드는 그대로 두고 `pipeline/v5/`에 별도로 구현한다.

## 1. 진단 (pipeline/v5/diag_information.py → dataset/eval/v5/diag_information.md)

### 1.1 "60%"의 구성
| 평가 | 세그먼트 | gemma3:12b A5 정확도 |
|---|---|---|
| dev (튜닝에 쓴 클래스) | 107 | 0.720–0.757 |
| 처음 보는 클래스·정상 (pristine) | 23 | 0.000–0.043 |
| **합산** | 130 | **≈ 0.60** |
| e2e (Stage1 후보 창만, 세션 게이트 없음) | 103 | 0.359 |

60%는 "잘 되는 dev 0.75 + 완전히 실패하는 처음 보는 클래스 0.0"의 평균이다.

### 1.2 데이터 한계인가? — 아니다 (대부분)
같은 130 세그먼트에서 라벨을 쓴 세션 단위 교차검증(LOSO) 분류기 상한:

| 특징 | 정확도 |
|---|---|
| **LLM이 받는 증거 번들 그대로** | **0.946** |
| 원시 전력+순시 전력 | 0.907 |
| + 사용률 + 메모리 + 프로세스 수 | 0.958 |

LLM 입력에 이미 클래스를 가를 정보가 있다. hidden_ml→크립토, sponge→SWMA, llmjacking→정상 오판은
**정보 부족이 아니라 판정 구조(텍스트 카드 + LLM 추론)의 한계**다. 카드 문장("평탄한 고부하")은
여러 클래스에 똑같이 맞고, LLM은 수치 경계를 학습한 적이 없다.

### 1.3 데이터 한계가 맞는 부분 — 처음 보는 공격의 "새로움"
학습에서 한 클래스를 빼고 kNN 거리로 새로움을 재면(Sun et al., ICML 2022 방식):

| 제거 클래스 | AUROC (원시 특징) | 해석 |
|---|---|---|
| burst_ramp, ltma, swma | 0.91–1.00 | 처음 봐도 "이상하다"는 잡힌다 |
| sponge, hidden_ml | 0.84–0.85 | 대체로 잡힌다 |
| **cryptojacking, llmjacking** | **0.49–0.57** | 정상(평탄 사전학습, 서빙)과 같은 분포. 텔레메트리만으로는 처음 보는 경우 구분 불가 |

크립토·LLMjacking은 **신고(declared) 맥락·프로세스(Stage3)** 없이는 원리상 처음 보는 공격으로 잡을 수 없다.
이는 v4 결론과 같으며, v5는 이를 정직하게 `unknown`이 아니라 신고 불일치 증거로만 다룬다.

### 1.4 파이프라인 결함 (코드에서 확인)
1. **닫힌 목록 과신**: LLM은 정답 카드가 없어도 가장 비슷한 카드를 고른다. unknown 발동 0 (RESULTS_v4 §5.3).
2. **카드 규칙 불일치**: requirement gate는 카드 required 증거가 실측과 맞지 않아 정상 판정까지 깎는다 (§6).
3. **번들 채널 누락**: 카드는 "메모리 높게 고정, 사용률만 출렁임"을 말하지만 번들에는 메모리·사용률 통계가 없다.
4. **과도 구간 착시**: e2e 정상 학습 오판은 창 1–2개짜리 시작/종료 구간. 최저 FFT 빈(0.039 Hz)이
   `strong_peak`·주기성 0.95로 렌더링돼 "주기적 사각파"로 읽힌다. (운영에서는 세션 게이트 5창이 막는다.)
5. **연구 루프 검색어**: 전력 슬롯만으로 키워드를 만들어 메커니즘(API 도용, 에너지-지연 공격)을 특정하지 못한다.

## 2. 참고 문헌·코드
- MalRAG (arXiv 2511.14129): 라벨 달린 **사례(트래픽 샘플)를 검색**해 LLM에 주고, 클래스별 거리 임계
  τ = d̄ + α·σ로 증거를 걸러 "어떤 사례와도 맞지 않으면 novel". → 사례 기억 + 클래스별 일치도.
- Deep kNN OOD (Sun et al., ICML 2022, github.com/deeplearning-wisc/knn-ood): 분포 가정 없는 kNN 거리 새로움 점수.
- Conformal prediction for LLM MCQA (Kumar et al., 2023, arXiv 2305.18404): 보정 집합으로 오류율을 보장하는 기권.
- CRAG (Yan et al., arXiv 2401.15884, github.com/HuskyInSalt/CRAG): 검색 평가기가 Correct/Ambiguous/Incorrect를
  매기고 부족하면 외부 검색으로 확장, 문서는 분해-재조합.
- Detecting Hidden ML Training With Zero-Overhead Telemetry (arXiv 2606.19262): NVML 9개 카운터,
  메모리 plateau·전력-메모리 상관·자기상관 특징으로 은닉 학습 98% (보지 못한 회피형 43–87%).
- Structured risk indicators로 zero-shot 탐지 (arXiv 2608.16508): LLM1이 구조화 지표, LLM2가 지표만 보고 판정.
- GPU 크립토재킹 탐지 (Pott et al. 2023; arXiv 2408.14554): GPU 부하 + 그래픽 메모리 조합이 핵심.

## 3. v5 설계

```
증거 번들(v4) ─┬─ 확장 관측 채널: 메모리·사용률·프로세스 (정상 기준 분위수 문장)
               ├─ 사례 기억(exemplar KB): 승인된 라벨 세션의 30 s 창 특징
               │     · 새로움 p (kNN 거리, 세션 단위 conformal)
               │     · 클래스별 일치 p_c (그 클래스 사례 거리의 conformal p)
               └─ 텍스트 카드 검색(v4 그대로)
                         ↓
LLM 판정 (사례 일치도 + 카드 + 확장 채널을 함께 봄)
                         ↓
Open-set 게이트 (결정적, α 고정 0.05, 데이터로 튜닝하지 않음)
  · attack:c 인데 c의 사례 일치 p_c < α  → unknown (카드와 사례 불일치)
  · normal 인데 새로움 p < α 이고 정상 일치 p_normal < α → unknown (설명되지 않는 이상)
  · 사례가 없는 카드(c에 승인 세션 없음) → attack 유지, "사례 근거 없음" 표시
                         ↓ unknown
연구 v5 (CRAG식)
  1. 대조 서술: 가장 가까운 기존 클래스·정상 대비 크게 벗어난 특징(메모리·사용률 포함)을 영어 관측 문장으로
  2. 가설 생성(HyDE식): LLM이 메커니즘 가설 3개 + 각 가설이 예측하는 관측 특징
  3. 검색: 가설별 arXiv/Semantic Scholar + 로컬 ATT&CK/ATLAS
  4. 평가기: 초록이 관측 특징을 예측하는가 (Correct/Ambiguous/Incorrect); 없으면 검색어 다듬어 1회 재시도
  5. 반증 검사: 가설의 예측 특징과 실제 관측을 결정적으로 대조 → 일치하는 가설만 DRAFT, 분석가 대기열
```

## 4. 평가 (누설 방지)
- 사례 기억 = round5b fold_00 **train+cal** 세션만. 평가 = 그 외 (test, holdout, misuse).
- P1 closed: 같은 세그먼트에서 v4 A5 결과(기존 jsonl)와 v5 비교.
- P2 open-set (leave-one-class-out): 클래스 c의 카드·사례를 모두 빼고 c 세그먼트가 `unknown`이 되는지,
  P1의 알려진 세그먼트가 잘못 `unknown`이 되는 비율.
- P3 연구: P2에서 unknown이 된 클래스에 대해 v4 연구 루프 vs v5 연구 루프가 올바른 메커니즘 문헌을 찾는지.
- α는 0.05로 사전 고정. test로 고르지 않는다.

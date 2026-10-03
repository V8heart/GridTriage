# GridPulse v4 구현 계획: RAG 판정 파이프라인 · 검증 · UI · Stage1 대안

작성: 2026-10-03 · 상위 문서: [plan_real_v3.md](plan_real_v3.md) · 상태: **진행 중** (각 단계 끝에 결과를 이 문서에 갱신)

---

## 0. 목적 (길 잃지 않기 위한 기준)

사용자 지시(2026-10-03):

1. RAG는 **여러 더미 문서 사이에서 맞는 문서를 잘 꺼내야** 하고, **그 검색 덕분에 판정이 좋아짐**을 보여야 한다.
2. RAG는 "이 공격일 확률 n%"를 내지 않는다. **유사도 높은 문서를 LLM에 넘기고, LLM이 최종 판정**을 낸다.
3. 목적에 맞는지 계속 검증하며 개선한다.
4. 합성 데이터는 **필요성을 검증한 뒤** 만들지 말지 결정한다.
5. 파이프라인과 연동되는 실시간 UI `app.py`.
6. 완료 후 LLM 2종(서버의 `gemma3:12b`, `deepseek-r1:14b`)과 RAG 구성 ablation.
7. Stage1의 구조적 개선 방식이 있으면 시도하고 결과를 정리한다.

### 0.1 매 단계 점검할 목적 체크리스트

| # | 질문 | 근거 문서 |
|---|---|---|
| P1 | 출력이 "차단·확정"이 아니라 **조사 우선순위 + 근거**인가 (triage) | v13 §2.2 (1), §7 |
| P2 | NVML로 못 보는 것을 본 척하지 않는가 (관측 경계 선언) | v13 §2.2 (2), §4.2.3 |
| P3 | 신규 위협을 **문서 추가만으로** 반영하는가 (재학습 0) | v13 §2.2 (3) |
| P4 | 판정 근거가 증거 이름·인용 문서로 역추적되는가 | v13 G3 |
| P5 | 평가에 test/holdout 누설이 없는가 (test로 조정 금지) | v13 §4.2.5 |
| P6 | 수치는 실측 기준이며, 합성은 별도 표기인가 | v13 §7 |

**v13과 달라지는 점 (명시)**: v13 §4.1.3은 "LLM은 최종 확정 권한을 갖지 않는다"였다.
이번에는 사용자 지시로 **공격 유형·정상 여부의 최종 판정을 LLM이 낸다.**
다만 P1은 유지한다. 판정은 조사 우선순위 출력이지 차단 명령이 아니다.
스키마 검증과 인용 검증에 실패하면 판정을 무효로 표시하고, 근거 없는 판정은 내보내지 않는다.

---

## 1. 조사 요약 (설계 근거)

| 근거 | 핵심 | 우리 설계에 반영 |
|---|---|---|
| TechniqueRAG (arXiv 2505.11988) | BM25로 K=40 후보 → LLM 재정렬 → 후보 집합 안에서만 생성. 재정렬이 최종 F1을 올림 (Expert 데이터셋 51.1 vs RankGPT 41.5) | 후보 집합 제한, 재정렬 단계, "검색 → 판정" 분리 평가 |
| Hierarchical RAG for ATT&CK (arXiv 2604.14166) | ATT&CK을 평면 인덱싱하면 계층 구조를 버린다는 지적 | 카드 ↔ ATT&CK ID 링크로 1-hop 확장 (계층 활용) |
| RAG for Security Incident Analysis (arXiv 2603.18196) | 원시 로그 대신 **집계 요약 청크**, top-k=7. no-RAG는 탐지 0% | 원시 시계열 대신 증거 번들·pattern 카드로 질의. no-RAG ablation 필수 |
| BM25→Corrective RAG 벤치마크 (arXiv 2604.01733), 하이브리드 사례들 | 하이브리드 RRF가 단일 방식보다 우수. cross-encoder 재정렬이 가장 큰 단일 개선 (MRR@3 +17.2 pp, Recall@5 0.695→0.816) | BM25 + bge-m3 dense → RRF → bge-reranker-v2-m3 |
| 이전 시험 (2026-10-03) | bge-m3: 정답 0.81 / 무관 0.41–0.49. reranker: 0.92 / ≤0.006 | 모델 선택 확정 (CPU 동작 확인) |
| Conformal martingale (Vovk; arXiv 2609.27179) | 창 단위 conformal p를 시간 누적하면 **오탐률을 통제하는 순차 변화 탐지** 가능 (conformal CUSUM) | Stage1 대안 S-B (§6) |
| ADBench (arXiv 2206.09426), 하드웨어 텔레메트리 이상탐지 (arXiv 2510.26008) | IForest·kNN·PCA-Mahalanobis가 가볍고 강한 기준선 | Stage1 대안 S-A (§6) |

---

## 2. 지식베이스 (KB)

| 계층 | 출처 | 규모 | 역할 |
|---|---|---|---|
| L1 위협 카드 | `corpus/*.md` 10건 + `## 관측 시그니처` 섹션 (**DRAFT, 사람 승인 필요**) | 10 | **판정 단위**. LLM이 고르는 대상 |
| L2 ATT&CK Enterprise | STIX 19.2 attack-pattern | 697 | 공격 일반 지식 + **더미(방해) 문서** |
| L2 ATT&CK ICS | STIX 19.2 | 97 | 〃 |
| L2 MITRE ATLAS | 5.6.0 techniques + case studies | 227 | AI 시스템 공격 지식 + 더미 |
| L3 관측성·런북 | NVML 관측 연산자 결과, 조사 절차 | 소수 | `unobservable` 판정 근거 |

- 공통 스키마: `doc_id, layer, source, title, text, attack_ids, card, lang, version`. 산출 `dataset/kb/kb_docs.jsonl`.
- **청크는 문서 단위 + 섹션 단위**로 한다. 카드는 섹션별, ATT&CK은 설명 1–2문단.
- L1 카드 front-matter에 `attack_ids`를 추가한다 (예: cryptojacking → T1496, T1496.001). 1-hop 확장에 쓴다.
- 원본 데이터는 `dataset/kb/raw/`(git 제외)에 두고, 버전 태그(ATT&CK 19.2, ATLAS 5.6.0)를 고정한다.

## 3. 검색 → LLM 판정 파이프라인

```
Stage1 후보 세그먼트
   │ 증거 번들 + pattern 카드 (원시 시계열 없음)
   ▼
[Q] 결정적 쿼리 렌더러 ─ 관측 시그니처 문장 (카드 섹션과 같은 어휘)
   ▼
[R1] BM25 (top-50) ─┐
[R2] bge-m3 dense ──┴─ RRF 융합 → top-50
   ▼
[R3] bge-reranker-v2-m3 (cross-encoder) → 재정렬 점수
   ▼
[C] 컨텍스트 구성: 재정렬 상위 k 문서 (카드 + ATT&CK/ATLAS 섞임), 인용 ID [doc_id]
   ▼
[J] LLM 최종 판정 (JSON 스키마)
    verdict: attack | normal | unobservable | unknown
    threat_card: 제공된 L1 카드 이름 중 하나 또는 null
    attack_ids: 제공된 문서의 ID만
    cited_docs: 제공된 doc_id만      ← 인용 검증
    evidence_used: 번들에 있는 증거 이름만
    explanation
   ▼
[V] 검증: 스키마 · 인용 실존 · 증거 실존. 실패하면 invalid로 표시하고 1회 재시도
```

- 확률을 출력하지 않는다. 재정렬 점수는 **문서 순위용**이며 판정 신뢰도로 쓰지 않는다.
- LLM은 **제공된 카드 중에서만** 고르거나 normal/unknown/unobservable을 낸다 (TechniqueRAG의 후보 제약).
- 선언 컨텍스트(`declared_*`)는 판정 입력에는 넣되 **쿼리에는 넣지 않는다** (v13 §4.4.3: 넣으면 검색이 오히려 하락).

## 4. 평가 설계 (P5 준수)

### 4.1 평가 집합
- **세그먼트 단위**: 실측 세션의 타깃 GPU마다 GT 구간 창들을 묶어 하나의 세그먼트로 만든다 (Stage2의 실제 입력 단위).
- 실측 90세션 = 기존 78 + LTMA 6 + serving 6 (새 12세션은 holdout).
- GT 매핑 (사람이 정한 표, 평가 전 고정):

| gt_variant | 정답 카드 | 허용 형제 | 관측성 |
|---|---|---|---|
| swma_basic/jitter/shallow | swma | — | O |
| swma_coordinated | swma | coordinated_multi_gpu | O |
| swma_piggyback / mimicry | (정답: normal 또는 unobservable) | swma | **U** (실측에 흔적 없음) |
| ltma_basic | ltma | — | O(순시) |
| crypto_flat | cryptojacking | — | O |
| normal_* | normal | benign_periodic, normal_workloads | — |

- 두 가지로 보고한다. ① **Stage2 단독**: 모든 세그먼트. ② **종단(end-to-end)**: Stage1 후보 세그먼트만.

### 4.2 지표
- 검색: 정답 카드의 Recall@1/3/5, MRR (KB 1,000+ 문서 중). "더미 사이에서 잘 꺼내는가"를 직접 잰다.
- 판정: 정상/공격 정확도, 공격 유형 정확도, macro-F1, 정상 오탐률, invalid 비율, 호출당 지연.

### 4.3 Ablation 행렬 (6절의 "검색 덕분에 판정이 좋아짐" 증명)

| ID | 검색 | LLM에 주는 문서 |
|---|---|---|
| A0 | 없음 | 카드 이름 목록만 (내용 없음) |
| A1 | 기존 TF-IDF (카드 10건만) | top-3 카드 본문 |
| A2 | 기존 TF-IDF를 **KB 1,000+ 전체**에 적용 | top-k |
| A3 | dense만 (bge-m3) | top-k |
| A4 | 하이브리드 (BM25 + dense, RRF) | top-k |
| **A5** | **하이브리드 + reranker** | top-k |
| A6 | A5 + 무작위 문서 k개로 교체 (위약 대조) | 무작위 k |

× LLM 2종 (gemma3:12b, deepseek-r1:14b). 핵심 비교는 A5 vs A0/A6 (검색 효과)와 A5 vs A2 (검색 품질 효과)이다.

## 5. 합성 데이터 필요성 판단 (결정 기준을 먼저 고정)

다음 중 하나라도 해당하면 만든다. 아니면 만들지 않는다.

| 기준 | 판단 방법 |
|---|---|
| C1 | 평가에 필요한 클래스가 실측으로 **원리적으로** 얻을 수 없다 | 클래스별 관측성 등급 (plan_real_v3 §3.1) |
| C2 | 그 클래스가 소개서 주장이나 판정 경로(예: `unobservable`)를 검증하는 데 필요하다 | 주장 목록 대조 |
| C3 | 합성을 실측 앵커로 검증할 수 있다 (예: piggyback 실측과 통계 비교) | KS 등 |

- C1을 만족하지 않는 클래스(재캡처로 얻을 수 있는 것)는 **재캡처**한다.
- 결정 결과는 §8에 기록한다.

## 6. Stage1 구조 대안

현 구조의 약점(실측 확인):
- ① 대표 정상 세션이 없는 워크로드는 100% 오탐 (새 serving).
- ② 피처별 tail-min이 1/(n+1)에서 포화.
- ③ 창마다 독립 판정이라 지속형 공격의 시간 누적 정보를 버림.

| ID | 방식 | 겨냥하는 약점 |
|---|---|---|
| S-0 | 현 round5b (코호트 robust-z, tail-min, conformal) | 기준 |
| S-A | 코호트별 **kNN 거리 / IsolationForest** 점수 → 같은 Mondrian conformal | ② 다변량 결합, 포화 완화 |
| S-B | 창 p값의 **conformal CUSUM (martingale)** 으로 세그먼트 단위 경보 | ③ 지속성 활용, 오탐률 통제 유지 |
| S-C | 정상 세션 추가 시 기준선 **증분 갱신** (새 serving 6세션 편입) | ① 정상 대표성 |

- 같은 split·같은 α 규칙(cal만)으로 비교하고, holdout 12세션을 최종 확인용으로 쓴다.

## 7. UI `app.py` (Streamlit)

- **Live 모드**: 앱이 NVML을 직접 폴링한다 (100 ms, 두 GPU, 순시 포함).
  - 30 s 창 / 15 s stride로 Stage1을 실시간 채점한다.
  - 후보가 나오면 RAG → LLM 판정을 비동기로 실행한다.
- **Replay 모드**: 저장된 세션을 N배속으로 흘려보내 같은 파이프라인을 실행한다 (시연용).
- 화면 구성:
  - 관측 경계 배너 (실효 갱신, 값 변경률)
  - 전력 타임라인 (평균 + 순시, 후보 창 음영)
  - Stage1 패널 (p, 증거)
  - 검색 결과 (문서·점수·출처 계층)
  - LLM 판정 (인용 클릭)
- 원칙: 화면에 새 분석 로직을 만들지 않는다. 파이프라인 함수만 호출한다. 조작 버튼(종료·차단)은 두지 않는다 (P1).

## 8. 진행 기록

| 날짜 | 단계 | 결과 |
|---|---|---|
| 10-03 | LTMA holdout 검증 | round5b가 미학습 LTMA 6세션에서 0% → 31.8% (fold_00의 38% 재현). 새 serving은 기준선 밖이라 100% 후보 → §6 S-C 대상 |
| 10-03 | KB 원본 확보 | ATT&CK 19.2 (Enterprise 697 + ICS 97), ATLAS 5.6.0 (170 + 사례 57) |
| 10-03 | KB 구축 | `pipeline/kb/build_kb.py` → 1,091 문서 (카드 섹션 70 + ATT&CK 794 + ATLAS 227). 카드 10건에 `attack_ids`와 `## 관측 시그니처`(DRAFT) 추가 |
| 10-03 | 검색기 | `pipeline/kb/retriever.py` (tfidf/bm25/dense/hybrid/hybrid_rerank, CPU), `pipeline/kb/query.py` (결정적 렌더러, a-priori 절단값) |
| 10-03 | 판정기 | `pipeline/stage2_rag_judge.py`: 검색 문서 → LLM 최종 판정, 후보 카드 제약, 인용·증거 검증, 1회 재시도 후 invalid |
| 10-03 | 평가 세트 | `pipeline/stage2_v4_dataset.py` → 107 세그먼트 (SWMA 21, LTMA 11, crypto 6, unobservable 5, normal 64) |
| 10-03 | UI | `app.py` (Streamlit) + `pipeline/live.py` 엔진. Replay 헤드리스 시험: 80창 채점, 판정 카드 표시, 예외 없음. 큰 배치에서 창이 누락되던 버그를 수정하고 회귀 테스트 추가 |
| 10-03 | §5 합성 판단 | **kHz 합성은 만들지 않음**: piggyback/mimicry 실측이 이미 NVML 대역 밖 앵커라 C2를 실측으로 충족. **규모 합성은 보류**. 카드만 있고 실측이 없던 llmjacking·hidden_ml·sponge·burst는 C1을 불충족(재현 가능)하므로 **실측 캡처** (`scripts/queues/misuse_v1.txt`, 12세션) |
| 10-03 | 검색 v4.1 | 렌더러를 train 기준으로 수정: 사각파는 swing ≥ 100 W + 0.05–1 Hz (규칙성은 변별력 없음), Stage1 가중치 < 1인 증거는 질의에서 제외, "신고 작업과 일치" 문구 추가, 카드당 최대 2섹션. KB 1,091건 중 정답 카드 MRR 0.44 → 0.69–0.75, R@3 0.37 → 0.81–0.84 (legacy 질의 + rerank는 R@3 0.18). ⚠ 실패 분석에서 test 세그먼트를 정성적으로 봤으므로, 깨끗한 평가는 holdout과 오용 캡처로 한다 |
| 10-03 | 판정 프롬프트 v2 | v1은 정상 세그먼트의 78%를 unknown으로 회피했다 (흔한 증거를 이상으로 읽음). 증거를 희소/참고로 나누고 normal·unknown 규칙을 명시. train 12건 점검 10/12. v1 부분 결과: `stage2_v4_ablation_prompt_v1_partial.jsonl` |
| 10-03 | Stage1 대안 | 검증 = fold_01·02 test (cal은 conformal 자기보정이라 선택에 무의미). 규칙 "평균 정상 오탐 ≤ 0.10에서 탐지 최대" → **SA-IForest** (오탐 0.057, 탐지 0.605). 보고 fold_00 test: S0 0.181/0.544 → IForest 0.125/0.470, holdout LTMA 0.318 → 0.412. kNN+CUSUM은 오탐이 최저(0.034)이며 holdout LTMA 0.765. **S-C (serving 정상 편입, round6)**: holdout 새 serving 오탐 1.00 → 0.015 (가장 큰 단일 효과) |
| 10-03 | LLM ablation | 107 세그먼트 × 7 구성 × 2 LLM. gemma A5 0.757 (A0 0.579, A1 0.533, A6 0.570; McNemar p<0.001). deepseek는 format=json에서 빈 템플릿 → 자유 출력 + JSON 추출로 재실행. 최고 0.720 (A3), 지연 3배 → **gemma3:12b + A5 채택** |
| 10-03 | 지속성 게이트 | 세션 단위 평가에서 정상 세션 75%가 최종 경보 (P1 위반) → 후보 4창 연속 조건 (검증 fold 01·02로 선택). fold_00: 정상 세션 경보 1.00 → 0.10, 보이는 공격 0.93 → 0.73. `live.py` trigger 4 |
| 10-03 | 오용 공격 실측 | llmjacking·hidden_ml·sponge·burst 각 3 (burst 1 무효) |
| 10-03 | 처음 보는 클래스 평가 | 일반화 실패 (gemma A5 정확도 0.04). 원인: 렌더러 과적합, 관측상 동일 공격, 기준선 밖 정상. A7 다중 질의는 개선 없음. 상세 `RESULTS_v4.md` §2.4 |
| 10-03 | 파이프라인 장치 | 데이터 수집 종료 → 파이프라인 전용 GPU를 가정. 검색 모델 기본 장치를 마지막 CUDA 장치(cuda:1)로 변경 (`GRIDPULSE_RETRIEVER_DEVICE`로 재지정 가능). UI에서 파이프라인 장치와 감시 GPU를 분리하고, 겹치면 경고. 재정렬 13 s(CPU) → 0.3 s, 판정 총 5–11 s (대부분 LLM) |

---

## 9. v4.2 단계 (2026-10-03 사용자 지시: 세션 보정 → 질의 슬롯화 → 정제 출처 웹 연구)

### 9.1 세션 단위 보정
- 근거: 창 단위 conformal은 세션 간 분포 차이로 깨진다 (cal 0.05 → test 0.18). 4창 게이트의 N은 경험값이다.
- 데이터: round5b의 fold_00/01/02 test가 78세션을 겹침 없이 나눈다 → 모든 세션에 out-of-fold(OOF) 창 점수가 있다.
- 세션 통계: (R) 최장 연속 후보 길이, (C) conformal CUSUM 최댓값 (비중첩 창), (L) 4창 묶음 Σ(−log p) 최댓값.
- 교차 보정: 평가 fold f의 임계 = 다른 두 fold의 OOF 정상 세션 통계의 conformal 분위 (α = 0.05, 0.10). 보장 대상: "정상 세션이 경보로 끝날 확률 ≤ α".
- 비교: 고정 N=4 게이트. 추가 평가: holdout 12 + 오용 11 (fold_00 모델, 보정 = 전체 OOF 정상 세션).
- 산출: `pipeline/session_calibration.py`, `dataset/eval/session_calibration.json`.

### 9.2 질의 변환 → 슬롯형 관측 프로필
- 슬롯: 전력 수준 / 변동 형태 / 시간 척도 / 규칙성 / 사용률–전력 관계 / 신고 일치 / 다중 GPU / 진행 로그 관계. 항상 모두 채운다.
- 절단값: 물리값(TDP 450 W, 관측 Nyquist ~1 Hz) 또는 a-priori 정의. 카드도 같은 슬롯의 '관측 프로필'을 위협 지식에서 작성한다.
- 검색: 카드 = 슬롯 구조 매칭 점수 + 텍스트 검색 융합, ATT&CK/ATLAS = 텍스트 검색.
- 비교: v4 질의(A5) vs 슬롯 질의, 개발 107 + 처음 보는 23 세그먼트, 검색 R@k와 LLM 판정 모두.

### 9.3 정제 출처 웹 연구 (unknown → DRAFT 카드)
- 출처: arXiv API, Semantic Scholar API, MITRE ATT&CK/ATLAS 공식 저장소 갱신. 일반 웹 검색 엔진은 쓰지 않는다.
- 흐름: unknown 판정 또는 낮은 검색 점수 → 슬롯 프로필 → 영어 메커니즘 검색어 → 초록 수집 → LLM이 카드 스키마로 추출 (출처 URL 필수, 관측 시그니처는 '가설') → `corpus/drafts/` → 사람 승인 → KB 편입.
- 검증: 카드 하나를 KB에서 빼고 해당 세션으로 파이프라인이 그 위협을 다시 문서화하는지 본다.
| 10-03 | §9.1 세션 보정 | R 통계 + α=0.10 교차 conformal → 5창 게이트 (`session_gate.json`), OOF 정상 세션 경보 0.052. live/dataset 연결 |
| 10-03 | §9.2 슬롯 프로필 | A8·A9 모두 개발 세트에서 A5보다 유의하게 나쁨 (p=0.031, 0.004). 처음 보는 세트는 소폭 개선(유의하지 않음). **A5 유지**, 프로필은 연구 검색어용 |
| 10-03 | §9.3 연구 루프 | arXiv/S2/ATT&CK 출처, 근거 검증, DRAFT 격리. LOO: 파이프라인은 동작하나 unknown 발동 0, 찾아온 위협이 제거된 클래스와 다름 → 발동 조건과 Stage3 맥락이 과제 |

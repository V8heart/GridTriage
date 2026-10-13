# Physics Validation Layer (Tier B) — Cursor 프롬프트

> 첨부: `stage1_kundur_scaffold.py` (2026-09-09 실제 실행 검증됨, ANDES 2.0.0)
> 이 스캐폴드는 핵심 메커니즘(부하 변조 → 발전기 진동, 주파수 근접 시 응답 증폭)이
> 실제로 작동함을 이미 증명했다. 아래 프롬프트는 이걸 견고한 파이프라인으로 완성하는 작업이다.

---

## 컨텍스트 (Cursor에게 그대로 전달)

우리는 GPU workload 이상탐지 파이프라인(GridTriage)의 Physical Threat Potential Score가
지금은 근거 없는 가중합이라는 문제를 갖고 있다. 이를 실증적으로 보정하기 위해
공개 동적 전력계통 테스트케이스(Kundur 2-area, 이후 WECC 179)에 우리 공격 파형을
주입해서 실제 발전기 진동·주파수 응답을 관측하는 Tier B 검증 레이어를 만든다.

**절대 하지 말 것**: "한국 실제 전력망에서 이 공격이 발생하면 X Hz 떨어진다"는 주장.
우리가 쓰는 계통은 공개 표준 테스트케이스이지 한국 계통의 실제 동적 파라미터가 아니다.
표현은 항상 "GridTriage가 고위험으로 분류한 workload 패턴이 공개 동적 테스트 계통에서
유의미하게 큰 물리적 응답을 유발하는지 검증했다"로 한정한다.

첨부한 `stage1_kundur_scaffold.py`를 반드시 먼저 읽고, 아래 3개 함정을 이해한 뒤 작업을 시작해줘:

1. **PQ.alter('p0', ...)는 TDS 도중 아무 효과가 없다.** ANDES의 PQ 모델은 `p0`를
   초기화 시점에만 쓰고, 실제 동적 시뮬레이션에는 `Ppf`(정전력 성분)를 쓴다.
   반드시 `ss.PQ.config.p2p=1.0, p2i=0, p2z=0`을 설정한 뒤 `Ppf`를 alter해야 한다.
2. **너무 잦은(예: 0.05초 간격) 연속적 변경은 솔버를 불안정하게 만든다.** ANDES 공식
   WECC forced-oscillation 예제와 동일하게, 사각파(square wave) + half-period 단위
   토글 방식을 써야 한다.
3. **PSS(전력계통안정화장치) 없는 케이스는 감쇠가 약해 불연속 지점에서 수렴 실패가
   잦다.** `kundur_full.xlsx`는 PSS가 0개, `kundur_ieeest.xlsx`는 PSS 1개를 포함한다.
   후자를 기본으로 쓰되, 그래도 실패하면 `max_chunk`(현재 0.5초)를 더 줄이고
   `ss.TDS.run()` 실패 시의 재시도/축소 로직을 보강해야 한다. 이 수렴 안정화 자체가
   Stage 1의 핵심 작업 중 하나다 — 완벽하게 도는 스크립트를 기대하지 말고, 실패 케이스를
   로그로 남기면서 견고하게 만드는 것 자체가 산출물이다.

## Phase 1 — Kundur 프로토타입 완성 (스캐폴드 → 견고한 파이프라인)

1. `stage1_kundur_scaffold.py`를 `bit2watt_impl/physics/stage1_kundur.py`로 옮기고,
   `dataset/synth_attacks.py`에 이미 정의된 정상/SWMA/LTMA/cryptojacking 파형 생성
   로직과 동일한 파라미터 체계(주파수, 진폭, 지속시간)를 공유하도록 리팩터링해줘.
2. `_run_to()`의 수렴 실패 처리를 보강해줘: 실패 시 `max_chunk`를 절반으로 줄여
   재시도하고, 그래도 3회 연속 실패하면 해당 케이스를 skip하되 로그에 남기는 방식으로.
3. 4종 공격(정상/주기적 변조/버스트/램프) 각각에 대해 `extract_features()`가 만드는
   `rocof_hz_s`, `osc_std`, `osc_ptp`, `dominant_freq_hz`를 CSV로 저장해줘.
4. 이 4개 feature를 기존 `pipeline/features.py`의 출력과 나란히 붙여서, 하나의
   attack_id가 (cyber-side feature) + (physics-side feature) 양쪽을 다 갖도록
   스키마를 통일해줘.

## Phase 2 — 주파수 스윕

5. `np.arange(0.1, 1.5, 0.05)` 범위로 공격 주파수를 스윕하면서 Phase 1 파이프라인을
   반복 실행하고, (주파수, osc_std) 곡선을 그려줘.
6. 이 곡선이 Kundur의 공개 inter-area mode(~0.6Hz) 근처에서 봉우리를 보이는지
   확인하고, 결과를 `dataset/eval/frequency_sweep_results.csv`와 그래프 이미지로
   저장해줘. 봉우리가 명확하지 않으면 그것도 있는 그대로 보고해줘 — 과장하지 마.

## Phase 3 — WECC 179 확장

7. ANDES 공식 저장소의 WECC 179버스 케이스(`andes/cases/wecc/wecc.raw`,
   `wecc_gencls.dyr`)를 사용해서 Phase 1의 주입 로직을 데이터센터 부하 버스에
   맞게 이식해줘. ANDES 공식 gallery의 forced-oscillation 예제
   (https://docs.andes.app/en/latest/gallery/forced-oscillation.html)가
   generator Paux에 주입하는 방식인데, 우리는 이걸 부하 버스의 PQ.Ppf 주입으로
   바꿔야 한다는 점을 유의해줘(Phase 1의 함정 1과 동일한 원리).
8. Kundur와 같은 경향(공격 주파수가 계통 고유모드에 가까울수록 응답 증폭)이
   더 큰 계통에서도 재현되는지 비교해줘.

## Phase 4 — GridTriage 연결

9. `pipeline/run_pipeline.py`의 출력(anomaly score, threat_id, RAG 매칭 결과)과
   physics layer의 출력(rocof_hz_s, osc_std)을 attack_id 기준으로 join하는
   `pipeline/physics_correlation.py`를 새로 작성해줘.
10. 최종적으로 "cyber 탐지기가 고위험으로 분류한 패턴이 물리 시뮬레이션에서도
    유의하게 큰 응답을 보이는가"를 상관계수나 그룹별 비교(boxplot)로 보여주는
    분석 스크립트를 작성해줘.
11. 이 결과를 §4.2 Physical Threat Potential Score의 가중치 보정에 반영하는 방법을
    제안해줘(예: 시뮬레이션에서 관측된 osc_std가 큰 특성군에 더 큰 가중치를 부여).

## 표현 규율 (모든 산출물에 적용)

- "한국 계통에서 실제로 발생하면" 같은 표현 금지
- 반드시 "공개 동적 테스트 계통(Kundur/WECC)에서" 라는 한정을 붙일 것
- 수렴 실패·불확실성도 결과의 일부로 정직하게 보고할 것(과장 금지)
- KPG-193(Tier A, 정적·위치별 민감도)과 이 Tier B(동적·범용 계통)는 절대 섞어서
  서술하지 말 것 — 각자 다른 결론을 낼 수 있는 별개의 실험임을 항상 명시

각 Phase가 끝나면 실행 로그, 생성된 파일 목록, 수치 결과 요약을 알려줘.

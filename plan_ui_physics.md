# 콘솔 물리 패널·계통 지표 연결 계획 (2026-10-04)

## 목적

경보가 난 구간에 대해 콘솔이 다음 두 가지를 함께 보여 준다.
1. **"이 패턴이 시설 규모로 커지면 테스트 계통이 어떻게 반응하는가"** (물리 재생)
2. **"우리 부하가 지금 계통에 주는 자극과, 계통 측 신호가 들어오면 그 상태"** (계통 지표)

시뮬레이션은 "현재 그리드 상태"가 아니라 **사전 민감도 판단 도구**로만 표시한다.

## 조사 결과 (설계에 반영한 것)

| 항목 | 내용 | 반영 |
|---|---|---|
| UPS 계통 측 신호 | RFC 1628 UPS-MIB: `upsInputFrequency` 1.3.6.1.2.1.33.1.3.3.1.2 (0.1 Hz 단위), `upsInputVoltage` …1.3 (RMS V) | `GridSource` 인터페이스. SNMP 구현은 장비 연결 시 추가하고, 지금은 CSV 재생 소스만 둔다 |
| 서버 PSU 신호 | DMTF Redfish `PowerSupply.LineInputVoltage`, `PowerSupplyMetrics`(입력 전압·전력·주파수) | 같은 인터페이스의 후보 소스로 문서화 |
| 부하 동특성 지표 | 대형 데이터센터 부하 규격 연구: 램프는 주파수 nadir·RoCoF로, 학습 진동은 전력의 푸리에 계수(대역 성분)로 제한 | 시설 지표 = 모드 대역(0.1–2 Hz) 변동 진폭(TDP 대비) + 최대 램프 + GPU 간 동기화 |
| 실제 사례 규모 | 데이터센터 부하 감소로 RoCoF 약 0.26 Hz/s, 주파수 +0.32 Hz (아일랜드) | 패널의 RoCoF 해석 참고값으로만 표기 |
| Streamlit | `st.fragment(run_every=…)`로 상태를 주기적으로 그리고, 무거운 작업은 별도 스레드에서 돌리며 공유 상태는 락으로 보호 | 물리 재생은 엔진의 전용 워커 스레드에서 실행. UI는 `snapshot()`만 읽는다 |
| 실행 환경 | ANDES 2.0은 Python 3.11 전용 환경(`.venv-physics`)에만 있고, 1회 재생에 약 6–7초 | 하위 프로세스 워커(JSON 입출력)와 시간 제한을 둔다. 세션 경보마다 1회만 돌리고 결과는 캐시한다 |

출처:
- [RFC 1628 UPS-MIB](https://www.ietf.org/rfc/rfc1628.html)
- [DMTF Redfish Resource and Schema Guide](https://www.dmtf.org/sites/default/files/standards/documents/DSP2046_2024.2.html)
- [Designing Grid-Aware Dynamic Specifications for Large Data Center Loads (arXiv 2609.18888)](https://arxiv.org/abs/2609.18888)
- [Eaton: Data center as a good grid citizen](https://www.eaton.com/content/dam/eaton/markets/data-center/the-impact-of-ai/whitepapers/eaton-whitepaper-technical-data-center-as-a-good-grid-citizen-en-us.pdf)
- [Streamlit fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments)

## 진폭 규칙 (소개서 §4.1.5와 일치)

```
ΔP_주입 / P_계통부하 = (ΔP_GPU / TDP) × 참여율 × (P_시설 / P_계통부하)
```

- ΔP_GPU/TDP는 측정값이다.
- 참여율과 시설 비중은 화면에서 고르는 시나리오 파라미터다(기본: 참여율 1.0, 시설 비중 1%).
- 선택한 부하 모선에 주입하는 비율은 계통 부하 대비 비율에 `P_계통부하 / p0_모선`을 곱해 환산한다.
- 모선 부하의 90%를 넘으면 잘라내고 `clipped`로 표시한다.
- 기존 방식(창 평균 대비 상대 변동, 15% 상한)은 `scale="window"`로 남겨 예전 결과를 재현할 수 있게 한다.

## 구현 단위

1. `bit2watt_impl/physics/simulation.py`
   - `inject_observed_waveform(..., facility=None)`: facility 파라미터가 있으면 위 규칙으로 진폭을 정한다.
   - 결과에 `injected_frac_of_system_load`와 `clipped`를 기록한다.
2. `bit2watt_impl/physics/replay_worker.py` (.venv-physics에서 실행)
   - stdin JSON을 받아 stdout JSON을 낸다(시간 제한 포함).
3. `pipeline/physics_bridge.py` (.venv에서 실행)
   - 하위 프로세스 호출, 결과 캐시, 환경 점검(`available()`)
   - 모드 겹침 판정: `dataset/eval/modal_analysis.json`의 WECC 모드(IBR 0%)와 비교
4. `pipeline/live.py`
   - `EngineSettings.physics`(on/off, 계통, 참여율, 시설 비중)
   - 세션 경보 첫 발생 때 그 구간의 전력 파형(순시 전력 우선)으로 물리 재생을 제출하고, 결과를 분석 항목에 붙인다
   - `facility_forcing(rows)`: 시설 합산 전력의 모드 대역 진폭·최대 램프·GPU 간 상관
   - `GridSource` 인터페이스: `NoGridSource`(미연결), `CsvGridSource`(t_epoch, freq_hz, voltage_v)
5. `app.py`
   - 사이드바 물리 설정
   - "계통 연계" 패널: 시설 자극 지표 + 계통 측 지표(미연결이면 수집 경로 안내)
   - 판정 카드마다 물리 응답(osc_std, RoCoF, 지배 주파수, 모드 겹침, 해석 문구)
6. 시험
   - 브리지(가짜 워커로 하위 프로세스 대체)
   - 진폭 규칙 단위 시험
   - 시설 지표
   - 엔진이 경보 시 물리 작업을 한 번만 제출하는지
   - AppTest로 재생 모드 E2E: 시작 → 창 채점 → 판정 → 물리 결과

## 범위 밖

- SNMP·Redfish 실제 수집기: 이 서버에 UPS·BMC 노출이 없어 시험할 수 없다.
- 실제 계통의 실시간 상태 추정.

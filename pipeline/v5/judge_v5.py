"""Stage2 v5 judge: v4 retrieved cards + extra NVML channels + case-memory fit, then an open-set gate.

Differences from pipeline.stage2_rag_judge (v4, unchanged):
  * the prompt carries memory / utilisation / process channels as normal-relative quantiles
    (the cards already talk about memory and SM utilisation, but v4 never sent them)
  * it carries case-memory evidence: for each class with approved sessions, how well this segment
    fits that class's cases (conformal p), and which cards have no case support
  * "p_value > 0.05 means normal" is dropped: Stage1 p is a window screen, not a class decision
  * a deterministic open-set gate (alpha fixed a priori, not tuned on data):
      (a) attack:c, c has case support and fit_p[c] < alpha             -> unknown (card/case mismatch)
      (b) normal, fit to normal and to normal@declared family < alpha       -> unknown
      (c) analyst-enrolled classes: when a class an analyst confirmed and enrolled is the nearest case
          (>= half of the window votes) and its own fit is not rejected (or it has a single case, so no
          reference yet), the verdict is that class -- the rule eval_enrollment validated (91-100% on
          later sessions after one enrolment). The LLM cannot apply it: a card-less class is not among
          the retrieved cards, and gemma3:12b misreads fit numbers. Seed classes are excluded: applied to
          them, the rule pulled never-seen threats onto SWMA cases (P2 0.81 -> 0.72). A 'normal' match also
          needs the declared family's normal cases to fit, or LLMjacking (serving under a false declaration)
          is laundered into an enrolled serving normal.
    LLM verdicts are closed-world (they pick the nearest card); the gate is what opens the world.
"""
from __future__ import annotations

import json
import time

from pipeline.stage2_rag_judge import (BENIGN_CARDS, SCHEMA, Judgment, _call_claude, _call_ollama, allowed_cards,
                                       is_api_model, split_evidence, validate)

ALPHA = 0.05
DOC_CHARS = 700
CHANNELS = {
    "m_mean_gb": "GPU 메모리 사용량(GB)",
    "m_range_mb": "메모리 변화폭(MB)",
    "u_mean": "GPU 사용률 평균(%)",
    "u_frac_hi": "사용률 90% 이상 비율",
    "up_corr": "사용률-전력 상관",
    "w_per_util": "사용률당 전력(W/%)",
    "n_procs": "GPU 프로세스 수",
    "i_minus_p_std": "순시-평균 전력 차 표준편차(W)",
    "p_acf5s": "전력 5초 자기상관",
}

INSTRUCTION = """당신은 AI 데이터센터 GPU 전력 위협 분류 분석가입니다.
'관측 증거'는 NVML 텔레메트리 구조화 요약이고, '추가 관측 채널'은 메모리·사용률·프로세스 통계를
승인된 정상 사례 분포의 분위(normal_q, 0=정상 최저, 1=정상 최고)와 함께 준 것입니다.
'사례 일치도'는 승인된 실제 사례(정상·공격) 기억과 비교한 결과입니다. fit_p는 그 클래스의 실제 사례들과
닮은 정도이며(보정된 p값, 0.05 미만이면 그 클래스 사례들과 맞지 않음), 사례가 없는 카드는 문서로만 판단합니다.
'검색 문서'에는 관련 없는 문서도 섞여 있습니다.

판정 규칙:
1. 관측 증거, 추가 채널, 사례 일치도, 검색 문서만 근거로 판단합니다.
2. 공격 카드를 고를 때 그 카드에 사례가 있으면 fit_p가 0.05 이상이어야 합니다. 사례가 없는 카드는
   관측 시그니처가 증거·채널과 구체적으로 맞을 때만 고릅니다.
3. 정상(normal_workloads, benign_periodic)은 정상 사례와 맞거나(정상 fit_p >= 0.05) 희소 증거가 없을 때입니다.
4. 어떤 카드·사례와도 맞지 않으면 unknown입니다. 가장 비슷한 카드에 억지로 맞추지 마세요.
   unobservable은 관측 대역(약 1 Hz) 밖이라 판단할 수 없을 때입니다.
5. cited_docs에는 문서 목록의 doc_id만, evidence_used에는 관측 증거·pattern·채널 이름만 씁니다.
   explanation에 근거를 2-3문장으로 씁니다. 이 판정은 조사 우선순위이며 차단 명령이 아닙니다."""


def channel_block(quantiles: dict) -> dict:
    return {CHANNELS[k]: v for k, v in quantiles.items() if k in CHANNELS}


def case_block(case: dict, cards: list[str]) -> dict:
    fit = {c: p for c, p in case["class_p"].items() if "@" not in c}
    return {
        "클래스별 fit_p": fit,
        "신고 작업 계열 정상 fit_p": case.get("declared_normal_p"),
        "가장 가까운 사례 클래스 비율": case["nearest_case_votes"],
        "사례 없는 카드": sorted(c for c in cards if c not in fit and c not in BENIGN_CARDS),
    }


def build_prompt(bundle: dict, docs: list[dict], cards: list[str], case: dict, channels: dict,
                 n_windows: int | None = None) -> str:
    rare, common = split_evidence(bundle)
    payload = {
        "관측 증거": {
            "declared_context": bundle.get("declared_context"),
            "pattern": bundle.get("pattern"),
            "희소 증거": rare,
            "참고 증거": common,
            "창 수(30초 창, 15초 간격)": n_windows,
        },
        "추가 관측 채널": channel_block(channels),
        "사례 일치도": case_block(case, cards),
        "allowed_threat_cards": cards,
        "검색 문서": [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"][:DOC_CHARS]} for d in docs],
        "출력 JSON 스키마": SCHEMA,
    }
    return INSTRUCTION + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)


def card_class(card: str | None) -> str | None:
    """Threat card -> case-memory class (benign cards map to 'normal')."""
    if card in BENIGN_CARDS:
        return "normal"
    return card


def open_set_gate(verdict: str, card: str | None, case: dict, *, alpha: float = ALPHA,
                  normal_rule: bool = False) -> dict | None:
    fit = case["class_p"]
    if verdict == "attack":
        cls = card_class(card)
        if cls in fit and fit[cls] < alpha:
            return {"rule": "a_card_case_mismatch", "card": card, "fit_p": fit[cls]}
        if card == "coordinated_multi_gpu" and "swma" in fit and fit["swma"] < alpha:
            return {"rule": "a_card_case_mismatch", "card": card, "fit_p": fit["swma"]}
    if normal_rule and verdict == "normal":
        dn = case.get("declared_normal_p")
        if fit.get("normal", 1.0) < alpha and (dn is None or dn < alpha):
            return {"rule": "b_unexplained_anomaly", "fit_p_normal": fit.get("normal"), "fit_p_declared": dn}
    return None


def enrolled_case_match(case: dict, enrolled: set[str] | frozenset = frozenset(), *,
                        alpha: float = ALPHA) -> str | None:
    votes = case.get("nearest_case_votes") or {}
    if not votes or not enrolled:
        return None
    top = max(votes, key=votes.get)
    if top not in enrolled or votes[top] < 0.5:
        return None
    p = (case.get("class_p") or {}).get(top)
    if p is not None and p < alpha:
        return None
    if top == "normal":
        # A misuse can look exactly like an enrolled normal job (LLMjacking = a serving workload under a false
        # declaration). Normal by case also needs the honest jobs of the *declared* family to fit.
        dn = case.get("declared_normal_p")
        if dn is None or dn < alpha:
            return None
    return top


def normalize_raw(raw: dict, cards: list[str]) -> dict:
    """Small judges sometimes write the card into the verdict field ({"verdict": "swma"}); read it as attack:swma."""
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict in cards and not raw.get("threat_card"):
        return {**raw, "verdict": "attack", "threat_card": verdict}
    if verdict in BENIGN_CARDS:
        return {**raw, "verdict": "normal", "threat_card": verdict}
    return raw


def judge_v5(bundle: dict, docs: list[dict], case: dict, channels: dict, *, model: str = "gemma3:12b",
             host: str | None = None, n_windows: int | None = None, retries: int = 1,
             normal_rule: bool = False, enrolled: set[str] | frozenset = frozenset()) -> Judgment:
    """``enrolled``: analyst-enrolled classes (rule c). Card-less ones are added to the allowed cards."""
    retrieved = allowed_cards(docs)
    cards = retrieved + sorted(c for c in enrolled if c not in retrieved and c != "normal")
    doc_ids = {d["doc_id"] for d in docs}
    evidence = (bundle.get("unexplainedness") or {}).get("evidence") or {}
    names = {k for k, v in evidence.items() if v} | set(bundle.get("pattern") or {}) | set(CHANNELS.values())
    prompt = build_prompt(bundle, docs, cards, case, channels, n_windows)
    started = time.monotonic()
    result = None
    for attempt in range(1, retries + 2):
        try:
            raw = _call_claude(prompt, model) if is_api_model(model) else _call_ollama(prompt, model, host=host)
            result = validate(normalize_raw(raw, cards), cards=cards, doc_ids=doc_ids, evidence_names=names)
        except Exception as exc:
            result = Judgment("unknown", None, valid=False, errors=[f"{type(exc).__name__}: {exc}"])
        result.attempts = attempt
        if result.valid:
            break
        prompt = prompt + "\n\n이전 출력 오류: " + "; ".join(result.errors) + "\n규칙을 지켜 다시 출력하세요."
    result.model = model
    result.latency_s = round(time.monotonic() - started, 2)
    result.gate = {"llm_verdict": result.verdict if result.valid else "invalid", "llm_card": result.threat_card}
    matched = enrolled_case_match(case, enrolled)  # deterministic: applies even when the LLM output was invalid
    if not result.valid and not matched:
        result.verdict, result.gate = "invalid", None
        return result
    if matched:
        if matched == "normal":
            result.gate.update(rule="c_enrolled_case_match", card="normal_workloads",
                               votes=case["nearest_case_votes"].get(matched), fit_p=case["class_p"].get(matched))
            result.verdict, result.threat_card = "normal", "normal_workloads"
            return result
        result.gate.update(rule="c_enrolled_case_match", card=matched, votes=case["nearest_case_votes"].get(matched),
                           fit_p=case["class_p"].get(matched))
        result.verdict, result.threat_card = "attack", matched
        return result
    fired = open_set_gate(result.verdict, result.threat_card, case, normal_rule=normal_rule)
    if fired:
        result.gate.update(fired)
        result.verdict, result.threat_card = "unknown", None
    return result

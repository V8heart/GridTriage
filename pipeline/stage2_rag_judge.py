"""Stage2 v4: retrieved documents -> LLM final verdict.

The retriever ranks KB documents; the LLM reads the evidence bundle and the
retrieved documents and returns the final verdict. No attack probability is
produced. The LLM may only name a threat card that was retrieved (candidate
constraint, cf. TechniqueRAG), and must cite retrieved doc_ids and evidence
names that exist in the bundle; violations trigger one retry, then ``invalid``.
A deterministic requirement gate then checks the chosen threat card against the
bundle: a required evidence that is off, or an exclusion evidence that is on,
turns ``attack`` into ``unknown`` (the research trigger), since the LLM tends
to force an unknown threat onto the nearest known card. Off by default: the approved
cards' required evidence does not yet match real captures (RESULTS_v4.md §6).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field

from pipeline.stage2_llm import ollama_url

VERDICTS = ("attack", "normal", "unobservable", "unknown")
BENIGN_CARDS = {"normal_workloads", "benign_periodic"}
DOC_CHARS = 700

INSTRUCTION = """당신은 AI 데이터센터 GPU 전력 위협 분류 분석가입니다.
아래 '관측 증거'는 NVML 텔레메트리에서 계산한 구조화 요약입니다. 원시 시계열은 없습니다.
'검색 문서'는 위협 카드와 MITRE ATT&CK/ATLAS 문서이며, 관련 없는 문서도 섞여 있을 수 있습니다.

판정 규칙:
1. 관측 증거와 검색 문서만 근거로 판단합니다. 문서의 '관측 시그니처'와 증거를 대조하세요.
2. '희소 증거'는 정상 학습·추론에서 드물게만 나타나는 신호입니다. '참고 증거'는 정상에서도 흔해 그 자체로 이상의 근거가 아닙니다.
   stage1.p_value가 0.05보다 크면 통계적으로 정상 범위입니다.
3. verdict는 다음 중 하나입니다.
   - attack: 희소 증거와 pattern이 특정 위협 카드의 관측 시그니처와 맞을 때. threat_card는 allowed_threat_cards 중 하나.
   - normal: 희소 증거가 없거나, 증거가 정상(normal_workloads, benign_periodic) 카드와 맞을 때.
   - unobservable: 이상이 의심되나 NVML 관측 대역(약 1 Hz) 밖이라 판정할 수 없을 때.
   - unknown: 희소 증거가 있으나 어떤 카드와도 맞지 않을 때에만.
4. cited_docs에는 아래 문서 목록에 있는 doc_id만 씁니다. evidence_used에는 관측 증거의 이름이나 pattern 키만 씁니다.
   확률이나 점수를 만들지 말고, explanation에 근거를 2-3문장으로 씁니다.
5. 이 판정은 조사 우선순위이며 차단 명령이 아닙니다."""

SCHEMA = {
    "verdict": "attack|normal|unobservable|unknown",
    "threat_card": "allowed_threat_cards 중 하나 또는 null",
    "cited_docs": ["doc_id"],
    "evidence_used": ["evidence name"],
    "explanation": "한국어 2-3문장",
}


@dataclass
class Judgment:
    verdict: str
    threat_card: str | None
    cited_docs: list[str] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    explanation: str = ""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    model: str = ""
    latency_s: float = 0.0
    attempts: int = 1
    gate: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def allowed_cards(docs: list[dict], all_cards: list[str] | None = None) -> list[str]:
    if all_cards is not None:
        return sorted(all_cards)
    seen = []
    for doc in docs:
        card = doc.get("card")
        if card and card not in seen:
            seen.append(card)
    return seen


def split_evidence(bundle: dict) -> tuple[list[str], list[str]]:
    """True evidence split by Stage1 train-normal surprisal weight (>= 1.0 is rare in normals)."""
    from pipeline.kb.query import ALWAYS_RENDER, MIN_EVIDENCE_WEIGHT, official_weights

    weights = official_weights() or {}
    true = sorted(k for k, v in ((bundle.get("unexplainedness") or {}).get("evidence") or {}).items() if v)
    neutral = ALWAYS_RENDER | {"period_comparable", "progress_log_available", "progress_log_missing"}
    rare = [k for k in true if k not in neutral and weights.get(k, MIN_EVIDENCE_WEIGHT) >= MIN_EVIDENCE_WEIGHT]
    return rare, [k for k in true if k not in rare]


def build_prompt(bundle: dict, docs: list[dict], cards: list[str], *, show_docs: bool = True) -> str:
    rare, common = split_evidence(bundle)
    payload = {
        "관측 증거": {
            "declared_context": bundle.get("declared_context"),
            "pattern": bundle.get("pattern"),
            **({"관측 프로필": bundle["observation_profile"]} if bundle.get("observation_profile") else {}),
            "희소 증거": rare,
            "참고 증거": common,
            "stage1": {k: (bundle.get("unexplainedness") or {}).get(k) for k in ("u_score", "p_value")},
        },
        "allowed_threat_cards": cards,
        "검색 문서": [
            {"doc_id": d["doc_id"], "title": d["title"], "text": d["text"][:DOC_CHARS]} for d in docs
        ] if show_docs else "제공 안 함 (카드 이름만 허용)",
        "출력 JSON 스키마": SCHEMA,
    }
    return INSTRUCTION + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)


# Reasoning models return an empty schema template under format=json (constrained
# decoding skips the reasoning), so they answer freely and the JSON is extracted.
REASONING_MODELS = ("deepseek-r1", "qwen3")


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fenced or re.findall(r"\{.*\}", text, flags=re.S)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def _call_ollama(prompt: str, model: str, *, timeout: int = 600, host: str | None = None) -> dict:
    url = f"{host.rstrip('/')}/api/generate" if host else ollama_url("/api/generate")
    reasoning = model.startswith(REASONING_MODELS)
    body = {"model": model, "prompt": prompt + ("\n\n최종 답은 JSON 객체 하나로만 출력하세요." if reasoning else ""),
            "stream": False, "options": {"temperature": 0, "seed": 20260922, "num_ctx": 16384}}
    if not reasoning:
        body["format"] = "json"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return _extract_json(out.get("response", ""))


def validate(raw: dict, *, cards: list[str], doc_ids: set[str], evidence_names: set[str]) -> Judgment:
    errors = []
    verdict = str(raw.get("verdict", "")).strip().lower()
    card = raw.get("threat_card")
    card = None if card in (None, "", "null", "None") else str(card).strip()
    if verdict not in VERDICTS:
        errors.append(f"bad verdict {verdict!r}")
    if card in BENIGN_CARDS and verdict == "attack":
        verdict = "normal"
    if verdict == "attack" and card not in cards:
        errors.append(f"threat_card {card!r} not in allowed {cards}")
    cited = [str(d) for d in (raw.get("cited_docs") or []) if isinstance(d, (str, int))]
    bad_cites = [d for d in cited if d not in doc_ids]
    if doc_ids and bad_cites:
        errors.append(f"cited docs not provided: {bad_cites[:3]}")
    used = [str(e) for e in (raw.get("evidence_used") or [])]
    return Judgment(verdict=verdict, threat_card=card if verdict == "attack" else (card if card in BENIGN_CARDS else None),
                    cited_docs=[d for d in cited if d in doc_ids],
                    evidence_used=[e for e in used if e in evidence_names],
                    explanation=str(raw.get("explanation", ""))[:1200],
                    valid=not errors, errors=errors)


def judge(bundle: dict, docs: list[dict], *, model: str = "gemma3:12b", all_cards: list[str] | None = None,
          show_docs: bool = True, retries: int = 1, host: str | None = None, gate: bool = False) -> Judgment:
    cards = allowed_cards(docs, all_cards)
    doc_ids = {d["doc_id"] for d in docs} if show_docs else set()
    pattern = bundle.get("pattern") or {}
    evidence = (bundle.get("unexplainedness") or {}).get("evidence") or {}
    names = {k for k, v in evidence.items() if v} | set(pattern)
    prompt = build_prompt(bundle, docs, cards, show_docs=show_docs)
    started = time.monotonic()
    result = None
    for attempt in range(1, retries + 2):
        try:
            raw = _call_ollama(prompt, model, host=host)
            result = validate(raw, cards=cards, doc_ids=doc_ids, evidence_names=names)
        except Exception as exc:  # network / JSON errors count as invalid attempts
            result = Judgment("unknown", None, valid=False, errors=[f"{type(exc).__name__}: {exc}"])
        result.attempts = attempt
        if result.valid:
            break
        prompt = prompt + "\n\n이전 출력 오류: " + "; ".join(result.errors) + "\n규칙을 지켜 다시 출력하세요."
    result.model = model
    result.latency_s = round(time.monotonic() - started, 2)
    if not result.valid:
        result.verdict = "invalid"
    elif gate and result.verdict == "attack":
        result.gate = requirement_gate(result.threat_card, docs, evidence)
        if result.gate:
            result.verdict, result.threat_card = "unknown", None
    return result


def requirement_gate(card: str | None, docs: list[dict], evidence: dict) -> dict | None:
    """Reasons the chosen card contradicts the bundle, or None when it holds (or carries no rules)."""
    rules = next((d for d in docs if d.get("card") == card and "requires" in d), None)
    if rules is None:
        return None
    missing = [e for e in rules["requires"] if not evidence.get(e)]
    excluded = [e for e in rules["excludes"] if evidence.get(e)]
    return {"card": card, "missing_required": missing, "exclusion_on": excluded} if missing or excluded else None


def is_correct(judgment: Judgment | dict, gt_card: str, accept: list[str]) -> bool:
    j = judgment if isinstance(judgment, dict) else judgment.to_dict()
    verdict, card = j["verdict"], j.get("threat_card")
    if gt_card == "normal":
        return verdict == "normal"
    if gt_card == "unobservable":
        return verdict in {"normal", "unobservable"} or (verdict == "attack" and card in accept)
    return verdict == "attack" and card in accept

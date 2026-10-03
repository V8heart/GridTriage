"""Evidence-grounded Stage 2 LLM judge."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, replace

from pipeline.corpus_schema import CorpusDoc, checklist_satisfied

VALID_VERDICTS = {"known", "partial", "unknown", "normal"}


def ollama_url(path: str = "/api/generate") -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{host}{path}"


@dataclass(frozen=True)
class Stage2Verdict:
    verdict: str
    closest_match: str | None
    confidence: float
    matched_evidence: list[str]
    contradicting_evidence: list[str]
    explanation: str
    fallback_used: bool = False
    prompt_eval_count: int | None = None


def _validate(data: dict) -> Stage2Verdict:
    missing = {"verdict", "closest_match", "confidence", "matched_evidence", "contradicting_evidence", "explanation"} - set(data)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if data["verdict"] not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {data['verdict']}")
    return Stage2Verdict(
        verdict=str(data["verdict"]),
        closest_match=data.get("closest_match"),
        confidence=float(data["confidence"]),
        matched_evidence=list(data["matched_evidence"]),
        contradicting_evidence=list(data["contradicting_evidence"]),
        explanation=str(data["explanation"]),
        fallback_used=bool(data.get("fallback_used", False)),
        prompt_eval_count=int(data["prompt_eval_count"]) if data.get("prompt_eval_count") is not None else None,
    )


def normal_verdict(name: str | None) -> Stage2Verdict:
    return Stage2Verdict(
        verdict="normal",
        closest_match=name,
        confidence=1.0,
        matched_evidence=[],
        contradicting_evidence=[],
        explanation=f"RAG top-1 '{name}' is a benign corpus document",
        fallback_used=False,
    )


def fallback_verdict(*, top1_name: str | None, reason: str = "fallback") -> Stage2Verdict:
    """Legacy fallback uses retriever top-1 name, never alphabet-first corpus doc."""
    return Stage2Verdict(
        verdict="partial" if top1_name else "unknown",
        closest_match=top1_name,
        confidence=0.0,
        matched_evidence=[],
        contradicting_evidence=[],
        explanation=f"fallback_used: {reason}",
        fallback_used=True,
    )


def distinction_excerpt(body: str) -> str:
    """Keep the handbook distinction section; do not send the full body."""
    lines = (body or "").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("##") and "구분 포인트" in line:
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## ") and "구분 포인트" not in lines[index]:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def checklist_facts(doc: CorpusDoc, evidence: dict) -> dict[str, list[str]]:
    """Deterministic required_met / excluded_by facts. Not a filter."""
    required_met: list[str] = []
    excluded_by: list[str] = []
    for item in doc.evidence:
        if item.necessity == "required" and evidence.get(item.name):
            required_met.append(item.name)
        elif item.necessity == "required_any":
            required_met.extend(name for name in item.any_of if evidence.get(name))
        elif item.necessity == "exclusion" and evidence.get(item.name):
            excluded_by.append(item.name)
    return {"required_met": required_met, "excluded_by": excluded_by}


def apply_consistency_checks(
    verdict: Stage2Verdict,
    evidence_bundle: dict,
    candidate_docs: list[CorpusDoc],
) -> Stage2Verdict:
    ev = ((evidence_bundle.get("unexplainedness") or {}).get("evidence") or {})
    allowed = set(ev)
    matched = [name for name in verdict.matched_evidence if name in allowed]
    contradicting = [name for name in verdict.contradicting_evidence if name in allowed]
    by_name = {doc.name: doc for doc in candidate_docs}
    value = verdict.verdict
    closest = verdict.closest_match
    if closest and closest not in by_name:
        value = "partial"
    category = by_name[closest].category if closest and closest in by_name else None
    if value == "normal" and category and category != "benign":
        value = "partial"
    if value == "known" and category == "benign":
        value = "partial"
    return replace(
        verdict,
        verdict=value,
        matched_evidence=matched,
        contradicting_evidence=contradicting,
    )


def judge(
    evidence_bundle: dict,
    candidate_docs: list[CorpusDoc],
    *,
    backend: str = "stub",
    model: str = "gemma3:12b",
    fallback: str = "legacy",
    retriever_top1: str | None = None,
) -> Stage2Verdict:
    top1 = retriever_top1
    evidence = evidence_bundle.get("unexplainedness", {}).get("evidence", {})
    if backend == "stub":
        hits: list[tuple[CorpusDoc, list[str], list[str]]] = []
        for doc in candidate_docs:
            ok, matched, contradicting = checklist_satisfied(doc, evidence)
            if ok:
                hits.append((doc, matched, contradicting))
        if not hits:
            return fallback_verdict(top1_name=top1, reason="stub_no_required_match")
        if len(hits) == 1:
            doc, matched, contradicting = hits[0]
            return apply_consistency_checks(
                Stage2Verdict(
                    "known",
                    doc.name,
                    0.8,
                    matched,
                    contradicting,
                    f"required checklist matched: {', '.join(matched)}",
                ),
                evidence_bundle,
                candidate_docs,
            )
        best_n = max(len(matched) for _, matched, _ in hits)
        top = [item for item in hits if len(item[1]) == best_n]
        names = ", ".join(doc.name for doc, _, _ in hits)
        if len(top) > 1:
            verdict = Stage2Verdict(
                "partial",
                None,
                0.0,
                [],
                [],
                f"multiple checklists matched: {names}",
            )
        else:
            doc, matched, contradicting = top[0]
            verdict = Stage2Verdict(
                "partial",
                doc.name,
                0.0,
                matched,
                contradicting,
                f"multiple checklists matched: {names}",
            )
        return apply_consistency_checks(verdict, evidence_bundle, candidate_docs)

    prompt = {
        "instruction": (
            "Judge only from the structured evidence names, pattern card, "
            "and the listed checklist facts. required_met/excluded_by are facts, not a verdict."
        ),
        "evidence_bundle": evidence_bundle,
        "candidate_docs": [
            {
                "name": doc.name,
                "threat_id": doc.threat_id,
                "category": doc.category,
                "evidence": [asdict(item) for item in doc.evidence],
                "distinction": distinction_excerpt(doc.body),
                **checklist_facts(doc, evidence),
            }
            for doc in candidate_docs
        ],
        "schema": {
            "verdict": "known|partial|unknown|normal",
            "closest_match": "doc name or null",
            "confidence": "0.0-1.0",
            "matched_evidence": [],
            "contradicting_evidence": [],
            "explanation": "cite only evidence names present in the bundle",
        },
    }
    try:
        req = urllib.request.Request(
            ollama_url("/api/generate"),
            data=json.dumps(
                {
                    "model": model,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_ctx": 16384,
                        "temperature": 0,
                        "seed": 20260922,
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read())
        parsed = json.loads(out["response"])
        if out.get("prompt_eval_count") is not None:
            parsed["prompt_eval_count"] = int(out["prompt_eval_count"])
        return apply_consistency_checks(_validate(parsed), evidence_bundle, candidate_docs)
    except Exception as exc:
        if fallback == "legacy":
            return fallback_verdict(top1_name=top1, reason=str(exc))
        raise

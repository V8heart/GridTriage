"""Claude judge path: claude-* models go through the Anthropic SDK and the same validation as Ollama."""
import json
import sys
import types

import pytest

from pipeline import stage2_rag_judge as rj


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


def _fake_anthropic(monkeypatch, payload, stop_reason="end_turn"):
    calls = []

    class Client:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

        def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                                         content=[_Block(json.dumps(payload, ensure_ascii=False))])

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    return calls


BUNDLE = {"pattern": {"periodicity_strength": 0.9},
          "unexplainedness": {"evidence": {"declared_family_mismatch": True}}}
DOCS = [{"doc_id": "card:swma#관측 시그니처", "layer": "card", "card": "swma", "title": "SWMA", "text": "주기적 전력"}]


def test_claude_model_routes_to_api_and_validates(monkeypatch):
    calls = _fake_anthropic(monkeypatch, {"verdict": "attack", "threat_card": "swma",
                                          "cited_docs": ["card:swma#관측 시그니처"],
                                          "evidence_used": ["periodicity_strength"], "explanation": "주기 일치"})
    monkeypatch.setattr(rj, "_call_ollama", lambda *a, **k: pytest.fail("Ollama must not be called"))
    j = rj.judge(BUNDLE, DOCS, model="claude-opus-5-5")
    assert j.valid and j.verdict == "attack" and j.threat_card == "swma" and j.model == "claude-opus-5-5"
    req = calls[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"]["format"]["type"] == "json_schema"


def test_claude_refusal_becomes_invalid(monkeypatch):
    _fake_anthropic(monkeypatch, {}, stop_reason="refusal")
    j = rj.judge(BUNDLE, DOCS, model="claude-opus-5-5", retries=0)
    assert j.verdict == "invalid" and "declined" in j.errors[0]

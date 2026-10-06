"""Research agent safeguards (no network): grounded sources only, valid slots, no ungrounded drafts."""
from pipeline.research import agent
from pipeline.research.sources import Paper


def test_extract_drops_unfetched_sources_and_invalid_slots(monkeypatch):
    papers = [Paper("arxiv", "T", "abstract", "http://arxiv.org/abs/1")]
    monkeypatch.setattr(agent, "_call_ollama", lambda *a, **k: {
        "is_threat": True, "name": "Some Threat!", "observable_profile": {"level": ["high", "huge"], "bogus": ["x"]},
        "attack_ids": ["T1496", "T9999"], "sources": ["http://arxiv.org/abs/1", "http://evil.example"]})
    card = agent.extract_card({}, papers, model="m", host=None, known_ids={"T1496"})
    assert card["sources"] == ["http://arxiv.org/abs/1"]
    assert card["rejected_sources"] == ["http://evil.example"]
    assert card["observable_profile"] == {"level": ["high"]}
    assert card["attack_ids"] == ["T1496"]
    assert card["name"] == "some_threat"


def test_no_draft_without_grounded_source(tmp_path):
    card = {"is_threat": True, "name": "x", "sources": [], "observable_profile": {}, "attack_ids": [],
            "summary": "", "mechanism": "", "uncertainty": ""}
    assert agent.write_draft(card, {}, [], tmp_path) is None


def test_keyword_queries_are_generic():
    q = agent.keyword_queries({"level": "high", "form": "flat", "declared": "mismatch"})
    assert all(part.startswith("GPU") for part in q)
    assert not any("s-" in part for part in q)  # never session ids


def test_signature_uses_query_vocabulary():
    from pipeline.kb.query import PHRASES
    from pipeline.research.agent import signature

    text = signature({"form": ["square"]}, {"required": ["log_period_mismatch"], "supporting": [], "exclusion": ["flat_power"]})
    assert PHRASES["log_mismatch"] in text
    assert "다음은 아니다: " + PHRASES["flat_high"] in text


def test_requirement_gate_turns_forced_card_into_unknown():
    from pipeline.stage2_rag_judge import requirement_gate

    docs = [{"card": "burst_ramp_attack", "requires": ["high_ramp"], "excludes": ["flat_power"]},
            {"card": "cryptojacking", "requires": [], "excludes": ["period_match"]}]
    gate = requirement_gate("burst_ramp_attack", docs, {"high_ramp": False, "flat_power": True})
    assert gate == {"card": "burst_ramp_attack", "missing_required": ["high_ramp"], "exclusion_on": ["flat_power"]}
    assert requirement_gate("cryptojacking", docs, {"period_match": False}) is None
    assert requirement_gate("no_rules_card", docs, {}) is None

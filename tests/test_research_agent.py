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

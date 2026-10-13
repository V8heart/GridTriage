"""v5 operational loop: alarm -> unknown -> research -> pending case -> analyst approval -> memory reload."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pipeline.v5.live_v5 as live_v5
from pipeline.v5.case_store import CaseStore
from pipeline.v5.judge_v5 import enrolled_case_match
from pipeline.v5.raw_features import ALL_RAW


def _case_windows(center, n_win=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"session_id": f"s{seed}", "gpu_id": 0, "win": w, "declared_job_family": "training",
                          **{f: center + rng.normal(0, 0.1) for f in ALL_RAW}} for w in range(n_win)])


@pytest.fixture
def store(tmp_path):
    s = CaseStore(tmp_path / "cm")
    for i in range(12):
        s.add_case(f"seed-n{i}", _case_windows(0.0, seed=i), "normal", action="seed", analyst="system", reason="t",
                   source={"kind": "seed"})
    return s


def test_ledger_chain_and_tamper_detection(store):
    assert store.verify() == []
    assert len(store.active_ids()) == 12
    store.revoke("seed-n0", analyst="kim", reason="bad")
    assert "seed-n0" not in store.active_ids() and store.verify() == []
    with pytest.raises(ValueError):
        store.revoke("seed-n0", analyst="kim", reason="again")
    # editing a case file after approval is detected
    path = store.cases_dir / "seed-n1.csv"
    path.write_text(path.read_text().replace("normal", "swma", 1))
    assert any("changed after approval" in p for p in store.verify())


def test_ledger_edit_detected(store):
    lines = store.ledger_path.read_text().splitlines()
    e = json.loads(lines[3])
    e["label"] = "swma"
    lines[3] = json.dumps(e)
    store.ledger_path.write_text("\n".join(lines) + "\n")
    assert any("hash mismatch" in p for p in store.verify())


def test_approval_requires_analyst_and_reason(store):
    with pytest.raises(ValueError):
        store.add_case("x", _case_windows(1.0), "swma", action="enroll", analyst="", reason="r", source={})
    with pytest.raises(ValueError):
        store.add_case("seed-n2", _case_windows(1.0), "normal", action="enroll", analyst="k", reason="r", source={})


def test_enrolled_case_rule():
    case = {"nearest_case_votes": {"new_x": 0.8, "swma": 0.2}, "class_p": {"normal": 0.01, "swma": 0.3}}
    assert enrolled_case_match(case, frozenset({"new_x"})) == "new_x"  # single case: no fit reference yet
    assert enrolled_case_match(case, frozenset()) is None  # seed classes are left to the LLM + gate
    assert enrolled_case_match({**case, "class_p": {"new_x": 0.01}}, frozenset({"new_x"})) is None  # own fit rejected
    assert enrolled_case_match({**case, "nearest_case_votes": {"new_x": 0.4, "swma": 0.6}}, frozenset({"new_x"})) is None
    normal = {"nearest_case_votes": {"normal": 0.9}, "class_p": {"normal": 0.3}}
    assert enrolled_case_match({**normal, "declared_normal_p": 0.4}, frozenset({"normal"})) == "normal"
    assert enrolled_case_match({**normal, "declared_normal_p": 0.01}, frozenset({"normal"})) is None
    assert enrolled_case_match({**normal, "declared_normal_p": None}, frozenset({"normal"})) is None


def test_enrolled_classes_exclude_seed_and_revoked(store):
    assert store.enrolled_classes() == frozenset()
    store.add_case("e1", _case_windows(3.0, seed=50), "sponge_attack", action="enroll", analyst="k", reason="r",
                   source={})
    assert store.enrolled_classes() == {"sponge_attack"}
    store.revoke("e1", analyst="k", reason="wrong")
    assert store.enrolled_classes() == frozenset()


def _engine(store, monkeypatch, verdict="unknown"):
    calls = {"analyze": 0, "research": 0}

    def fake_analyze(bundle, windows, retriever, memory, **kw):
        calls["analyze"] += 1
        calls["classes"] = memory.classes
        return {"query": "q", "hits": [], "case": {"class_p": {"normal": 0.01}}, "channels": {},
                "judgment": {"verdict": verdict, "threat_card": None, "gate": {"rule": "b_unexplained_anomaly"},
                             "explanation": "e"}}

    def fake_research(bundle, channels, case, **kw):
        calls["research"] += 1
        return {"observed_tokens": ["power_flat"], "contrast": "c",
                "hypotheses": [{"name": "h1", "mechanism": "m", "score": 1.0, "support": 1.0, "contradictions": [],
                                "papers": []}]}

    monkeypatch.setattr(live_v5, "analyze_run_v5", fake_analyze)
    monkeypatch.setattr(live_v5, "segment_bundle", lambda part, cfg: {"bundle": {"declared_context": {}}})
    stage1 = SimpleNamespace(config={}, name="test")
    settings = live_v5.EngineSettingsV5(store_dir=store.root, queue_dir=store.root / "queue")
    eng = live_v5.LiveEngineV5(source=None, retriever=None, settings=settings, stage1=stage1, store=store,
                               researcher=fake_research)
    t0 = 1_000_000.0
    from collections import deque
    eng.state.rows[0] = deque({"t_epoch": t0 + i / 10, "gpu_id": 0, "power_w": 300.0 + (i % 7), "power_instant_w": 300.0,
                               "util_gpu_pct": 90.0, "fb_used_mb": 4000.0, "observed_n_procs": 1,
                               "declared_job_family": "training"} for i in range(1200))
    run = [{"t_end": t0 + 30 + 15 * i, "window_row": {}, "u_score": 1.0, "p_value": 0.01, "evidence_bool": {}}
           for i in range(5)]
    return eng, run, calls


def test_unknown_alarm_triggers_research_once_and_queues_a_case(store, monkeypatch):
    eng, run, calls = _engine(store, monkeypatch)
    eng._submit(0, run)
    eng._submit(0, run + [{**run[-1], "t_end": run[-1]["t_end"] + 15}])  # re-analysis of the same alarm
    eng.wait_idle()
    snap = eng.snapshot()
    assert calls["analyze"] == 2 and calls["research"] == 1
    aid = eng.alarm_id(0, run)
    job = snap["research"][aid]
    assert job["status"] == "done" and (store.root / "queue").exists()
    pend = store.pending()
    assert len(pend) == 1 and pend[0]["verdict"] == "unknown"
    assert pend[0]["research_queue"] == job["path"]


def test_no_research_when_not_unknown(store, monkeypatch):
    eng, run, calls = _engine(store, monkeypatch, verdict="normal")
    eng._submit(0, run)
    eng.wait_idle()
    assert calls["research"] == 0 and len(store.pending()) == 1


def test_approval_is_recorded_and_reloads_memory(store, monkeypatch):
    eng, run, calls = _engine(store, monkeypatch)
    eng._submit(0, run)
    eng.wait_idle()
    assert "new_x" not in calls["classes"]
    cid = store.pending()[0]["case_id"]
    entry = store.approve_pending(cid, "new_x", analyst="kim", reason="confirmed by host forensics")
    assert entry["prior"]["verdict"] == "unknown" and entry["prior"]["research_queue"]
    assert store.pending() == [] and store.verify() == []
    assert "new_x" in eng.memory().classes  # ledger changed -> memory rebuilt
    # a reviewed alarm is not queued again when re-analysed
    eng2, run2, _ = _engine(store, monkeypatch)
    eng2._submit(0, run2)
    eng2.wait_idle()
    assert store.pending() == []


def test_dismiss_is_recorded(store, monkeypatch):
    eng, run, _ = _engine(store, monkeypatch)
    eng._submit(0, run)
    eng.wait_idle()
    cid = store.pending()[0]["case_id"]
    e = store.dismiss_pending(cid, analyst="kim", reason="planned load test")
    assert e["action"] == "dismiss" and cid not in store.active_ids() and store.verify() == []


def test_escalation_is_sticky_across_reanalyses(store, monkeypatch):
    eng, run, calls = _engine(store, monkeypatch, verdict="unknown")
    eng._submit(0, run)
    eng.wait_idle()
    eng2, _, _ = _engine(store, monkeypatch, verdict="normal")  # same alarm id, later re-analysis says normal
    eng2._submit(0, run)
    eng2.wait_idle()
    pend = store.pending()[0]
    assert pend["verdict"] == "normal" and pend["escalated"] is True
    assert [h["verdict"] for h in pend["history"]] == ["unknown", "normal"]


def test_card_name_in_verdict_is_normalized():
    from pipeline.v5.judge_v5 import normalize_raw

    assert normalize_raw({"verdict": "swma"}, ["swma"]) == {"verdict": "attack", "threat_card": "swma"}
    assert normalize_raw({"verdict": "attack", "threat_card": "swma"}, ["swma"])["verdict"] == "attack"


def test_enrolled_match_survives_invalid_llm_output(monkeypatch):
    import pipeline.v5.judge_v5 as jv

    monkeypatch.setattr(jv, "_call_ollama", lambda *a, **k: {"verdict": "attack", "threat_card": "nope"})
    case = {"nearest_case_votes": {"new_x": 0.9}, "class_p": {"normal": 0.01}, "declared_normal_p": None}
    j = jv.judge_v5({"pattern": {}, "unexplainedness": {"evidence": {}}}, [], case, {}, enrolled=frozenset({"new_x"}))
    assert j.verdict == "attack" and j.threat_card == "new_x" and j.gate["rule"] == "c_enrolled_case_match"
    assert j.gate["llm_verdict"] == "invalid"
    j2 = jv.judge_v5({"pattern": {}, "unexplainedness": {"evidence": {}}}, [], case, {})
    assert j2.verdict == "invalid"

"""v5 console (console_v5/app.py): the analyst review section records decisions only with a name and reason."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.v5.case_store import CaseStore
from pipeline.v5.raw_features import ALL_RAW

APP = Path(__file__).resolve().parents[1] / "console_v5" / "app.py"
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _windows(center, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"session_id": f"s{seed}", "gpu_id": 0, "win": w, "declared_job_family": "training",
                          **{f: center + rng.normal(0, 0.1) for f in ALL_RAW}} for w in range(6)])


@pytest.fixture
def store(tmp_path):
    s = CaseStore(tmp_path / "cm")
    for i in range(3):
        s.add_case(f"seed-n{i}", _windows(0.0, i), "normal", action="seed", analyst="system", reason="t", source={})
    q = tmp_path / "q.md"
    q.write_text("### adversarial_resource_exhaustion (점수 2.2)\n- [correct] RECUR")
    s.save_pending("replay-x-gpu0-t1", _windows(5.0, 9), {
        "alarm_id": "0:1", "gpu_id": 0, "source": "replay x", "n_windows": 9, "verdict": "unknown",
        "threat_card": None, "escalated": True, "explanation": "e", "case_fit": {"normal": 0.01},
        "history": [{"n_windows": 5, "verdict": "unknown", "threat_card": None, "gate_rule": "a_card_case_mismatch"}],
        "research_queue": str(q)})
    return s


def _app(store):
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    next(t for t in at.text_input if t.label == "사례 저장소").set_value(str(store.root))
    at.run()
    assert not at.exception
    return at


def _decide(at, label=None, action=None, reason="host forensics confirmed"):
    if action:
        next(r for r in at.radio if r.label == "결정").set_value(action)
    if label:
        next(s for s in at.selectbox if s.label == "라벨").set_value(label)
    at.text_area[0].set_value(reason)
    next(b for b in at.button if b.label == "결정 기록").click()
    at.run()


def test_review_shows_pending_case_and_research(store):
    at = _app(store)
    assert any(t.label == "대기 사례 (1)" for t in at.tabs)
    assert any("adversarial_resource_exhaustion" in m.value for m in at.markdown)


def test_approval_needs_analyst_name(store):
    at = _app(store)
    _decide(at, label="sponge_attack")
    assert any("분석가 이름" in e.value for e in at.error)
    assert all(e["action"] == "seed" for e in store.ledger())


def test_approval_is_recorded_with_name_and_reason(store):
    at = _app(store)
    next(t for t in at.text_input if t.label.startswith("분석가 이름")).set_value("kim")
    at.run()
    _decide(at, label="sponge_attack")
    e = store.ledger()[-1]
    assert (e["action"], e["label"], e["analyst"], e["reason"]) == ("enroll", "sponge_attack", "kim",
                                                                    "host forensics confirmed")
    assert e["prior"]["escalated"] is True and store.verify() == [] and store.pending() == []


def test_dismissal_is_recorded(store):
    at = _app(store)
    next(t for t in at.text_input if t.label.startswith("분석가 이름")).set_value("kim")
    at.run()
    _decide(at, action="기각 (등록하지 않음)", reason="planned load test")
    e = store.ledger()[-1]
    assert e["action"] == "dismiss" and e["analyst"] == "kim" and store.pending() == []

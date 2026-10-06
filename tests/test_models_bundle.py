"""A fresh clone (no dataset/real) must run the official pipeline from the frozen models/ copy."""
import json

import pipeline.v4_core as core


def test_official_stage1_loads_from_models_without_local_refit(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "REFIT", tmp_path / "no_refit")
    assert core.model_root(*core.OFFICIAL_STAGE1) == core.MODELS
    s1 = core.load_stage1()
    assert s1.config["alpha"] == 0.05
    assert s1.calibration["evidence_surprisal_weights"]["inst_power_unsteady"] > 1.0


def test_frozen_bundle_matches_the_documented_gate():
    gate = json.loads((core.MODELS / "session_gate.json").read_text())
    assert gate["min_run_windows"] == 5 and gate["stage1"] == "round5b"


def test_frozen_bundle_has_no_session_identifiers():
    for path in core.MODELS.rglob("*.json"):
        assert '"s-' not in path.read_text(), path  # fitted statistics only, no capture/session ids

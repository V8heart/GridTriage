from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.corpus_schema import parse_corpus_v2, validate_all_corpus_docs
from pipeline.rag_analyzer import CORPUS_DIR
from pipeline.run_pipeline import _segment_candidate_frame
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_llm import (
    apply_consistency_checks,
    distinction_excerpt,
    fallback_verdict,
    judge,
    ollama_url,
    Stage2Verdict,
)


def test_ollama_url_reads_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://10.41.12.52:11434")
    assert ollama_url("/api/generate") == "http://10.41.12.52:11434/api/generate"
    monkeypatch.setenv("OLLAMA_HOST", "10.41.12.52:11434")
    assert ollama_url("api/generate") == "http://10.41.12.52:11434/api/generate"


def test_evidence_bundle_excludes_raw_timeseries():
    bundle = build_evidence_bundle({
        "window_id": "w1",
        "power_w": [1, 2, 3],
        "util_gpu_pct": [4, 5, 6],
        "impact_components": {"swing_frac_tdp": 0.2},
        "evidence_bool": {"period_match": True},
    })
    assert "power_w" not in str(bundle)
    assert "util_gpu_pct" not in str(bundle)


def test_corpus_v2_schema_validates_all_docs():
    assert validate_all_corpus_docs(CORPUS_DIR) == []


def test_stage2_deterministic_given_fixed_evidence():
    docs = [parse_corpus_v2(CORPUS_DIR / "benign_periodic.md")]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {"period_match": True, "period_mismatch": False}})
    verdicts = [judge(bundle, docs, backend="stub", retriever_top1="benign_periodic") for _ in range(3)]
    assert len({(v.verdict, v.closest_match) for v in verdicts}) == 1
    assert verdicts[0].verdict == "partial"
    assert verdicts[0].closest_match == "benign_periodic"


def test_stage2_llm_output_schema_enforced_by_fallback():
    assert fallback_verdict(top1_name=None, reason="bad schema").fallback_used is True


def test_fallback_uses_retriever_top1_not_first_doc():
    docs = [
        parse_corpus_v2(CORPUS_DIR / "benign_periodic.md"),
        parse_corpus_v2(CORPUS_DIR / "swma.md"),
    ]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {}})
    verdict = judge(bundle, docs, backend="stub", retriever_top1="swma")
    assert verdict.fallback_used is True
    assert verdict.closest_match == "swma"


def test_explanation_only_cites_declared_evidence_names():
    docs = [parse_corpus_v2(CORPUS_DIR / "benign_periodic.md")]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {"period_match": True, "period_mismatch": False}})
    verdict = judge(bundle, docs, backend="stub", retriever_top1="benign_periodic")
    allowed = set(verdict.matched_evidence) | set(verdict.contradicting_evidence)
    for item in verdict.matched_evidence:
        assert item in allowed


def test_segment_pattern_is_window_median_not_first():
    frame = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "gpu_id": 0,
                "start": 0,
                "end": 300,
                "window_id": "a",
                "window_start_s": 0.0,
                "window_end_s": 30.0,
                "evidence": {
                    "dominant_freq_hz": 0.10,
                    "periodicity_strength": 0.2,
                    "duty_regularity": 0.2,
                    "swing_abs_w": 10.0,
                    "mean_w": 100.0,
                    "step_period_s": 1.0,
                    "n_transitions": 2,
                    "ramp_p95_w_per_s": 5.0,
                },
                "evidence_bool": {"flat_power": True},
                "u_score": 1.0,
            },
            {
                "session_id": "s1",
                "gpu_id": 0,
                "start": 150,
                "end": 450,
                "window_id": "b",
                "window_start_s": 15.0,
                "window_end_s": 45.0,
                "evidence": {
                    "dominant_freq_hz": 0.50,
                    "periodicity_strength": 0.8,
                    "duty_regularity": 0.8,
                    "swing_abs_w": 90.0,
                    "mean_w": 300.0,
                    "step_period_s": 3.0,
                    "n_transitions": 10,
                    "ramp_p95_w_per_s": 50.0,
                },
                "evidence_bool": {"flat_power": True},
                "u_score": 2.0,
            },
        ]
    )
    out = _segment_candidate_frame(frame, {"stage2": {"evidence_rate_min": 0.6}})
    pattern = out.iloc[0]["pattern"]
    assert pattern["dominant_freq_hz"] == pytest.approx(0.30)
    assert pattern["step_period_s"] == pytest.approx(2.0)
    assert pattern["n_transitions_per_min"] == pytest.approx(12.0)
    assert "ramp_max" not in pattern
    assert "ramp_max_w_per_s" not in pattern
    bundle = build_evidence_bundle(
        {
            "window_id": "seg",
            "evidence_bool": {"flat_power": True},
            "pattern": pattern,
        }
    )
    assert bundle["pattern"]["dominant_freq_hz"] == pytest.approx(0.30)
    assert bundle["unexplainedness"]["evidence"]["period_comparable"] is True
    assert bundle["unexplainedness"]["evidence"]["log_period_mismatch"] is True


def test_distinction_excerpt_keeps_only_구분_포인트():
    text = distinction_excerpt(parse_corpus_v2(CORPUS_DIR / "swma.md").body)
    assert "구분 포인트" in text
    assert "Bit2Watt (Ji" not in text


def test_consistency_known_benign_and_normal_attack_become_partial():
    docs = [
        parse_corpus_v2(CORPUS_DIR / "benign_periodic.md"),
        parse_corpus_v2(CORPUS_DIR / "swma.md"),
    ]
    bundle = build_evidence_bundle(
        {"window_id": "w1", "evidence_bool": {"period_match": True, "log_period_mismatch": True}}
    )
    known_benign = apply_consistency_checks(
        Stage2Verdict("known", "benign_periodic", 0.9, ["period_match", "invented"], [], "x"),
        bundle,
        docs,
    )
    assert known_benign.verdict == "partial"
    assert "invented" not in known_benign.matched_evidence
    normal_attack = apply_consistency_checks(
        Stage2Verdict("normal", "swma", 0.9, [], [], "x"),
        bundle,
        docs,
    )
    assert normal_attack.verdict == "partial"
    outside = apply_consistency_checks(
        Stage2Verdict("known", "not_a_doc", 0.9, [], [], "x"),
        bundle,
        docs,
    )
    assert outside.verdict == "partial"


def _all_docs():
    return [parse_corpus_v2(path) for path in sorted(CORPUS_DIR.glob("*.md"))]


def _flags(names: dict[str, bool]):
    return {"window_id": "w", "unexplainedness": {"evidence": names}}


def _stub(names: dict[str, bool], top1: str = "burst_ramp_attack"):
    return judge(_flags(names), _all_docs(), backend="stub", retriever_top1=top1)


def test_reference_window_matches_only_swma():
    verdict = _stub({"high_ramp": True, "log_period_mismatch": True, "progress_log_available": True})
    assert verdict.verdict == "known"
    assert verdict.closest_match == "swma"
    assert verdict.fallback_used is False


def test_reference_window_with_family_mismatch_still_swma():
    verdict = _stub(
        {
            "high_ramp": True,
            "log_period_mismatch": True,
            "progress_log_available": True,
            "declared_family_mismatch": True,
        }
    )
    assert verdict.verdict == "known"
    assert verdict.closest_match == "swma"


def test_single_flags_do_not_match_a_checklist():
    for names in ({"high_ramp": True}, {"cross_job_sync": True}, {"period_match": True}):
        verdict = _stub(names, top1="llmjacking")
        assert verdict.fallback_used is True
        assert verdict.verdict == "partial"
        assert verdict.closest_match == "llmjacking"


def test_log_period_mismatch_alone_is_swma():
    verdict = _stub({"log_period_mismatch": True})
    assert verdict.verdict == "known"
    assert verdict.closest_match == "swma"


def test_log_period_mismatch_with_sync_and_loose_period_match_is_coordinated():
    verdict = _stub(
        {"log_period_mismatch": True, "cross_job_sync": True, "period_match": True, "high_ramp": True}
    )
    assert verdict.verdict == "known"
    assert verdict.closest_match == "coordinated_multi_gpu"


def test_period_match_with_log_is_partial_benign_over_normal():
    verdict = _stub({"period_match": True, "progress_log_available": True})
    assert verdict.verdict == "partial"
    assert verdict.closest_match == "benign_periodic"
    assert "normal_workloads" in verdict.explanation
    assert verdict.fallback_used is False


def test_loose_period_match_with_tight_mismatch_is_swma():
    verdict = _stub({"period_match": True, "log_period_mismatch": True, "progress_log_available": True})
    assert verdict.verdict == "known"
    assert verdict.closest_match == "swma"


def test_ramp_and_unexplained_changepoint_is_burst():
    verdict = _stub({"high_ramp": True, "unexplained_changepoint": True})
    assert verdict.verdict == "known"
    assert verdict.closest_match == "burst_ramp_attack"


def test_ramp_and_family_mismatch_is_partial_llmjacking():
    verdict = _stub({"high_ramp": True, "declared_family_mismatch": True})
    assert verdict.verdict == "partial"
    assert verdict.closest_match == "llmjacking"
    assert "hidden_ml_training" in verdict.explanation


VISIBLE_STAGE2 = (
    Path(__file__).resolve().parents[1]
    / "dataset/real/pipeline/refit/round3/stage2_p3_stub_visible.json"
)

# Frozen stub verdicts after the checklist split. closest_match is None on a tie.
VISIBLE_STUB_EXPECT = {
    "s-02c5189d7f553551:0:0:300": ("partial", None),
    "s-03ea7ec9a9bc7f6d:0:0:300": ("known", "hidden_ml_training"),
    "s-094d82425da94890:0:0:300": ("known", "swma"),
    "s-163297c80327844e:0:1350:1650": ("partial", "cryptojacking"),
    "s-163297c80327844e:0:3600:3900": ("partial", "cryptojacking"),
    "s-163297c80327844e:0:4650:4950": ("partial", "cryptojacking"),
    "s-163297c80327844e:0:5400:5700": ("partial", "cryptojacking"),
    "s-163297c80327844e:0:6450:6750": ("partial", "cryptojacking"),
    "s-163297c80327844e:0:6900:7200": ("partial", None),
    "s-2b6a98b47d629286:0:0:300": ("known", "hidden_ml_training"),
    "s-2d84528e7aee2ba8:0:0:300": ("partial", "llmjacking"),
    "s-491c04c15bf1744a:0:150:450": ("partial", "hidden_ml_training"),
    "s-548b4ba63139ca34:0:0:300": ("partial", "ltma"),
    "s-548b4ba63139ca34:0:4200:4500": ("partial", "normal_workloads"),
    "s-5cc533e92d4d6a4e:0:150:450": ("partial", None),
    "s-5cc533e92d4d6a4e:0:5400:5700": ("partial", None),
    "s-61f7d0d920907cb1:0:0:300": ("partial", "swma"),
    "s-6eb35312282ed125:0:5250:5550": ("partial", None),
    "s-6eb35312282ed125:0:6600:6900": ("partial", "burst_ramp_attack"),
    "s-8314a2f2da64e776:0:0:300": ("known", "coordinated_multi_gpu"),
    "s-8314a2f2da64e776:1:0:300": ("known", "coordinated_multi_gpu"),
    "s-8feb3a509125d970:0:0:300": ("partial", "ltma"),
    "s-915bea923b9e2019:0:0:300": ("partial", None),
    "s-915bea923b9e2019:1:0:300": ("partial", None),
    "s-9f66235f40fd65e3:0:0:300": ("partial", "ltma"),
    "s-9f66235f40fd65e3:0:1800:2100": ("partial", "normal_workloads"),
    "s-9f66235f40fd65e3:0:3900:4200": ("partial", "normal_workloads"),
    "s-9f66235f40fd65e3:0:4350:4650": ("partial", "normal_workloads"),
    "s-a4751177d4e7c50e:0:0:300": ("partial", None),
    "s-a595e33152d4bc59:0:0:300": ("known", "ltma"),
    "s-ba6b6bda088c746d:0:0:300": ("partial", "ltma"),
    "s-bf6900c34500344e:0:0:300": ("partial", None),
    "s-bf6900c34500344e:1:3300:3600": ("known", "burst_ramp_attack"),
    "s-bf6900c34500344e:1:5250:5550": ("partial", None),
    "s-c6793e53d1bc9f84:0:0:300": ("partial", None),
    "s-ca28a669ce3c07c7:0:0:300": ("known", "hidden_ml_training"),
    "s-d4dbf93e3f6b141a:0:0:300": ("known", "hidden_ml_training"),
    "s-d74ed92368a694b1:0:0:300": ("partial", "normal_workloads"),
    "s-df6fda03fb93c615:0:0:300": ("partial", None),
    "s-e20f23bb1450db38:0:0:300": ("partial", None),
}


@pytest.mark.skipif(not VISIBLE_STAGE2.is_file(), reason="real visible stage2 dump absent")
def test_visible_40_segment_stub_snapshot():
    rows = json.loads(VISIBLE_STAGE2.read_text(encoding="utf-8"))
    docs = _all_docs()
    got = {}
    for row in rows:
        top = (row.get("rag_top") or [[None]])[0][0]
        verdict = judge(row["stage2_evidence_bundle"], docs, backend="stub", retriever_top1=top)
        got[row["window_id"]] = (verdict.verdict, verdict.closest_match)
        if row["window_id"] in {
            "s-915bea923b9e2019:0:0:300",
            "s-915bea923b9e2019:1:0:300",
        }:
            assert verdict.verdict == "partial"
            assert verdict.closest_match is None
            assert "coordinated_multi_gpu" in verdict.explanation
            assert "ltma" in verdict.explanation
    assert got["s-8314a2f2da64e776:0:0:300"] == ("known", "coordinated_multi_gpu")
    assert got == VISIBLE_STUB_EXPECT


def test_bundle_cross_job_sync_uses_noidle_index():
    bundle = build_evidence_bundle(
        {
            "window_id": "w1",
            "evidence_bool": {"cross_job_sync": True},
            "cross_job_sync_index": 0.9,
            "cross_job_sync_index_noidle": 0.0,
        },
        config={"tau_sync": 0.8},
    )
    assert bundle["unexplainedness"]["evidence"]["cross_job_sync"] is False

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bit2watt_impl.physics.simulation import power_series_to_relative_events
from dataset.synth_attacks import cryptojacking, swma
from pipeline.features import compute_window_features
from pipeline import run_pipeline
from pipeline.run_pipeline import run, stage1_screen


def _base_run_args(**overrides):
    args = dict(
        telemetry="",
        baseline_mean=120.0,
        sample_hz=None,
        window=200,
        stride=100,
        z_threshold=2.5,
        rag_backend="tfidf",
        llm_backend="stub",
        llm_model="unused",
        physics_validate=False,
        physics_timeout_s=60.0,
        physics_test_system="kundur_ieeest",
        out=None,
    )
    args.update(overrides)
    return Namespace(**args)


def test_feature_masks_keep_util_aligned():
    power = np.array([100, 101, np.nan, 103, 104, 105, 106, 107, 108], dtype=float)
    util = np.arange(len(power), dtype=float)
    feats = compute_window_features(power, util, sample_hz=10)
    assert feats["util_mean"] == np.mean(util[np.isfinite(power)])


def test_stage1_detects_flat_high_load():
    frame = cryptojacking(n=400)
    screened = stage1_screen(
        frame,
        baseline_mean_w=120,
        sample_hz=10,
        window=200,
        stride=100,
    )
    assert screened["is_candidate"].all()
    assert screened["candidate_reasons"].str.contains("persistent_high_load").all()


def test_stage1_detects_mechanical_swma():
    frame = swma(n=400)
    screened = stage1_screen(
        frame,
        baseline_mean_w=120,
        sample_hz=10,
        window=200,
        stride=100,
    )
    assert screened["candidate_reasons"].str.contains("mechanical_periodicity").all()


def test_schema_groups_prevent_cross_session_windows():
    first = pd.DataFrame({"power_w": np.full(200, 100.0)})
    second = pd.DataFrame({"power_w": np.full(200, 300.0)})
    assert len(stage1_screen(first, window=200, stride=100)) == 1
    assert len(stage1_screen(second, window=200, stride=100)) == 1


def test_pipeline_emits_tier_b_join_fields(tmp_path, capsys):
    telemetry = tmp_path / "swma.csv"
    swma(n=400).to_csv(telemetry, index=False)
    results = run(_base_run_args(telemetry=str(telemetry)))
    capsys.readouterr()
    assert results
    result = results[0]
    assert result["attack_id"]
    assert result["window_id"]
    assert result["threat_id"] == result["verdict"]["closest_match"]
    assert 0 <= result["anomaly_score"] <= 1
    assert result["score_version"] == "cyber-stage1-or-v1"
    assert "physics_osc_std" not in result


def test_power_series_to_relative_events_clips_and_downsamples():
    t = np.arange(40) / 10.0
    power = 200.0 + 80.0 * np.sin(2 * np.pi * 0.5 * t)
    events, meta = power_series_to_relative_events(
        power, sample_hz=10.0, max_amp_frac=0.15, min_interval_s=0.1
    )
    assert meta["peak_raw_delta"] > 0.15
    assert meta["amp_scale"] < 1.0
    assert all(abs(delta) <= 0.15 + 1e-9 for _, delta in events)
    assert len(events) < len(power)
    assert events[-1][1] == 0.0


def test_pipeline_physics_validate_fills_fields(tmp_path, capsys):
    telemetry = tmp_path / "swma_small.csv"
    swma(n=220).to_csv(telemetry, index=False)
    fake = {
        "test_system": "kundur_ieeest",
        "mode": "observed_waveform_replay",
        "converged": True,
        "failure_reason": "",
        "osc_std": 0.0123,
        "rocof_hz_s": 0.0456,
        "dominant_freq_hz": 0.5,
        "osc_ptp": 0.03,
    }
    with patch(
        "bit2watt_impl.physics.simulation.inject_observed_waveform_with_timeout",
        return_value=fake,
    ) as mocked:
        results = run(
            _base_run_args(
                telemetry=str(telemetry),
                window=200,
                stride=200,
                physics_validate=True,
                physics_timeout_s=5.0,
            )
        )
    capsys.readouterr()
    assert mocked.called
    assert results
    result = results[0]
    assert "verdict" in result
    assert result["physics_converged"] is True
    assert result["physics_osc_std"] == 0.0123
    assert result["physics_rocof_hz_s"] == 0.0456
    assert result["physics_dominant_freq_hz"] == 0.5
    assert result["physics_osc_ptp"] == 0.03
    assert result["physics_mode"] == "observed_waveform_replay"


def test_pipeline_physics_failure_does_not_abort(tmp_path, capsys):
    telemetry = tmp_path / "swma_fail.csv"
    swma(n=220).to_csv(telemetry, index=False)
    fake = {
        "test_system": "kundur_ieeest",
        "mode": "observed_waveform_replay",
        "converged": False,
        "failure_reason": "TDS entered busted state; fresh replay required",
        "osc_std": None,
        "rocof_hz_s": None,
        "dominant_freq_hz": None,
        "osc_ptp": None,
    }
    with patch(
        "bit2watt_impl.physics.simulation.inject_observed_waveform_with_timeout",
        return_value=fake,
    ):
        results = run(
            _base_run_args(
                telemetry=str(telemetry),
                window=200,
                stride=200,
                physics_validate=True,
            )
        )
    capsys.readouterr()
    assert results[0]["physics_converged"] is False
    assert results[0]["physics_osc_std"] is None
    assert results[0]["verdict"]


def test_run_flushes_stage2_after_each_session(tmp_path):
    first = swma(n=220)
    second = cryptojacking(n=220)
    telemetry = tmp_path / "two_sessions.csv"
    pd.concat([first, second], ignore_index=True).to_csv(telemetry, index=False)
    out = tmp_path / "stage2.json"
    snapshots = []
    orig = run_pipeline._write_stage2_results

    def spy(path, results):
        orig(path, results)
        if path is not None:
            snapshots.append([row["session_id"] for row in results])

    with patch.object(run_pipeline, "_write_stage2_results", side_effect=spy):
        results = run(
            _base_run_args(
                telemetry=str(telemetry),
                window=200,
                stride=200,
                out=str(out),
            )
        )

    assert results
    session_ids = [row["session_id"] for row in results]
    assert len(set(session_ids)) == 2
    session_flushes = [snap for snap in snapshots if snap]
    assert any(len(set(snap)) == 1 for snap in session_flushes)
    assert any(len(set(snap)) == 2 for snap in session_flushes)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [row["session_id"] for row in saved] == session_ids


def test_run_keeps_finished_session_if_later_session_fails(tmp_path):
    first = swma(n=220)
    second = cryptojacking(n=220)
    telemetry = tmp_path / "two_sessions_crash.csv"
    pd.concat([first, second], ignore_index=True).to_csv(telemetry, index=False)
    out = tmp_path / "stage2.json"
    flushes = {"n": 0}
    orig = run_pipeline._write_stage2_results

    def spy(path, results):
        orig(path, results)
        flushes["n"] += 1
        if flushes["n"] == 1:
            raise RuntimeError("stop after first session")

    with patch.object(run_pipeline, "_write_stage2_results", side_effect=spy):
        with pytest.raises(RuntimeError, match="stop after first session"):
            run(
                _base_run_args(
                    telemetry=str(telemetry),
                    window=200,
                    stride=200,
                    out=str(out),
                )
            )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved
    assert len({row["session_id"] for row in saved}) == 1

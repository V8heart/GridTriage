"""Console physics link: amplitude rule, bridge caching/failure, facility forcing, one replay per alarm."""
import numpy as np
import pandas as pd
import pytest

from bit2watt_impl.physics.simulation import FACILITY_BUS_CAP, facility_bus_fraction
from pipeline import physics_bridge as pb
from pipeline.grid_monitor import CsvGridSource, facility_forcing


def _square(seconds=60, hz=10, period=2.0, lo=250.0, hi=400.0, phase=0.0):
    t = np.arange(seconds * hz) / hz
    return t, np.where(((t + phase) % period) < period / 2, hi, lo)


def test_facility_rule_matches_brief():
    _, p = _square()  # swing 150 W around the mean -> ±75 W
    bus, meta = facility_bus_fraction(p, p0_bus=50.0, p_system=1000.0, tdp_w=450.0, participation=0.5,
                                      facility_share=0.02)
    assert meta["injected_frac_of_system_load"] == pytest.approx(75 / 450 * 0.5 * 0.02)
    assert np.max(np.abs(bus)) == pytest.approx(75 / 450 * 0.5 * 0.02 * 1000 / 50)
    assert not meta["clipped"]
    big, meta = facility_bus_fraction(p, p0_bus=1.0, p_system=1000.0, tdp_w=450.0, participation=1.0,
                                      facility_share=0.1)
    assert meta["clipped"] and np.max(np.abs(big)) == pytest.approx(FACILITY_BUS_CAP)


def test_mode_overlap_uses_harmonics():
    modes = [{"freq_hz": 1.0, "damping_ratio": 0.05}]
    assert pb.mode_overlap(0.5, modes)["harmonic"] == 2
    assert pb.mode_overlap(0.3, modes) is None


def test_replay_caches_and_reports_failures():
    calls = []

    def runner(req):
        calls.append(req)
        return {"converged": True, "osc_std": 1e-4}

    _, p = _square(period=2.0)
    settings = pb.PhysicsSettings(facility_share=0.013)
    a = pb.replay(p, 10.0, settings, runner=runner)
    b = pb.replay(p, 10.0, settings, runner=runner)
    assert a is b and len(calls) == 1
    assert a["forcing_freq_hz"] == pytest.approx(0.5, abs=0.02)
    assert calls[0]["facility"]["facility_share"] == 0.013

    def broken(req):
        raise RuntimeError("tds diverged")

    out = pb.replay(p * 1.01, 10.0, settings, runner=broken)
    assert out["converged"] is False and "tds diverged" in out["failure_reason"]


def _rows(sync: bool):
    rows = {}
    for gpu in (0, 1):
        t, p = _square(phase=0.0 if sync or gpu == 0 else 1.0)
        rows[gpu] = [{"t_epoch": 1e6 + ti, "power_w": pi, "power_instant_w": pi} for ti, pi in zip(t, p)]
    return rows


def test_facility_forcing_sees_band_and_sync():
    synced, opposed = facility_forcing(_rows(True)), facility_forcing(_rows(False))
    assert synced["band_peak_hz"] == pytest.approx(0.5, abs=0.05)
    assert synced["gpu_band_corr"] > 0.9 > opposed["gpu_band_corr"]
    assert synced["band_amp_pu"] > 5 * opposed["band_amp_pu"]  # anti-phase GPUs cancel at the facility


def test_csv_grid_source(tmp_path):
    path = tmp_path / "grid.csv"
    path.write_text("t_epoch,freq_hz,voltage_v\n0,60.00,230\n1,59.98,229\n2,59.90,228\n3,59.95,229\n")
    snap = CsvGridSource(path).snapshot(2.0)
    assert snap["freq_hz"] == pytest.approx(59.90)
    assert snap["rocof_max_hz_s"] == pytest.approx(0.08)
    assert snap["voltage_v"] == 228


def test_engine_replays_once_per_alarm(monkeypatch):
    live = pytest.importorskip("pipeline.live")
    from pipeline.v4_core import OFFICIAL_STAGE1, model_root

    if not (model_root(*OFFICIAL_STAGE1) / OFFICIAL_STAGE1[0] / OFFICIAL_STAGE1[1] / "cohort_baseline_v2.json").exists():
        pytest.skip("official Stage1 artifacts missing")
    eng = live.LiveEngine(source=None, retriever=None,
                          settings=live.EngineSettings(trigger_windows=3, physics=pb.PhysicsSettings()))
    submitted = []
    monkeypatch.setattr(eng, "_submit", lambda gpu, run: None)
    monkeypatch.setattr(eng, "_submit_physics", lambda gpu, run: submitted.append(eng.alarm_id(gpu, run)))
    for i in range(10):
        eng._track_run({"gpu_id": 0, "is_candidate": True, "t_end": 100.0 + 15 * i})
    assert submitted == ["0:100.0"]

    eng2 = live.LiveEngine(source=None, retriever=None, settings=live.EngineSettings(physics=pb.PhysicsSettings()))
    eng2.physics_runner = lambda req: {"converged": True, "osc_std": 2e-4}
    t, p = _square(seconds=90)
    eng2.ingest([{"t_epoch": 1e6 + ti, "gpu_id": 0, "power_w": pi, "power_instant_w": pi} for ti, pi in zip(t, p)])
    run = [{"t_end": 1e6 + 30 + 15 * i} for i in range(4)]
    eng2._submit_physics(0, run)
    eng2._physics_pool.shutdown(wait=True)
    res = eng2.snapshot()["physics"][eng2.alarm_id(0, run)]
    assert res["status"] == "done" and res["osc_std"] == 2e-4
    assert res["forcing_freq_hz"] == pytest.approx(0.5, abs=0.02)


def test_measured_indicators_match_simulated_panel_fields():
    from pipeline.grid_monitor import grid_indicators

    t = np.arange(0, 60, 1 / 30)
    f = 60 + 0.01 * np.sin(2 * np.pi * 0.4 * t) + 0.0005 * t / 60  # 0.4 Hz oscillation on a slow drift
    g = grid_indicators(t, f, 60.0)
    assert g["osc_freq_hz"] == pytest.approx(0.4, abs=0.02)
    assert g["swing_mhz"] == pytest.approx(20.0, abs=1.5)
    assert g["rocof_max_hz_s"] == pytest.approx(2 * np.pi * 0.4 * 0.01, rel=0.25)  # peak slope, 0.5 s window
    slow = grid_indicators(t[::30], f[::30], 60.0)  # 1 sample/s (typical UPS SNMP polling)
    assert slow["osc_freq_hz"] is None and slow["sample_hz"] == pytest.approx(1.0)


def test_relative_time_file_loops_with_data_time(tmp_path):
    t = np.arange(0, 120, 0.1)
    path = tmp_path / "pcc.csv"
    pd.DataFrame({"t_s": t, "freq_hz": 60 + 0.005 * np.sin(2 * np.pi * 0.5 * t)}).to_csv(path, index=False)
    src = CsvGridSource(path, horizon_s=60)
    first = src.snapshot(1_790_000_000.0)  # any epoch: the file is anchored on first use
    assert first["osc_freq_hz"] == pytest.approx(0.5, abs=0.02)
    assert src.snapshot(1_790_000_000.0 + 500)["freq_hz"] is not None  # past the file end: loops

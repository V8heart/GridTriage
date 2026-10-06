"""Live engine: window scoring cadence must not depend on how rows are batched."""
import numpy as np
import pytest

import pipeline.live as live
from pipeline.v4_core import OFFICIAL_STAGE1, model_root

pytestmark = pytest.mark.skipif(
    not (model_root(*OFFICIAL_STAGE1) / OFFICIAL_STAGE1[0] / OFFICIAL_STAGE1[1] / "cohort_baseline_v2.json").exists(),
    reason="official Stage1 artifacts missing (models/ or dataset/real refit)",
)


def _rows(seconds=100, hz=10):
    rng = np.random.default_rng(0)
    out = []
    for i in range(seconds * hz):
        t = 1_000_000.0 + i / hz
        for gpu in (0, 1):
            out.append({"t_epoch": t, "gpu_id": gpu, "gpu_model": "NVIDIA GeForce RTX 4090",
                        "power_w": 300 + rng.normal(0, 2), "power_instant_w": 300 + rng.normal(0, 5),
                        "util_gpu_pct": 95.0, "observed_n_procs": 1,
                        **live.declared_fields("llm_finetune")})
    return out


def _engine(monkeypatch):
    monkeypatch.setattr(live, "analyze_segment", lambda *a, **k: {"query": "", "hits": [], "judgment": {}})
    return live.LiveEngine(source=None, retriever=None)


def test_one_batch_and_ticks_score_the_same_windows(monkeypatch):
    rows = _rows()
    batch = _engine(monkeypatch)
    batch.ingest(rows)
    ticked = _engine(monkeypatch)
    for i in range(0, len(rows), 2):
        ticked.ingest(rows[i:i + 2])
    expected = 2 * (int((100 - 30) / 15) + 1)
    assert len(batch.state.windows) == len(ticked.state.windows) >= expected - 2

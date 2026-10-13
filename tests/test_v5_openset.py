"""Deterministic parts of the v5 open-set path (no LLM, no network, no captures)."""
import numpy as np
import pandas as pd

from pipeline.research.sources import Paper
from pipeline.v5.exemplar_kb import CaseMemory, conformal_p
from pipeline.v5.judge_v5 import open_set_gate
from pipeline.v5.raw_features import ALL_RAW, class_of, window_features
from pipeline.v5.research_v5 import Hypothesis, falsify, lexical_gate, observed_tokens


def _windows(cls, center, n_sessions=3, n_win=6, seed=0, family="training"):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for w in range(n_win):
            rows.append({"session_id": f"{cls}-{seed}-{s}", "gpu_id": 0, "cls": cls, "declared_job_family": family,
                         **{f: center + rng.normal(0, 0.1) for f in ALL_RAW}})
    return pd.DataFrame(rows)


def test_conformal_p_bounds():
    ref = np.array([1.0, 2.0, 3.0])
    assert conformal_p(10.0, ref) == 0.25
    assert conformal_p(0.0, ref) == 1.0


def test_case_memory_fits_own_class_and_rejects_far_segment():
    # >= 12 sessions per class: the pooled conformal reference needs > 20 values to reach p < 0.05
    mem = CaseMemory(pd.concat([_windows("normal", 0.0, n_sessions=12), _windows("swma", 5.0, n_sessions=12, seed=1)]))
    near = _windows("normal", 0.0, n_sessions=1, seed=7)
    far = _windows("x", 50.0, n_sessions=1, seed=8)
    s_near, s_far = mem.score(near, "training"), mem.score(far, "training")
    assert max(s_near["nearest_case_votes"], key=s_near["nearest_case_votes"].get) == "normal"
    assert s_near["class_p"]["normal"] > 0.05
    assert all(p < 0.05 for p in s_far["class_p"].values())
    assert s_far["declared_normal_key"] == "normal@training"


def test_gate_rules():
    case = {"class_p": {"cryptojacking": 0.01, "normal": 0.01, "swma": 0.5}, "declared_normal_p": 0.01}
    assert open_set_gate("attack", "cryptojacking", case)["rule"] == "a_card_case_mismatch"
    assert open_set_gate("attack", "swma", case) is None
    assert open_set_gate("attack", "sponge_attack", case) is None  # card without cases is not gated
    assert open_set_gate("normal", "normal_workloads", case) is None
    assert open_set_gate("normal", "normal_workloads", case, normal_rule=True)["rule"] == "b_unexplained_anomaly"
    ok = {"class_p": {"normal": 0.3}, "declared_normal_p": 0.01}
    assert open_set_gate("normal", None, ok, normal_rule=True) is None


def test_window_features_and_class_of():
    t = np.arange(300)
    w = pd.DataFrame({"power_w": 200 + 100 * (t // 10 % 2), "power_instant_w": 200 + 100 * (t // 10 % 2),
                      "util_gpu_pct": 50.0 + 0 * t, "fb_used_mb": 2048.0 + 0 * t, "observed_n_procs": 1})
    f = window_features(w)
    assert set(ALL_RAW) <= set(f)
    assert abs(f["p_fdom"] - 0.5) < 0.05 and f["m_mean_gb"] == 2.0
    assert class_of("normal_ddp", None) == "normal"
    assert class_of("swma", "swma_mimicry") == "unobservable"


def test_observed_tokens_and_falsify():
    bundle = {"pattern": {"mean_w": 440, "swing_abs_w": 5, "duty_regularity": 0.1, "dominant_freq_hz": 0.04},
              "unexplainedness": {"evidence": {"declared_family_mismatch": True}}}
    channels = {"u_frac_hi": {"value": 0.95, "normal_q": 0.9}, "m_mean_gb": {"value": 0.3, "normal_q": 0.02},
                "m_range_mb": {"value": 2, "normal_q": 0.1}, "n_procs": {"value": 1, "normal_q": 0.5}}
    tokens = observed_tokens(bundle, channels, {"class_p": {"normal": 0.01}})
    assert {"power_high", "power_flat", "util_saturated", "mem_small", "declared_mismatch", "unlike_known_cases"} <= set(tokens)
    h = Hypothesis("miner", "", ["power_flat", "mem_small", "mem_large"], ["q"])
    falsify(h, tokens)
    assert h.contradictions == ["mem_large"] and h.support == round((2 - 1) / 3, 3)


def test_lexical_gate():
    h = Hypothesis("sponge_examples", "", [], ["energy latency attack neural network"])
    assert lexical_gate(h, Paper("arxiv", "Sponge Examples: Energy-Latency Attacks", "inputs that maximise energy", "u"))
    assert not lexical_gate(h, Paper("arxiv", "Road safety in C-V2X", "vehicular age of information", "u"))

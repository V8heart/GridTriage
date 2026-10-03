import math

import pandas as pd

from pipeline.session_calibration import conformal_threshold, session_stats


def test_conformal_threshold_controls_alarm_rate():
    cal = list(range(1, 20))  # 19 normal sessions
    thr = conformal_threshold(cal, 0.10)
    # (1 + #{cal >= thr}) / 20 <= 0.10  ->  at most one calibration value reaches thr
    assert sum(c >= thr for c in cal) <= 1
    assert conformal_threshold([1, 2], 0.10) == math.inf  # too few sessions: never alarm


def test_session_stats_longest_run():
    part = pd.DataFrame({"start": range(8), "cand0": [0, 1, 1, 0, 1, 1, 1, 0], "p0": [0.5] * 8})
    assert session_stats(part)["R"] == 3


def test_profile_match_prefers_specific_cards():
    from pipeline.kb.profile import match_score

    obs = {"level": "high", "form": "flat", "timescale": "none", "inst": "no", "util": "coupled",
           "declared": "consistent", "multi_gpu": "single", "log": "na"}
    vague = {"declared": "consistent"}
    specific = {"level": "high", "form": "flat", "timescale": "none", "inst": "no", "util": "coupled"}
    assert match_score(obs, specific) > match_score(obs, vague)
    assert match_score(obs, {"form": "square"}) < 0

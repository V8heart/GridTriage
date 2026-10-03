"""Deterministic evidence-bundle -> retrieval query renderer.

The phrases mirror the ``## 관측 시그니처`` sections of the threat cards, so
query and card are written in the same controlled vocabulary (symmetric
retrieval). Cut points were set from the fold_00 *train* segments only (swing separates
SWMA ~180 W from normal ~6 W; duty regularity and periodicity strength do not
separate classes, so they are not used). Evidence is rendered only when its
Stage1 surprisal weight is >= MIN_EVIDENCE_WEIGHT (fires on <= 37% of train
normals); undiscriminating evidence would steer every query the same way.
The declared job name is never rendered (v13 §4.4.3), only whether the power
pattern is consistent with it (declared_family_mismatch is False).
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

PHRASES = {
    "square": "전력 사각파 반복 (power square wave), 규칙적인 high/low 전환 (regular duty cycle)",
    "period": "주기 {p:.1f}초 (period {p:.1f} s)",
    "big_swing": "큰 전력 진폭 swing {w:.0f} W (large power swing)",
    "small_swing": "작은 전력 진폭 (small swing)",
    "flat_high": "평탄한 고부하 (flat high load near TDP)",
    "sustained": "고부하 지속 (sustained high load)",
    "low_power": "저전력 (low power)",
    "inst": "1초 미만 순간 전력 요동 (sub-second instant power fluctuation)",
    "ramp": "급격한 전력 상승·하강 (high power ramp)",
    "decoupled": "GPU 사용률과 전력 불일치 (power-utilization decoupling)",
    "sync": "여러 GPU 동시 동기화 (multi-GPU synchronized power)",
    "log_mismatch": "진행 로그 스텝과 전력 주기 불일치 (log period mismatch)",
    "log_match": "진행 로그 스텝과 전력 주기 일치 (period matches training steps)",
    "declared_mismatch": "신고 작업 유형과 전력 패턴 불일치 (declared job mismatch)",
    "declared_consistent": "신고 작업 유형과 전력 패턴 일치 (consistent with declared job)",
    "changepoint": "로그로 설명되지 않는 변화점 (unexplained changepoint)",
    "peak": "뚜렷한 스펙트럼 첨두 (strong spectral peak)",
}

# Cuts in W / Hz. TDP of the RTX 4090 is 450 W.
SQUARE_SWING_W = 100.0
MIN_EVIDENCE_WEIGHT = 1.0
BIG_SWING_W = 100.0
SMALL_SWING_W = 30.0
FLAT_HIGH_MEAN_W = 300.0
LOW_POWER_W = 80.0
OBSERVABLE_MAX_HZ = 1.0  # ~2 Hz NVML refresh -> Nyquist ~1 Hz


# Benign evidence has surprisal weight 0 by construction but is the main normal cue.
ALWAYS_RENDER = frozenset({"period_match"})
OFFICIAL_CALIBRATION = (Path(__file__).resolve().parents[2] / "dataset" / "real" / "pipeline" / "refit"
                        / "round5b" / "fold_00" / "stage1_v2_calibration.json")


@functools.lru_cache(maxsize=1)
def official_weights() -> dict | None:
    if not OFFICIAL_CALIBRATION.exists():
        return None
    return json.loads(OFFICIAL_CALIBRATION.read_text()).get("evidence_surprisal_weights")


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def render_query(bundle: dict, weights: dict | None | str = "official") -> str:
    """``weights``: Stage1 evidence surprisal weights ("official" = round5b fold_00; None keeps all)."""
    if weights == "official":
        weights = official_weights()
    pattern = bundle.get("pattern") or {}
    evidence = (bundle.get("unexplainedness") or {}).get("evidence") or {}
    freq = _num(pattern.get("dominant_freq_hz"))
    swing = _num(pattern.get("swing_abs_w"))
    mean_w = _num(pattern.get("mean_w"))

    parts: list[str] = []
    square = (
        swing is not None and swing >= SQUARE_SWING_W
        and freq is not None and 0.05 <= freq <= OBSERVABLE_MAX_HZ
    )
    if square:
        parts.append(PHRASES["square"])
        parts.append(PHRASES["period"].format(p=1.0 / freq))
    if swing is not None and swing >= BIG_SWING_W:
        parts.append(PHRASES["big_swing"].format(w=swing))
    elif swing is not None and swing <= SMALL_SWING_W:
        parts.append(PHRASES["small_swing"])
    if mean_w is not None:
        if mean_w >= FLAT_HIGH_MEAN_W and (swing is None or swing <= SMALL_SWING_W):
            parts.append(PHRASES["flat_high"])
        elif mean_w <= LOW_POWER_W:
            parts.append(PHRASES["low_power"])
    for key, phrase in (
        ("sustained_high_load", "sustained"),
        ("inst_power_unsteady", "inst"),
        ("high_ramp", "ramp"),
        ("util_power_decoupled", "decoupled"),
        ("cross_job_sync", "sync"),
        ("log_period_mismatch", "log_mismatch"),
        ("period_match", "log_match"),
        ("declared_family_mismatch", "declared_mismatch"),
        ("unexplained_changepoint", "changepoint"),
        ("strong_peak", "peak"),
    ):
        keep = weights is None or key in ALWAYS_RENDER or key not in weights or weights[key] >= MIN_EVIDENCE_WEIGHT
        if evidence.get(key) and keep:
            parts.append(PHRASES[phrase])
    if evidence.get("declared_family_mismatch") is False:
        parts.append(PHRASES["declared_consistent"])
    if not parts:
        parts.append("뚜렷한 이상 패턴 없음 (no distinctive anomaly pattern)")
    return ". ".join(parts) + "."

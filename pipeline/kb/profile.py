"""Slot-based observation profile (plan_rag_pipeline_v4.md §9.2).

The v4 query only spoke when a known-class cue fired, so unseen classes were
described by one stray sentence. A profile always fills every slot, so any
segment gets a full description in a fixed vocabulary that threat cards share
(their ``profile:`` front-matter). Cut points are physical or defined a priori:

  level      mean power / TDP(450 W): < 0.15 idle-low, < 0.5 mid, else high
  form       swing < 30 W flat; >= 100 W and duty regularity >= 0.6 square;
             >= 30 W otherwise burst; 30-100 W regular small_periodic
  timescale  dominant period vs the ~1 s observation limit and 10 s
  inst / util / declared / multi_gpu / log  straight from Stage1 evidence
"""
from __future__ import annotations

import math

TDP_W = 450.0
SLOTS = ("level", "form", "timescale", "inst", "util", "declared", "multi_gpu", "log")

TEXT = {
    "level": {"low": "저전력 (low power)", "mid": "중간 전력 (mid power)", "high": "고전력 (high power near TDP)"},
    "form": {"flat": "평탄한 전력 (flat power)", "square": "규칙적 사각파 (regular square wave)",
             "burst": "불규칙 버스트 (irregular bursts)", "small_periodic": "작은 주기 변동 (small periodic variation)"},
    "timescale": {"none": "뚜렷한 주기 없음 (no dominant period)", "sub_second": "1초 미만 (sub-second)",
                  "seconds": "수 초 주기 (1-10 s)", "long": "10초 이상 느린 변화 (slower than 10 s)"},
    "inst": {"yes": "순간 전력 요동 있음 (sub-second instant fluctuation)", "no": "순간 전력 안정 (steady instant power)"},
    "util": {"decoupled": "사용률과 전력 불일치 (power-utilization decoupling)",
             "coupled": "사용률과 전력 일치 (power follows utilization)"},
    "declared": {"mismatch": "신고 작업과 불일치 (declared job mismatch)",
                 "consistent": "신고 작업과 일치 (consistent with declared job)"},
    "multi_gpu": {"sync": "여러 GPU 동기화 (multi-GPU synchronized)", "single": "단일 GPU 독립 (single GPU)"},
    "log": {"mismatch": "진행 로그 주기와 불일치 (log period mismatch)", "match": "진행 로그 주기와 일치 (log period match)",
            "na": "진행 로그와 비교 불가 (no comparable log)"},
}
LABEL = {"level": "전력 수준", "form": "변동 형태", "timescale": "시간 척도", "inst": "순간 요동",
         "util": "사용률-전력", "declared": "신고 일치", "multi_gpu": "다중 GPU", "log": "진행 로그"}


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def profile_slots(bundle: dict) -> dict[str, str]:
    pattern = bundle.get("pattern") or {}
    ev = (bundle.get("unexplainedness") or {}).get("evidence") or {}
    mean_w = _num(pattern.get("mean_w")) or 0.0
    swing = _num(pattern.get("swing_abs_w")) or 0.0
    regularity = _num(pattern.get("duty_regularity")) or 0.0
    freq = _num(pattern.get("dominant_freq_hz"))

    frac = mean_w / TDP_W
    level = "low" if frac < 0.15 else "mid" if frac < 0.5 else "high"
    if swing < 30:
        form = "flat"
    elif swing >= 100 and regularity >= 0.6:
        form = "square"
    elif regularity >= 0.6 and swing < 100:
        form = "small_periodic"
    else:
        form = "burst"
    if form == "flat" or freq is None or freq <= 0:
        timescale = "none"
    else:
        period = 1.0 / freq
        timescale = "sub_second" if period < 1.0 else "seconds" if period <= 10.0 else "long"
    if ev.get("log_period_mismatch"):
        log = "mismatch"
    elif ev.get("period_match"):
        log = "match"
    else:
        log = "na"
    return {
        "level": level, "form": form, "timescale": timescale,
        "inst": "yes" if ev.get("inst_power_unsteady") else "no",
        "util": "decoupled" if ev.get("util_power_decoupled") else "coupled",
        "declared": "mismatch" if ev.get("declared_family_mismatch") else "consistent",
        "multi_gpu": "sync" if ev.get("cross_job_sync") else "single",
        "log": log,
    }


def render_profile(slots: dict[str, str]) -> str:
    """Fixed-order sentence used as the retrieval query (and for web research keywords)."""
    return ". ".join(f"{LABEL[s]}: {TEXT[s][slots[s]]}" for s in SLOTS if slots.get(s) in TEXT[s]) + "."


def render_card_profile(profile: dict[str, list[str] | str]) -> str:
    """Card-side text of a profile (allowed values per slot; 'any' = not informative)."""
    parts = []
    for s in SLOTS:
        allowed = profile.get(s, "any")
        if allowed == "any":
            continue
        values = allowed if isinstance(allowed, list) else [allowed]
        parts.append(f"{LABEL[s]}: " + " 또는 ".join(TEXT[s][v] for v in values))
    return ". ".join(parts) + "."


def match_score(slots: dict[str, str], profile: dict[str, list[str] | str], weights: dict[str, float] | None = None) -> float:
    """Specificity-aware match in [-1, 1]: (+w per satisfied slot - w per violated slot) / total slot weight.

    Dividing by *all* slots (not only the card's informative ones) keeps a vague card
    (few constraints, all trivially met) from outranking a specific card whose many
    constraints are all met.
    """
    weights = weights or {}
    total = sum(float(weights.get(s, 1.0)) for s in SLOTS)
    score = 0.0
    for s in SLOTS:
        allowed = profile.get(s, "any")
        if allowed == "any":
            continue
        w = float(weights.get(s, 1.0))
        values = allowed if isinstance(allowed, list) else [allowed]
        score += w if slots.get(s) in values else -w
    return score / total

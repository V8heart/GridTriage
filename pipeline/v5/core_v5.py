"""Operational v5 analysis of one alarmed run (shared by the live engine and offline tools).

  run rows (raw NVML rows of the alarmed span)
    -> 30 s feature windows (raw_features.frame_windows)
    -> case-memory fit + normal-relative channels
    -> v4 query + retrieval (unchanged) -> v5 judge with the open-set gate (a)+(b)

Gate (b) is on by default here: in the v5 evaluation it cut harmful errors 0.23 -> 0.08 and sent
81% of never-seen attacks to the analyst, at the cost of unseen normal workloads becoming 'unknown'
(RESULTS_v5.md §4). 'unknown' is what triggers research and an enrolment candidate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.kb.query import render_query
from pipeline.v5.judge_v5 import CHANNELS, judge_v5
from pipeline.v5.raw_features import frame_windows

MIN_WINDOWS = 1
CORPUS = Path(__file__).resolve().parents[2] / "corpus"
NON_THREAT = {"normal", "unobservable"}


def card_names() -> set[str]:
    return {p.stem for p in CORPUS.glob("*.md")}


def run_windows(rows: pd.DataFrame, hz: float | None = None) -> pd.DataFrame:
    """Feature windows of one GPU's raw rows; hz is inferred from the sampling when not given."""
    if rows.empty:
        return pd.DataFrame()
    rows = rows.sort_values("t_epoch")
    if hz is None:
        step = float(rows["t_epoch"].diff().median() or 0.1)
        hz = 1.0 / max(step, 1e-3)
    for col, default in (("fb_used_mb", float("nan")), ("observed_n_procs", float("nan")),
                         ("power_instant_w", float("nan")), ("util_gpu_pct", float("nan"))):
        if col not in rows:
            rows = rows.assign(**{col: default})
    rows = rows.assign(**{c: pd.to_numeric(rows[c], errors="coerce") for c in
                          ("power_w", "power_instant_w", "util_gpu_pct", "fb_used_mb", "observed_n_procs")})
    feats = frame_windows(rows, hz=round(hz))
    out = pd.DataFrame(feats)
    if not out.empty:
        out["declared_job_family"] = str(rows["declared_job_family"].iloc[0]) if "declared_job_family" in rows else ""
    return out


def analyze_run_v5(bundle: dict, windows: pd.DataFrame, retriever, memory, *, model: str = "gemma3:12b",
                   k: int = 8, mode: str = "hybrid_rerank", host: str | None = None, n_windows: int | None = None,
                   normal_rule: bool = True, enrolled: set[str] | frozenset = frozenset()) -> dict:
    """``memory`` is a CaseMemory; ``enrolled`` the classes with analyst-enrolled cases (CaseStore.enrolled_classes)."""
    query = render_query(bundle)
    hits = retriever.search(query, k=k, mode=mode)
    docs = [h.doc for h in hits]
    fam = (bundle.get("declared_context") or {}).get("job_family")
    if len(windows) >= MIN_WINDOWS:
        case = memory.score(windows, fam)
        channels = memory.normal_quantiles(windows, list(CHANNELS))
    else:  # too little clean data for case features: the judge still runs, the gate cannot fire
        case = {"class_p": {}, "nearest_case_votes": {}, "declared_normal_p": None, "novelty_p": None}
        channels = {}
    judgment = judge_v5(bundle, docs, case, channels, model=model, host=host, n_windows=n_windows,
                        normal_rule=normal_rule, enrolled=enrolled)
    return {
        "query": query,
        "hits": [{"doc_id": h.doc_id, "rank": h.rank, "score": round(h.score, 4), "layer": h.doc["layer"],
                  "title": h.doc["title"], "card": h.doc.get("card"), "text": h.doc["text"]} for h in hits],
        "judgment": judgment.to_dict(),
        "case": case,
        "channels": channels,
    }

"""Shared v4 pipeline steps used by evaluation scripts and the live engine.

  load_stage1      frozen Stage1 model (config + cohort baseline + calibration)
  segment_bundle   candidate windows -> LLM-safe evidence bundle (no raw arrays)
  analyze_segment  bundle -> query -> retrieved KB docs -> LLM final verdict
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.baseline import CohortBaseline
from pipeline.features import features_to_description
from pipeline.kb.query import render_query
from pipeline.run_pipeline import _segment_pattern
from pipeline.stage1_v2 import load_config
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_segments import aggregate_bool_evidence

ROOT = Path(__file__).resolve().parent.parent
REFIT = ROOT / "dataset" / "real" / "pipeline" / "refit"  # local refits (gitignored, needs the raw captures)
MODELS = ROOT / "models"  # frozen official artifacts in git: fitted statistics only, no telemetry
LEGACY_KEYS = ("mean_w", "swing_ratio", "duty_regularity", "n_transitions", "periodicity_strength",
               "high_load_fraction", "ramp_max_w_per_s")
OFFICIAL_STAGE1 = ("round5b", "fold_00")
SESSION_GATE = REFIT / "session_gate.json" if (REFIT / "session_gate.json").exists() else MODELS / "session_gate.json"
FALLBACK_MIN_RUN = 4  # validation-fold choice used before session calibration existed


def session_gate_windows() -> int:
    """Consecutive candidate windows that raise a session alarm (session-calibrated, §9.1)."""
    if SESSION_GATE.exists():
        value = json.loads(SESSION_GATE.read_text()).get("min_run_windows")
        if value:
            return int(value)
    return FALLBACK_MIN_RUN


@dataclass
class Stage1Model:
    config: dict
    baseline: CohortBaseline
    calibration: dict
    name: str


def model_root(model: str, fold: str) -> Path:
    """A local refit when present, else the frozen copy under models/ (fresh clones)."""
    return REFIT if (REFIT / model / fold / "stage1_v2_calibration.json").exists() else MODELS


def load_stage1(model: str = OFFICIAL_STAGE1[0], fold: str = OFFICIAL_STAGE1[1]) -> Stage1Model:
    root = model_root(model, fold)
    cfg = load_config(root / model / "stage1_v2.yaml")
    cal = json.loads((root / model / fold / "stage1_v2_calibration.json").read_text())
    alpha = json.loads((root / model / fold / "stage1_alpha_sweep.json").read_text())["selected_alpha"]["alpha"]
    cfg.update(alpha=alpha, alpha_high_impact=alpha, evidence_thresholds=cal["evidence_thresholds"])
    cfg["evidence_weights"] = {**cfg.get("evidence_weights", {}), **cal.get("evidence_surprisal_weights", {})}
    base = CohortBaseline.load(root / model / fold / "cohort_baseline_v2.json")
    return Stage1Model(cfg, base, cal, f"{model}/{fold}")


def segment_bundle(part: pd.DataFrame, config: dict) -> dict:
    """Aggregate scored windows (needs evidence_bool, u_score, p_value) into one bundle."""
    stage2_cfg = config.get("stage2") or {}
    drop = ("progress_log_missing",) if stage2_cfg.get("drop_progress_log_missing") else ()
    evs = [{k: v for k, v in (ev or {}).items() if isinstance(v, bool)} for ev in part["evidence_bool"]]
    bools, rates = aggregate_bool_evidence(evs, rate_min=float(stage2_cfg.get("evidence_rate_min", 0.6)), drop_keys=drop)
    row = part.iloc[0].to_dict()
    row.update(evidence_bool=bools, pattern=_segment_pattern(part), u_score=float(part["u_score"].median()),
               p_value=float(part["p_value"].median()))
    if "cross_job_sync_index_noidle" in part:
        noidle = pd.to_numeric(part["cross_job_sync_index_noidle"], errors="coerce")
        if noidle.notna().any():
            row["cross_job_sync_index_noidle"] = float(noidle.median())
    legacy = {k: float(pd.to_numeric(part[k], errors="coerce").median()) for k in LEGACY_KEYS if k in part}
    return {
        "bundle": build_evidence_bundle(row, config=config),
        "evidence_rates": rates,
        "legacy_query": features_to_description(legacy) if len(legacy) == len(LEGACY_KEYS) else "",
        "n_windows": int(len(part)),
    }


def analyze_segment(bundle: dict, retriever, *, model: str = "gemma3:12b", k: int = 8,
                    mode: str = "hybrid_rerank", host: str | None = None, gate: bool = False) -> dict:
    from pipeline.stage2_rag_judge import judge

    query = render_query(bundle)
    hits = retriever.search(query, k=k, mode=mode)
    judgment = judge(bundle, [h.doc for h in hits], model=model, host=host, gate=gate)
    return {
        "query": query,
        "hits": [{"doc_id": h.doc_id, "rank": h.rank, "score": round(h.score, 4), "layer": h.doc["layer"],
                  "title": h.doc["title"], "card": h.doc.get("card"), "text": h.doc["text"]} for h in hits],
        "judgment": judgment.to_dict(),
    }

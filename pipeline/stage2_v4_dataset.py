"""Build the Stage2 v4 evaluation segments from real captures.

One segment per target GPU of a session: its ground-truth-labelled windows
(attack windows for attack sessions, every window for normal sessions), plus a
candidate-only segment when Stage1 raised any window. Each segment carries the
LLM-safe evidence bundle (no raw arrays), the legacy description query, and the
human-fixed answer key below. Stage1 is the frozen round5b fold_00 model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.stage1_v2 import build_windows, load_truth_index, score_window
from pipeline.v4_core import REFIT, load_stage1, segment_bundle, session_gate_windows

ROOT = Path(__file__).resolve().parent.parent

# Answer key, fixed before any Stage2 run (plan_rag_pipeline_v4.md §4.1).
# value: (gt_card, accepted cards, observability tier)
NORMAL_CARDS = ("normal", "normal_workloads", "benign_periodic")
ANSWER_KEY = {
    "swma_basic": ("swma", ("swma",), "O"),
    "swma_jitter": ("swma", ("swma",), "O"),
    "swma_shallow": ("swma", ("swma",), "O"),
    "swma_coordinated": ("swma", ("swma", "coordinated_multi_gpu"), "O"),
    "swma_piggyback": ("unobservable", ("unobservable", *NORMAL_CARDS, "swma"), "U"),
    "swma_mimicry": ("unobservable", ("unobservable", *NORMAL_CARDS, "swma"), "U"),
    "ltma_basic": ("ltma", ("ltma",), "O"),
    "crypto_flat": ("cryptojacking", ("cryptojacking",), "O"),
    # misuse captures (bit2watt_impl.misuse_workloads), fixed before they were captured
    "llmjacking_serving": ("llmjacking", ("llmjacking",), "O"),
    "hidden_finetune": ("hidden_ml_training", ("hidden_ml_training",), "O"),
    "sponge_long_gen": ("sponge_attack", ("sponge_attack",), "O"),
    "burst_irregular": ("burst_ramp_attack", ("burst_ramp_attack",), "O"),
}
# Stage2 runs only on >= PERSIST_WINDOWS consecutive Stage1 candidate windows (75 s at a
# 15 s stride). Chosen on validation folds 01/02: normal-session alarms 4%/13% vs 91%/100%
# with any single candidate, visible-attack sessions 78%/100% (plan_rag_pipeline_v4.md §8).
PERSIST_WINDOWS = session_gate_windows()  # now session-calibrated (pipeline/session_calibration.py)


def longest_candidate_run(part):
    """Longest run of consecutive candidate windows (by window start) in one GPU's frame."""
    ordered = part.sort_values("start")
    best, cur = [], []
    for idx, cand in zip(ordered.index, ordered["is_candidate"]):
        cur = cur + [idx] if cand else []
        if len(cur) > len(best):
            best = cur
    return ordered.loc[best]


def answer_for(gt_label: str, gt_variant: str | None) -> tuple[str, tuple[str, ...], str]:
    if str(gt_label).startswith("normal"):
        return "normal", NORMAL_CARDS, "O"
    return ANSWER_KEY[str(gt_variant)]


def build(telemetry: list[Path], model: str = "round5b", fold: str = "fold_00") -> list[dict]:
    s1 = load_stage1(model, fold)
    cfg, base, cal = s1.config, s1.baseline, s1.calibration
    manifest = json.loads((REFIT / model / fold / "split_manifest.json").read_text())
    split = {s: k for k in ("train", "cal", "test") for s in manifest.get(k, [])}
    truth = load_truth_index(ROOT / "dataset/real/index/labels.csv", sessions_root=ROOT / "dataset/real/sessions")

    df = pd.concat([pd.read_csv(p, low_memory=False) for p in telemetry], ignore_index=True)
    windows = build_windows(df, window_s=30, stride_s=15, config=cfg, sessions_root=ROOT / "dataset/real/sessions",
                            truth_index=truth, progress_source="raw")
    scored = [score_window(r.to_dict(), base, cal, cfg) for _, r in windows.iterrows()]
    windows = windows.reset_index(drop=True)
    for key in ("u_score", "p_value", "is_candidate", "evidence_bool"):
        windows[key] = [s[key] for s in scored]

    segments = []
    for (sid, gid), part in windows.groupby(["session_id", "gpu_id"], sort=False):
        info = truth[(str(sid), int(gid))]
        if info.get("gpu_role") != "target":
            continue
        label, variant = str(info["session_gt_label"]), info.get("gt_variant")
        gt_part = part[part["gt_is_attack"].astype(bool)] if not label.startswith("normal") else part
        if gt_part.empty:
            continue
        gt_card, accept, tier = answer_for(label, variant)
        common = {"session_id": str(sid), "gpu_id": int(gid), "gt_label": label, "gt_variant": variant,
                  "gt_card": gt_card, "accept": list(accept), "observability": tier,
                  "split": split.get(str(sid), "holdout"),
                  "declared_job_type": str(part["declared_job_type"].iloc[0])}
        segments.append({"seg_id": f"{sid}:{gid}:all", "scope": "all", **common, **segment_bundle(gt_part, cfg)})
        cand = gt_part[gt_part["is_candidate"]]
        if not cand.empty:
            segments.append({"seg_id": f"{sid}:{gid}:cand", "scope": "candidate", **common, **segment_bundle(cand, cfg)})
        run = longest_candidate_run(gt_part)
        if len(run) >= PERSIST_WINDOWS:
            segments.append({"seg_id": f"{sid}:{gid}:persist", "scope": "persistent", **common, **segment_bundle(run, cfg)})
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, nargs="+", default=[
        ROOT / "dataset/real/pipeline/telemetry_v4_inst.csv",
        ROOT / "dataset/real/pipeline/telemetry_holdout_v2.csv"])
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage2_v4_segments.jsonl")
    parser.add_argument("--model", default="round5b", help="frozen Stage1 refit (round5b = official, round6 = S-C)")
    parser.add_argument("--exclude-train", action="store_true", help="drop sessions in the model's train/cal split")
    args = parser.parse_args()
    segments = build(args.telemetry, model=args.model)
    if args.exclude_train:
        segments = [s for s in segments if s["split"] not in ("train", "cal")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for seg in segments:
            fh.write(json.dumps(seg, ensure_ascii=False, default=str) + "\n")
    frame = pd.DataFrame(segments)
    print(frame.groupby(["scope", "gt_card"]).size().to_string())


if __name__ == "__main__":
    main()

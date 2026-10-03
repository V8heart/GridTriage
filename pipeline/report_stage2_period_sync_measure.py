"""P0-after diagnostic: step-period / period_match / sync pairs. No GPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pipeline.stage1_v2 import build_windows, load_config, score_window
from pipeline.baseline import CohortBaseline
from pipeline.synchronization import session_window_sync_metrics
from pipeline.stage1_v2 import idle_gate_decision


FP_COORD_SESSIONS = {
    "s-03ea7ec9a9bc7f6d",
    "s-2b6a98b47d629286",
    "s-a4751177d4e7c50e",
    "s-a595e33152d4bc59",
    "s-ca28a669ce3c07c7",
    "s-d4dbf93e3f6b141a",
    "s-d74ed92368a694b1",
    "s-df6fda03fb93c615",
}


def _step(evidence) -> float | None:
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("step_period_s")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return float(value)


def _match(evidence) -> object:
    if not isinstance(evidence, dict):
        return None
    return evidence.get("period_match")


def run(args) -> dict:
    config = load_config(args.stage1_config)
    telemetry = pd.read_csv(args.telemetry, low_memory=False)
    labels = pd.read_csv(args.labels, low_memory=False)
    labels["session_id"] = labels["session_id"].astype(str)
    labels["gpu_id"] = labels["gpu_id"].astype(int)

    windows = build_windows(
        telemetry,
        window_s=30.0,
        stride_s=15.0,
        sessions_root=args.sessions_root,
        config=config,
        progress_source=args.progress_source,
    )
    baseline = CohortBaseline.load(args.baseline)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    scores = [score_window(row.to_dict(), baseline, calibration, config) for _, row in windows.iterrows()]
    windows["is_candidate"] = [item["is_candidate"] for item in scores]
    windows["idle_status"] = [item.get("idle_status") for item in scores]
    windows["u_score"] = [item.get("u_score") for item in scores]

    work = windows.merge(
        labels[["session_id", "gpu_id", "gt_label", "gt_variant", "gpu_role", "declared_job_type"]],
        on=["session_id", "gpu_id"],
        how="left",
    )
    work["session_id"] = work["session_id"].astype(str)
    work["step_period_s"] = work["evidence"].map(_step)
    work["period_match_raw"] = work["evidence"].map(_match)
    work["freq"] = pd.to_numeric(work.get("dominant_freq_hz"), errors="coerce")

    target = work[work["gpu_role"].astype(str).eq("target")].copy()
    rows = []
    for variant, group in target.groupby(target["gt_variant"].astype(str), dropna=False):
        steps = group["step_period_s"].dropna()
        matches = group["period_match_raw"]
        rows.append(
            {
                "variant": str(variant),
                "n": int(len(group)),
                "step_notnull": int(steps.notna().sum()) if len(group) else 0,
                "step_frac": float(steps.notna().mean()) if len(group) else 0.0,
                "step_p10": float(steps.quantile(0.10)) if len(steps) else None,
                "step_p50": float(steps.median()) if len(steps) else None,
                "step_p90": float(steps.quantile(0.90)) if len(steps) else None,
                "match_true": int((matches == True).sum()),  # noqa: E712
                "match_false": int((matches == False).sum()),  # noqa: E712
                "match_none": int(matches.isna().sum() + (matches.map(lambda x: x is None)).sum()),
                "freq_p50": float(group["freq"].median()) if group["freq"].notna().any() else None,
                "cand": float(group["is_candidate"].mean()) if len(group) else 0.0,
            }
        )

    pairs = []
    for sid in sorted(FP_COORD_SESSIONS):
        sub = work[work["session_id"].eq(sid)]
        if sub.empty:
            pairs.append({"session_id": sid, "note": "no windows"})
            continue
        t0 = float(sub["window_start_s"].iloc[0])
        t1 = float(sub["window_end_s"].iloc[0])
        session_df = telemetry[telemetry["session_id"].astype(str).eq(sid)]
        metrics = session_window_sync_metrics(session_df, t0, t1)
        gpu_idle = {}
        for gpu, g in session_df.groupby("gpu_id"):
            win = g[(g["timestamp"] >= t0) & (g["timestamp"] <= t1)] if "timestamp" in g.columns else g
            row = {
                "mean_w": float(pd.to_numeric(win.get("power_w"), errors="coerce").mean()) if len(win) else None,
                "observed_n_procs": float(pd.to_numeric(win.get("observed_n_procs"), errors="coerce").median())
                if "observed_n_procs" in win.columns and len(win)
                else None,
            }
            idle = idle_gate_decision(row, config) or {}
            declared = None
            if "declared_job_type" in g.columns:
                declared = str(g["declared_job_type"].iloc[0])
            gpu_idle[int(gpu)] = {
                "idle_status": idle.get("idle_status"),
                "declared": declared,
                "mean_w": row["mean_w"],
                "n_procs": row["observed_n_procs"],
            }
        pair = metrics.get("cross_job_sync_pair")
        pairs.append(
            {
                "session_id": sid,
                "cross_job_sync_index": metrics.get("cross_job_sync_index"),
                "cross_job_sync_pair": pair,
                "gpus": gpu_idle,
                "idle_pair": bool(pair)
                and any((gpu_idle.get(int(g)) or {}).get("idle_status") == "idle" for g in pair),
            }
        )

    report = {
        "n_windows": int(len(windows)),
        "n_candidates": int(windows["is_candidate"].sum()),
        "by_variant": rows,
        "fp_sync_pairs": pairs,
        "progress_source": args.progress_source,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Stage2 period / sync measure (after P0)", "", f"source=`{args.progress_source}` windows={report['n_windows']} cands={report['n_candidates']}", ""]
    md.append("| variant | n | step% | step p10/p50/p90 | match T/F | freq p50 | cand |")
    md.append("|---|---:|---:|---|---|---:|---:|")
    for row in rows:
        step = (
            f"{row['step_p10']:.3f}/{row['step_p50']:.3f}/{row['step_p90']:.3f}"
            if row["step_p50"] is not None
            else "—"
        )
        freq = "" if row["freq_p50"] is None else f"{row['freq_p50']:.3f}"
        md.append(
            f"| {row['variant']} | {row['n']} | {row['step_frac']:.2f} | {step} | "
            f"{row['match_true']}/{row['match_false']} | {freq} | {row['cand']:.3f} |"
        )
    md.extend(["", "## FP coordinated sync pairs", ""])
    for item in pairs:
        md.append(f"- `{item['session_id']}` pair={item.get('cross_job_sync_pair')} "
                  f"idx={item.get('cross_job_sync_index')} idle_pair={item.get('idle_pair')} "
                  f"gpus={item.get('gpus')}")
    args.md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", default="dataset/real/pipeline/telemetry_with_procs.csv")
    parser.add_argument("--labels", default="dataset/real/index/labels.csv")
    parser.add_argument("--sessions-root", default="dataset/real/sessions")
    parser.add_argument("--stage1-config", default="config/stage1_v2_real.yaml")
    parser.add_argument("--baseline", default="dataset/real/pipeline/refit/round3/fold_00/cohort_baseline_v2.json")
    parser.add_argument("--calibration", default="dataset/real/pipeline/refit/round3/fold_00/stage1_v2_calibration.json")
    parser.add_argument("--progress-source", default="raw")
    parser.add_argument("--out", type=Path, default=Path("dataset/real/pipeline/refit/round3/stage2_period_sync_measure.json"))
    parser.add_argument("--md", type=Path, default=Path("dataset/real/pipeline/refit/round3/stage2_period_sync_measure.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

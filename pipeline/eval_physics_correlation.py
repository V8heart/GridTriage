"""Plots and cautious PTPS calibration proposal for joined Tier B results."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gridtriage-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.physics_correlation import correlation_report


def build_ptps_proposal(frame: pd.DataFrame, *, minimum_rows: int = 8) -> dict:
    usable = frame[
        frame["converged"].fillna(False)
        & frame["osc_std"].notna()
        & frame["cyber_anomaly_score_max"].notna()
    ]
    feature_columns = [
        column
        for column in usable.columns
        if column.startswith("cyber_")
        and column not in {"cyber_candidate_windows", "cyber_anomaly_score_max", "cyber_anomaly_score_median"}
        and pd.api.types.is_numeric_dtype(usable[column])
    ]
    associations: dict[str, dict] = {}
    for column in feature_columns:
        subset = usable[[column, "osc_std"]].dropna()
        if len(subset) < 3 or subset[column].nunique() < 2:
            continue
        rho, pvalue = spearmanr(subset[column], subset["osc_std"])
        if np.isfinite(rho):
            associations[column] = {
                "n": int(len(subset)),
                "rho": float(rho),
                "pvalue": float(pvalue),
            }

    proposal = {
        "scope": "calibration proposal from public dynamic test systems only",
        "usable_rows": int(len(usable)),
        "minimum_rows_for_numeric_weights": minimum_rows,
        "status": "directional_only",
        "feature_associations": associations,
        "proposed_weights": None,
        "warning": "Do not interpret these values as impact estimates for a real regional grid.",
    }
    if len(usable) >= minimum_rows and associations:
        magnitudes = {
            key: abs(value["rho"])
            for key, value in associations.items()
            if value["n"] >= minimum_rows
        }
        total = sum(magnitudes.values())
        if total > 0:
            proposal["status"] = "candidate_weights_require_external_validation"
            proposal["proposed_weights"] = {
                key: value / total for key, value in magnitudes.items()
            }
    return proposal


def evaluate(
    joined_path: Path,
    *,
    report_path: Path,
    proposal_path: Path,
    boxplot_path: Path,
) -> tuple[dict, dict]:
    frame = pd.read_csv(joined_path)
    report = correlation_report(frame)
    proposal = build_ptps_proposal(frame)

    for path in (report_path, proposal_path, boxplot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    usable = frame[
        frame["converged"].fillna(False)
        & frame["osc_std"].notna()
        & frame["threat_id"].notna()
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = [
        (str(name), group["osc_std"].dropna().to_numpy())
        for name, group in usable.groupby("threat_id", dropna=False)
        if len(group["osc_std"].dropna())
    ]
    if grouped:
        ax.boxplot([values for _, values in grouped], tick_labels=[name for name, _ in grouped])
    else:
        ax.text(0.5, 0.5, "No converged matched rows", ha="center", va="center")
        ax.set_xticks([])
    ax.set_ylabel("Max generator omega std (p.u.)")
    ax.set_title("Physics response by cyber threat match (public test systems)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=160)
    plt.close(fig)
    return report, proposal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined",
        type=Path,
        default=Path("dataset/eval/cyber_physics_joined.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/eval/cyber_physics_correlation.json"),
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path("dataset/eval/ptps_calibration_proposal.json"),
    )
    parser.add_argument(
        "--boxplot",
        type=Path,
        default=Path("dataset/eval/physics_response_boxplot.png"),
    )
    args = parser.parse_args()
    report, proposal = evaluate(
        args.joined,
        report_path=args.report,
        proposal_path=args.proposal,
        boxplot_path=args.boxplot,
    )
    print(json.dumps({"correlation": report, "ptps": proposal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Phase 2: periodic-load frequency sweep on the public Kundur system."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gridtriage-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bit2watt_impl.physics.simulation import (
    build_kundur_system,
    ensure_parent,
    simulate_profile,
)
from dataset.attack_profiles import sweep_profile


def run_sweep(
    frequencies: np.ndarray,
    *,
    output: Path,
    plot: Path,
    summary: Path,
    amplitude_frac: float = 0.10,
    duration_s: float = 14.0,
    pq_idx: str | None = None,
    max_chunk_s: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for frequency in frequencies:
        profile = sweep_profile(
            float(frequency),
            amplitude_frac=amplitude_frac,
            duration_s=duration_s,
        )
        result = simulate_profile(
            build_kundur_system,
            profile,
            test_system="kundur_ieeest",
            model_name="GENROU",
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s,
        )
        rows.append(result.row)
        ensure_parent(output)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"[Kundur sweep] {frequency:.2f} Hz: "
            f"{'ok' if result.row['converged'] else result.row['failure_reason']}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    successful = frame[frame["converged"].fillna(False)].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not successful.empty:
        ax.plot(successful["frequency_hz"], successful["osc_std"], marker="o", ms=3)
    failed = frame[~frame["converged"].fillna(False)]
    if not failed.empty:
        ax.scatter(failed["frequency_hz"], np.zeros(len(failed)), marker="x", label="failed")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=1, label="~0.6 Hz reference")
    ax.set(xlabel="Attack frequency (Hz)", ylabel="Max generator omega std (p.u.)")
    ax.set_title("Kundur public test system: periodic PQ-load response")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent(plot)
    fig.savefig(plot, dpi=160)
    plt.close(fig)

    report: dict[str, object] = {
        "test_system": "kundur_ieeest",
        "scope": "public dynamic test system; not a real regional grid",
        "runs": int(len(frame)),
        "converged": int(len(successful)),
        "convergence_rate": float(len(successful) / len(frame)) if len(frame) else 0.0,
        "peak_frequency_hz": None,
        "peak_osc_std": None,
        "near_0_6_peak_frequency_hz": None,
        "global_peak_near_0_6": None,
        "interpretation": "insufficient converged runs",
    }
    if not successful.empty:
        peak = successful.loc[successful["osc_std"].idxmax()]
        report["peak_frequency_hz"] = float(peak["frequency_hz"])
        report["peak_osc_std"] = float(peak["osc_std"])
        report["global_peak_near_0_6"] = bool(
            abs(float(peak["frequency_hz"]) - 0.6) <= 0.1
        )
        report["interpretation"] = (
            "global osc_std peak is near the ~0.6 Hz reference"
            if report["global_peak_near_0_6"]
            else "global osc_std peak near ~0.6 Hz was not observed"
        )
        near = successful[
            (successful["frequency_hz"] >= 0.5)
            & (successful["frequency_hz"] <= 0.7)
        ]
        if not near.empty:
            report["near_0_6_peak_frequency_hz"] = float(
                near.loc[near["osc_std"].idxmax(), "frequency_hz"]
            )
    ensure_parent(summary)
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_results.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_kundur.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_kundur_summary.json"),
    )
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--stop", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--amplitude-frac", type=float, default=0.10)
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument("--pq-idx", default=None)
    parser.add_argument("--max-chunk-s", type=float, default=0.5)
    args = parser.parse_args()
    run_sweep(
        np.arange(args.start, args.stop, args.step),
        output=args.output,
        plot=args.plot,
        summary=args.summary,
        amplitude_frac=args.amplitude_frac,
        duration_s=args.duration_s,
        pq_idx=args.pq_idx,
        max_chunk_s=args.max_chunk_s,
    )


if __name__ == "__main__":
    main()

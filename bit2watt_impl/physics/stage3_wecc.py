"""Phase 3: PQ-load injection on the public ANDES WECC 179-bus case."""
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
    build_wecc_system,
    ensure_parent,
    simulate_profile,
)
from dataset.attack_profiles import PHASE1_PROFILES, sweep_profile


def run_wecc(
    *,
    output: Path,
    smoke_output: Path,
    comparison_plot: Path,
    comparison_summary: Path,
    kundur_csv: Path,
    frequencies: np.ndarray,
    amplitude_frac: float,
    duration_s: float,
    pq_idx: str | None,
    max_chunk_s: float,
    smoke_only: bool = False,
) -> pd.DataFrame:
    smoke_rows: list[dict] = []
    for name in ("normal", "periodic"):
        result = simulate_profile(
            build_wecc_system,
            PHASE1_PROFILES[name],
            test_system="wecc_179_gencls",
            model_name="GENCLS",
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s,
            pq_selection="largest_p0",
        )
        smoke_rows.append(result.row)
        print(f"[WECC smoke] {name}: {'ok' if result.row['converged'] else 'failed'}")
    smoke = pd.DataFrame(smoke_rows)
    ensure_parent(smoke_output)
    smoke.to_csv(smoke_output, index=False)

    if smoke_only or not smoke["converged"].fillna(False).all():
        if not smoke["converged"].fillna(False).all():
            print("WECC smoke failed; full sweep skipped. Failure is preserved in the CSV.")
        return pd.DataFrame()

    rows: list[dict] = []
    for frequency in frequencies:
        result = simulate_profile(
            build_wecc_system,
            sweep_profile(
                float(frequency),
                amplitude_frac=amplitude_frac,
                duration_s=duration_s,
            ),
            test_system="wecc_179_gencls",
            model_name="GENCLS",
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s,
            pq_selection="largest_p0",
        )
        rows.append(result.row)
        ensure_parent(output)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"[WECC sweep] {frequency:.2f} Hz: {'ok' if result.row['converged'] else 'failed'}")
    frame = pd.DataFrame(rows)
    ensure_parent(output)
    frame.to_csv(output, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    datasets: list[tuple[str, pd.DataFrame]] = [("WECC 179", frame)]
    if kundur_csv.exists():
        datasets.insert(0, ("Kundur", pd.read_csv(kundur_csv)))
    report: dict[str, object] = {
        "scope": "comparison of public dynamic test systems; not a real regional grid",
        "systems": {},
    }
    for label, data in datasets:
        good = data[data["converged"].fillna(False)]
        if good.empty:
            report["systems"][label] = {"runs": int(len(data)), "converged": 0, "peak_hz": None}
            continue
        ax.plot(good["frequency_hz"], good["osc_std"], marker="o", ms=3, label=label)
        peak = good.loc[good["osc_std"].idxmax()]
        report["systems"][label] = {
            "runs": int(len(data)),
            "converged": int(len(good)),
            "peak_hz": float(peak["frequency_hz"]),
            "peak_osc_std": float(peak["osc_std"]),
        }
    ax.axvline(0.6, color="gray", linestyle=":", linewidth=1, label="Kundur ~0.6 Hz ref.")
    ax.axvline(0.37, color="gray", linestyle="--", linewidth=1, label="WECC gallery 0.37 Hz ref.")
    wecc_peak = report["systems"].get("WECC 179", {}).get("peak_hz")
    report["wecc_reference_hz"] = 0.37
    report["wecc_global_peak_near_reference"] = (
        bool(abs(float(wecc_peak) - 0.37) <= 0.1) if wecc_peak is not None else None
    )
    report["interpretation"] = (
        "bundled WECC GENCLS global peak is near the 0.37 Hz gallery reference"
        if report["wecc_global_peak_near_reference"]
        else "bundled WECC GENCLS/PQ experiment did not reproduce a global peak near 0.37 Hz"
    )
    ax.set(xlabel="Attack frequency (Hz)", ylabel="Max generator omega std (p.u.)")
    ax.set_title("Public test systems: periodic PQ-load response")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent(comparison_plot)
    fig.savefig(comparison_plot, dpi=160)
    plt.close(fig)
    ensure_parent(comparison_summary)
    comparison_summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/wecc_frequency_sweep_results.csv"),
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path("dataset/eval/wecc_smoke_results.csv"),
    )
    parser.add_argument(
        "--kundur-csv",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_results.csv"),
    )
    parser.add_argument(
        "--comparison-plot",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_comparison.png"),
    )
    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=Path("dataset/eval/frequency_sweep_comparison.json"),
    )
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--stop", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--amplitude-frac", type=float, default=0.10)
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument("--pq-idx", default=None)
    parser.add_argument("--max-chunk-s", type=float, default=0.5)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run_wecc(
        output=args.output,
        smoke_output=args.smoke_output,
        comparison_plot=args.comparison_plot,
        comparison_summary=args.comparison_summary,
        kundur_csv=args.kundur_csv,
        frequencies=np.arange(args.start, args.stop, args.step),
        amplitude_frac=args.amplitude_frac,
        duration_s=args.duration_s,
        pq_idx=args.pq_idx,
        max_chunk_s=args.max_chunk_s,
        smoke_only=args.smoke_only,
    )


if __name__ == "__main__":
    main()

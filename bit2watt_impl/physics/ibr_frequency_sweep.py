"""Phase 3: attack-frequency × IBR-penetration sweep on public WECC 179.

Uses only penetrations that pass Phase-2 PFlow+TDS baselines. Metrics reuse
GENCLS omega features from remaining synchronous machines after SG→REGCA1
replacement. Results are reported for the public test system only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gridtriage-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bit2watt_impl.physics.simulation import (
    build_wecc_system_at_penetration,
    ensure_parent,
    simulate_profile,
)
from bit2watt_impl.physics.waveforms import build_event_schedule
from dataset.attack_profiles import sweep_profile


def _builder(penetration: float, profile=None) -> Callable[[], Any]:
    def _build():
        ppf_events = None
        if profile is not None and profile.kind == "sine":
            ppf_events = [
                (event.time_s, event.relative_delta)
                for event in build_event_schedule(profile)
            ]
        return build_wecc_system_at_penetration(
            penetration,
            require_pflow=True,
            ppf_events=ppf_events,
        )

    return _build


def load_viable_penetrations(
    baseline_csv: Path,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Return (valid pens, excluded records with reasons)."""
    frame = pd.read_csv(baseline_csv)
    excluded: list[dict[str, Any]] = []
    if "valid_operating_point" in frame.columns:
        valid_mask = frame["valid_operating_point"].fillna(False).astype(bool)
    else:
        # Backward compatible fallback for older baseline CSVs.
        valid_mask = frame["pflow_ok"].astype(bool) & frame["tds_ok"].astype(bool)
    for _, row in frame.iterrows():
        if bool(valid_mask.loc[row.name]):
            continue
        excluded.append(
            {
                "target_penetration": float(row["target_penetration"]),
                "reason": str(
                    row.get("excluded_reason")
                    or row.get("failure_reason")
                    or "not_valid_operating_point"
                ),
                "f_hz": (None if pd.isna(row.get("f_hz")) else float(row.get("f_hz"))),
                "pflow_ok": bool(row.get("pflow_ok")),
                "tds_ok": bool(row.get("tds_ok")),
            }
        )
    viable = frame.loc[valid_mask]
    if viable.empty:
        raise RuntimeError(f"no valid operating-point penetrations in {baseline_csv}")
    pens = [float(x) for x in viable["target_penetration"].tolist()]
    return pens, excluded


def run_2d_sweep(
    *,
    penetrations: list[float],
    frequencies: np.ndarray,
    output: Path,
    heatmap: Path,
    overlay: Path,
    summary: Path,
    amplitude_frac: float = 0.012,
    duration_s: float = 14.0,
    max_chunk_s: float = 0.5,
    criteria: int = 0,
    excluded_penetrations: list[dict[str, Any]] | None = None,
    waveform_kind: str = "sine",
) -> pd.DataFrame:
    rows: list[dict] = []
    # Fresh runs only — schema/metrics changed (rocof_rms etc.).
    done: set[tuple[float, float]] = set()

    for pen in penetrations:
        probe = build_wecc_system_at_penetration(pen, require_pflow=True)
        meta = dict(getattr(probe, "_gridtriage_ibr_meta", {}) or {})
        del probe

        for frequency in frequencies:
            key = (round(float(pen), 6), round(float(frequency), 6))
            if key in done:
                continue
            profile = sweep_profile(
                float(frequency),
                amplitude_frac=amplitude_frac,
                duration_s=duration_s,
                kind=waveform_kind,
            )
            result = simulate_profile(
                _builder(pen, profile),
                profile,
                test_system="wecc_179_gencls_ibr",
                model_name="GENCLS",
                max_chunk_s=max_chunk_s,
                pq_selection="largest_p0",
                criteria=criteria,
                max_rebuilds=0 if waveform_kind == "sine" else 2,
            )
            row = dict(result.row)
            row["target_penetration"] = float(pen)
            row["achieved_penetration"] = meta.get("achieved_penetration")
            row["replaced_count"] = meta.get("replaced_count")
            row["inertia_before"] = meta.get("inertia_before")
            row["inertia_after"] = meta.get("inertia_after")
            row["inertia_reduction_frac"] = meta.get("inertia_reduction_frac")
            rows.append(row)
            ensure_parent(output)
            pd.DataFrame(rows).to_csv(output, index=False)
            status = "ok" if row.get("converged") else row.get("failure_reason")
            print(
                f"[IBR 2D] {pen:.0%} @ {frequency:.2f} Hz: {status} "
                f"osc_std={row.get('osc_std')} "
                f"rocof_rms={row.get('rocof_rms_hz_s')} "
                f"rocof_max={row.get('rocof_max_hz_s')}"
            )

    frame = pd.DataFrame(rows)
    ensure_parent(output)
    frame.to_csv(output, index=False)
    _write_plots(frame, heatmap=heatmap, overlay=overlay)
    report = _summary_report(
        frame,
        penetrations=penetrations,
        amplitude_frac=amplitude_frac,
        criteria=criteria,
        excluded_penetrations=excluded_penetrations or [],
        waveform_kind=waveform_kind,
    )
    ensure_parent(summary)
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return frame


def _write_plots(frame: pd.DataFrame, *, heatmap: Path, overlay: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["converged"] = work["converged"].fillna(False).astype(bool)

    # Heatmap of osc_std
    pivot = work.pivot_table(
        index="target_penetration",
        columns="frequency_hz",
        values="osc_std",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not pivot.empty:
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{float(v):.0%}" for v in pivot.index])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{float(c):.2f}" for c in pivot.columns], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, label="osc_std (omega p.u.)")
    ax.set_xlabel("Attack frequency (Hz)")
    ax.set_ylabel("Target IBR penetration")
    ax.set_title("Public WECC 179: osc_std vs frequency × IBR penetration")
    fig.tight_layout()
    ensure_parent(heatmap)
    fig.savefig(heatmap, dpi=160)
    plt.close(fig)

    # Overlay curves
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for pen, group in work.groupby("target_penetration"):
        ok = group[group["converged"]].sort_values("frequency_hz")
        if ok.empty:
            continue
        ax.plot(
            ok["frequency_hz"],
            ok["osc_std"],
            marker="o",
            ms=3,
            label=f"{float(pen):.0%} IBR",
        )
        bad = group[~group["converged"]]
        if not bad.empty:
            ax.scatter(
                bad["frequency_hz"],
                np.zeros(len(bad)),
                marker="x",
                color="gray",
                s=20,
            )
    ax.set_xlabel("Attack frequency (Hz)")
    ax.set_ylabel("Max remaining-GENCLS omega std (p.u.)")
    ax.set_title("Public WECC 179: response vs attack frequency by IBR penetration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent(overlay)
    fig.savefig(overlay, dpi=160)
    plt.close(fig)


def write_waveform_comparison(
    square_csv: Path,
    sine_csv: Path,
    *,
    plot: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Place square and sine osc_std curves side by side and summarize 50%."""
    square = pd.read_csv(square_csv)
    sine = pd.read_csv(sine_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    report: dict[str, Any] = {
        "scope": "public dynamic test system; not a real regional grid",
        "square_csv": str(square_csv),
        "sine_csv": str(sine_csv),
        "by_waveform": {},
    }
    for ax, (kind, frame) in zip(axes, (("square", square), ("sine", sine))):
        frame = frame.copy()
        frame["converged"] = frame["converged"].fillna(False).astype(bool)
        stats: dict[str, Any] = {}
        for pen, group in frame.groupby("target_penetration"):
            ok = group[group["converged"]].sort_values("frequency_hz")
            stats[f"{float(pen):.2f}"] = {
                "runs": int(len(group)),
                "converged": int(len(ok)),
                "osc_std_min": float(ok["osc_std"].min()) if len(ok) else None,
                "osc_std_max": float(ok["osc_std"].max()) if len(ok) else None,
                "peak_frequency_hz": (
                    float(ok.loc[ok["osc_std"].idxmax(), "frequency_hz"])
                    if len(ok)
                    else None
                ),
            }
            if len(ok):
                ax.plot(
                    ok["frequency_hz"],
                    ok["osc_std"],
                    marker="o",
                    ms=3,
                    label=f"{float(pen):.0%} IBR",
                )
        report["by_waveform"][kind] = stats
        ax.set_title(f"{kind}: converged osc_std")
        ax.set_xlabel("Attack frequency (Hz)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Max remaining-GENCLS omega std (p.u.)")
    fig.suptitle("Public WECC 179: square vs stepped-sine frequency response")
    fig.tight_layout()
    ensure_parent(plot)
    fig.savefig(plot, dpi=160)
    plt.close(fig)

    sine_50 = report["by_waveform"]["sine"].get("0.50", {})
    if sine_50.get("converged", 0) == 0:
        conclusion = (
            "The 50% stepped-sine sweep produced no converged runs, so "
            "frequency separation or resonance cannot be assessed."
        )
    else:
        lo = sine_50["osc_std_min"]
        hi = sine_50["osc_std_max"]
        ratio = (hi / lo) if lo and lo > 0 else None
        conclusion = (
            "The 50% stepped-sine response is frequency-dependent."
            if ratio is not None and ratio >= 1.2
            else "The 50% stepped-sine response remains comparatively flat."
        )
    report["fifty_percent_conclusion"] = conclusion
    ensure_parent(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _summary_report(
    frame: pd.DataFrame,
    *,
    penetrations: list[float],
    amplitude_frac: float,
    criteria: int,
    excluded_penetrations: list[dict[str, Any]],
    waveform_kind: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "test_system": "wecc_179_gencls",
        "scope": "public dynamic test system; not a real regional grid",
        "method": (
            "capacity-ranked GENCLS→REGCA1+REECA1 replacement; "
            f"PQ.Ppf {waveform_kind} injection"
        ),
        "waveform_kind": waveform_kind,
        "amplitude_frac": float(amplitude_frac),
        "tds_criteria": int(criteria),
        "amplitude_frac_note": (
            "Identical amplitude_frac is used across penetrations so relative "
            "comparisons remain valid even when absolute responses shrink. "
            "Primary RoCoF evidence uses rocof_rms_hz_s (transition-masked); "
            "rocof_max_hz_s / legacy rocof_hz_s is diagnostic only for waveform edges."
        ),
        "penetrations_requested": penetrations,
        "excluded_penetrations": excluded_penetrations,
        "runs": int(len(frame)),
        "converged": int(frame["converged"].fillna(False).sum()) if len(frame) else 0,
        "by_penetration": {},
        "penetration_vs_mean_osc_std": {},
        "penetration_vs_mean_rocof_rms_hz_s": {},
        "spearman_penetration_osc_std": None,
        "spearman_penetration_rocof_rms": None,
        "response_increases_with_penetration": None,
        "rocof_rms_increases_with_penetration": None,
        "interpretation": "insufficient data",
    }
    if frame.empty:
        return report

    ok = frame[frame["converged"].fillna(False)].copy()
    report["convergence_rate"] = float(len(ok) / len(frame))

    for pen, group in frame.groupby("target_penetration"):
        g_ok = group[group["converged"].fillna(False)]
        mean_rms = None
        mean_max = None
        if len(g_ok):
            if "rocof_rms_hz_s" in g_ok and g_ok["rocof_rms_hz_s"].notna().any():
                mean_rms = float(g_ok["rocof_rms_hz_s"].mean(skipna=True))
            if "rocof_max_hz_s" in g_ok and g_ok["rocof_max_hz_s"].notna().any():
                mean_max = float(g_ok["rocof_max_hz_s"].mean(skipna=True))
            elif "rocof_hz_s" in g_ok and g_ok["rocof_hz_s"].notna().any():
                mean_max = float(g_ok["rocof_hz_s"].mean(skipna=True))
        report["by_penetration"][f"{float(pen):.2f}"] = {
            "runs": int(len(group)),
            "converged": int(len(g_ok)),
            "mean_osc_std": float(g_ok["osc_std"].mean()) if len(g_ok) else None,
            "mean_rocof_rms_hz_s": mean_rms,
            "mean_rocof_max_hz_s": mean_max,
            # Legacy key retained but documented as max/diagnostic.
            "mean_rocof_hz_s": mean_max,
            "inertia_after": (
                float(g_ok["inertia_after"].iloc[0])
                if len(g_ok) and "inertia_after" in g_ok
                else None
            ),
        }
        if len(g_ok):
            report["penetration_vs_mean_osc_std"][f"{float(pen):.2f}"] = float(
                g_ok["osc_std"].mean()
            )
            if mean_rms is not None:
                report["penetration_vs_mean_rocof_rms_hz_s"][f"{float(pen):.2f}"] = mean_rms

    if not ok.empty and ok["target_penetration"].nunique() >= 2:
        def _slopes(metric: str) -> list[float]:
            out: list[float] = []
            if metric not in ok.columns:
                return out
            for _, grp in ok.groupby("frequency_hz"):
                sub = grp.dropna(subset=[metric])
                if sub["target_penetration"].nunique() < 2:
                    continue
                x = sub["target_penetration"].to_numpy(dtype=float)
                y = sub[metric].to_numpy(dtype=float)
                if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and len(x) >= 2:
                    out.append(float(np.polyfit(x, y, 1)[0]))
            return out

        osc_slopes = _slopes("osc_std")
        rms_slopes = _slopes("rocof_rms_hz_s")
        n_levels = int(ok["target_penetration"].nunique())
        report["comparable_penetration_levels"] = n_levels
        failed_pens = sorted(
            {
                float(p)
                for p, g in frame.groupby("target_penetration")
                if not g["converged"].fillna(False).any()
            }
        )
        report["attack_tds_failed_penetrations"] = failed_pens

        try:
            from scipy.stats import spearmanr

            agg = ok.groupby("target_penetration")["osc_std"].mean().reset_index()
            if len(agg) >= 3:
                rho, pvalue = spearmanr(agg["target_penetration"], agg["osc_std"])
                report["spearman_penetration_osc_std"] = {
                    "n": int(len(agg)),
                    "rho": float(rho) if np.isfinite(rho) else None,
                    "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                }
            if "rocof_rms_hz_s" in ok.columns:
                agg_r = (
                    ok.dropna(subset=["rocof_rms_hz_s"])
                    .groupby("target_penetration")["rocof_rms_hz_s"]
                    .mean()
                    .reset_index()
                )
                if len(agg_r) >= 3:
                    rho, pvalue = spearmanr(
                        agg_r["target_penetration"], agg_r["rocof_rms_hz_s"]
                    )
                    report["spearman_penetration_rocof_rms"] = {
                        "n": int(len(agg_r)),
                        "rho": float(rho) if np.isfinite(rho) else None,
                        "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                    }
        except Exception:
            pass

        notes = []
        if excluded_penetrations:
            notes.append(
                f"Excluded before sweep (invalid operating point): {excluded_penetrations}."
            )
        if failed_pens:
            notes.append(
                f"Attack-TDS failed penetrations {failed_pens} "
                "(distinct from operating-point exclusion)."
            )
        note = (" " + " ".join(notes)) if notes else ""

        if osc_slopes:
            mean_slope = float(np.mean(osc_slopes))
            report["mean_osc_std_vs_penetration_slope"] = mean_slope
            report["response_increases_with_penetration"] = bool(mean_slope > 0)
        if rms_slopes:
            mean_rms_slope = float(np.mean(rms_slopes))
            report["mean_rocof_rms_vs_penetration_slope"] = mean_rms_slope
            report["rocof_rms_increases_with_penetration"] = bool(mean_rms_slope > 0)

        if n_levels < 3:
            report["interpretation"] = (
                f"Only {n_levels} IBR levels produced converged attack runs on this "
                "public WECC case; trend is descriptive_only."
                + note
            )
            report["inference_status"] = "descriptive_only"
        else:
            report["inference_status"] = "candidate_for_validation"
            osc_bit = (
                "osc_std increased with penetration"
                if report.get("response_increases_with_penetration")
                else "osc_std did not increase with penetration"
            )
            rms_bit = (
                "rocof_rms increased with penetration"
                if report.get("rocof_rms_increases_with_penetration")
                else "rocof_rms did not increase with penetration"
            )
            report["interpretation"] = (
                f"On this public WECC case with identical amplitude_frac={amplitude_frac}, "
                f"{osc_bit}; {rms_bit}. Do not use rocof_max/legacy rocof_hz_s as "
                f"frequency-resolved evidence."
                + note
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WECC IBR penetration × attack-frequency 2D sweep (public test system)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("dataset/eval/ibr_penetration_baseline.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_sweep.csv"),
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_heatmap.png"),
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_overlay.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_summary.json"),
    )
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--stop", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--amplitude-frac",
        type=float,
        default=0.012,
        help="Common PQ relative amplitude across penetrations (fair comparison)",
    )
    parser.add_argument(
        "--criteria",
        type=int,
        default=0,
        help="ANDES TDS criteria flag (0 recommended for WECC+IBR attack sweeps)",
    )
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument(
        "--waveform-kind",
        choices=["sine", "square"],
        default="sine",
        help="Sweep waveform; use square to reproduce the previous experiment",
    )
    parser.add_argument(
        "--penetrations",
        type=float,
        nargs="*",
        default=None,
        help="Override baseline-valid list",
    )
    args = parser.parse_args()
    excluded: list[dict[str, Any]] = []
    if args.penetrations:
        pens = [float(x) for x in args.penetrations]
    else:
        pens, excluded = load_viable_penetrations(args.baseline)
    freqs = np.arange(args.start, args.stop + 1e-9, args.step)
    # Remove stale CSV so schema/metrics are not mixed with prior runs.
    if args.output.exists():
        args.output.unlink()
    run_2d_sweep(
        penetrations=pens,
        frequencies=freqs,
        output=args.output,
        heatmap=args.heatmap,
        overlay=args.overlay,
        summary=args.summary,
        amplitude_frac=args.amplitude_frac,
        duration_s=args.duration_s,
        criteria=args.criteria,
        excluded_penetrations=excluded,
        waveform_kind=args.waveform_kind,
    )


if __name__ == "__main__":
    main()

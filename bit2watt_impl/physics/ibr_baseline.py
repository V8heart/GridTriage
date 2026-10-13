"""Phase 2: IBR penetration baselines on public WECC 179 GENCLS."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bit2watt_impl.physics.ibr_penetration import mean_online_frequency_hz
from bit2watt_impl.physics.simulation import (
    build_wecc_system_at_penetration,
    ensure_parent,
    run_to,
)


DEFAULT_TARGETS = (0.0, 0.3, 0.5, 0.7)
NOMINAL_FREQ_HZ = 60.0
FREQ_TOL_HZ = 0.05


def evaluate_penetration(
    target: float,
    *,
    tds_tf_s: float = 17.0,
    nominal_freq_hz: float = NOMINAL_FREQ_HZ,
    freq_tol_hz: float = FREQ_TOL_HZ,
) -> dict:
    row: dict = {
        "test_system": "wecc_179_gencls",
        "scope": "public dynamic test system; not a real regional grid",
        "target_penetration": float(target),
        "achieved_penetration": None,
        "replaced_count": None,
        "inertia_before": None,
        "inertia_after": None,
        "inertia_reduction_frac": None,
        "pflow_ok": False,
        "tds_ok": False,
        "valid_operating_point": False,
        "f_hz": None,
        "t_reached_s": None,
        "failure_reason": "",
        "excluded_reason": "",
        "replaced_idx": "",
    }
    try:
        ss = build_wecc_system_at_penetration(target, require_pflow=False)
    except Exception as exc:
        row["failure_reason"] = f"build:{type(exc).__name__}: {exc}"
        row["excluded_reason"] = row["failure_reason"]
        return row

    meta = getattr(ss, "_gridtriage_ibr_meta", {}) or {}
    row.update(
        {
            "achieved_penetration": meta.get("achieved_penetration"),
            "replaced_count": meta.get("replaced_count"),
            "inertia_before": meta.get("inertia_before"),
            "inertia_after": meta.get("inertia_after"),
            "inertia_reduction_frac": meta.get("inertia_reduction_frac"),
            "replaced_idx": ",".join(meta.get("replaced_idx") or []),
        }
    )
    pflow_ok = bool(getattr(ss, "_gridtriage_pflow_ok", False))
    row["pflow_ok"] = pflow_ok
    if not pflow_ok:
        row["failure_reason"] = "power flow did not converge"
        row["excluded_reason"] = row["failure_reason"]
        return row

    ss.TDS.config.criteria = 1
    ss.TDS.config.tstep = 1.0 / 30.0
    ss.TDS.config.shrinkt = 1
    if hasattr(ss.TDS, "init"):
        ss.TDS.init()
    ok, _, reason = run_to(ss, float(tds_tf_s), max_chunk_s=0.5)
    row["tds_ok"] = bool(ok)
    row["t_reached_s"] = float(getattr(ss.dae, "t", 0.0) or 0.0)
    if not ok:
        row["failure_reason"] = reason
        row["excluded_reason"] = f"idle_tds_failed:{reason}"
        return row
    f_hz = mean_online_frequency_hz(ss)
    row["f_hz"] = f_hz
    if f_hz is None:
        row["excluded_reason"] = "frequency_unavailable"
        return row
    if abs(float(f_hz) - float(nominal_freq_hz)) > float(freq_tol_hz):
        row["excluded_reason"] = (
            f"frequency_off_nominal:{f_hz:.6f}_vs_{nominal_freq_hz}±{freq_tol_hz}"
        )
        return row
    row["valid_operating_point"] = True
    return row


def run_baseline(
    targets: tuple[float, ...] | list[float] = DEFAULT_TARGETS,
    *,
    output: Path,
    tds_tf_s: float = 17.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for target in targets:
        row = evaluate_penetration(float(target), tds_tf_s=tds_tf_s)
        rows.append(row)
        status = "valid" if row["valid_operating_point"] else (row["excluded_reason"] or "failed")
        print(
            f"[IBR baseline] target={target:.0%} "
            f"achieved={row['achieved_penetration']} "
            f"n={row['replaced_count']} "
            f"M:{row['inertia_before']}->{row['inertia_after']} "
            f"pflow={row['pflow_ok']} tds={row['tds_ok']} "
            f"f_hz={row['f_hz']} valid={row['valid_operating_point']} ({status})"
        )
        ensure_parent(output)
        pd.DataFrame(rows).to_csv(output, index=False)

    frame = pd.DataFrame(rows)
    ensure_parent(output)
    frame.to_csv(output, index=False)
    viable = frame[frame["valid_operating_point"].astype(bool)]
    if viable.empty:
        print("[IBR baseline] no valid operating points")
    else:
        max_t = float(viable["target_penetration"].max())
        print(f"[IBR baseline] max valid target_penetration={max_t:.0%}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "WECC GENCLS→REGCA1 penetration baselines (public test system only)"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/ibr_penetration_baseline.csv"),
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGETS),
    )
    parser.add_argument(
        "--tds-tf-s",
        type=float,
        default=17.0,
        help="Idle TDS horizon; long enough to catch late idle divergence (e.g. 70%)",
    )
    args = parser.parse_args()
    run_baseline(args.targets, output=args.output, tds_tf_s=args.tds_tf_s)


if __name__ == "__main__":
    main()

"""Session-level calibration of the Stage1 alarm (plan_rag_pipeline_v4.md §9.1).

Window conformal p-values assume windows are exchangeable, but the real shift is
between sessions, so the window guarantee breaks (cal 0.05 -> test 0.18). Here
the alarm is decided per session from a session statistic, and the threshold is
calibrated on *normal sessions*:

  R  longest run of consecutive Stage1 candidate windows
  C  max of a conformal CUSUM over non-overlapping windows
  L  max over 4 consecutive non-overlapping windows of sum(-log p)

Cross-conformal over the three round5b folds: their test splits partition the 78
sessions, so every session has out-of-fold (OOF) window scores from the model that
never saw it. Fold f is alarmed with the threshold from the OOF normal sessions of
the other two folds. Holdout and misuse sessions (scored by fold_00) use all 78 OOF
normal sessions. Guarantee targeted: P(normal session alarmed) <= alpha.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.stage1_alternatives import scored_windows
from pipeline.v4_core import REFIT, ROOT

FOLDS = ("fold_00", "fold_01", "fold_02")
SESSION_GATE = REFIT / "session_gate.json"
INVISIBLE = {"swma_piggyback", "swma_mimicry"}
STATS = ("R", "C", "L")


def _non_overlapping(p: pd.Series) -> np.ndarray:
    # stride 15 s, window 30 s: every other window is disjoint
    return np.clip(p.to_numpy()[::2].astype(float), 1e-6, 1.0)


def session_stats(part: pd.DataFrame, *, eps: float = 0.92, block: int = 4) -> dict:
    part = part.sort_values("start")
    cand = part["cand0"].to_numpy(dtype=bool)
    best = cur = 0
    for c in cand:
        cur = cur + 1 if c else 0
        best = max(best, cur)
    p = _non_overlapping(part["p0"])
    s, cmax = 1.0, 1.0
    for pi in p:
        s = max(1.0, s) * eps * pi ** (eps - 1.0)
        cmax = max(cmax, s)
    nl = -np.log(p)
    lmax = max((nl[i:i + block].sum() for i in range(max(1, len(nl) - block + 1))), default=0.0)
    return {"R": float(best), "C": float(np.log(cmax)), "L": float(lmax)}


def session_table(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, gid), part in windows[windows["role"] == "target"].groupby(["session_id", "gpu_id"]):
        attack = bool(part["attack"].any())
        rows.append({"session_id": sid, "gpu_id": int(gid), "variant": str(part["variant"].iloc[0]),
                     "family": str(part["family"].iloc[0]), "attack": attack,
                     "visible": attack and str(part["variant"].iloc[0]) not in INVISIBLE,
                     **session_stats(part)})
    return pd.DataFrame(rows)


def conformal_threshold(cal: np.ndarray, alpha: float) -> float:
    """Smallest t with (1 + #{cal >= t}) / (n + 1) <= alpha; alarms when stat >= t."""
    cal = np.sort(np.asarray(cal, dtype=float))
    n = len(cal)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(np.nextafter(cal[k - 1], math.inf))


def evaluate(oof: pd.DataFrame, extra: pd.DataFrame, alphas=(0.05, 0.10)) -> dict:
    out = {}
    normal_oof = oof[~oof["attack"]]
    for stat in STATS:
        for alpha in alphas:
            alarms = []
            for f in FOLDS:
                cal = normal_oof[normal_oof["fold"] != f][stat].to_numpy()
                thr = conformal_threshold(cal, alpha)
                part = oof[oof["fold"] == f]
                alarms.append(part.assign(alarm=part[stat] >= thr))
            a = pd.concat(alarms)
            thr_all = conformal_threshold(normal_oof[stat].to_numpy(), alpha)
            e = extra.assign(alarm=extra[stat] >= thr_all)
            out[f"{stat}@{alpha}"] = _metrics(a, e, thr_all)
    for n_runs in (2, 4, 6):  # fixed persistence gates for comparison
        a = oof.assign(alarm=oof["R"] >= n_runs)
        e = extra.assign(alarm=extra["R"] >= n_runs)
        out[f"fixed_R>={n_runs}"] = _metrics(a, e, float(n_runs))
    return out


def _metrics(a: pd.DataFrame, e: pd.DataFrame, thr: float) -> dict:
    def rate(frame, mask):
        sub = frame[mask]
        return round(float(sub["alarm"].mean()), 3) if len(sub) else None

    return {
        "threshold_all": thr,
        "oof_normal_alarm": rate(a, ~a["attack"]),
        "oof_normal_alarm_by_fold": {f: rate(a, (~a["attack"]) & (a["fold"] == f)) for f in FOLDS},
        "oof_visible_attack_alarm": rate(a, a["visible"]),
        "oof_attack_alarm_by_variant": a[a["attack"]].groupby("variant")["alarm"].mean().round(3).to_dict(),
        "holdout_new_serving_alarm": rate(e, e["variant"] == "bursty"),
        "holdout_ltma_alarm": rate(e, e["variant"] == "ltma_basic"),
        "misuse_alarm_by_variant": e[e["attack"] & (e["variant"] != "ltma_basic")].groupby("variant")["alarm"].mean().round(3).to_dict(),
        "n_oof_normal": int((~a["attack"]).sum()), "n_oof_visible_attack": int(a["visible"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/session_calibration.json")
    parser.add_argument("--alpha", type=float, default=0.10, help="session alarm level written to the gate")
    args = parser.parse_args()
    oof_parts = []
    for f in FOLDS:
        w = scored_windows("round5b", f, [])
        test = set(json.loads((REFIT / "round5b" / f / "split_manifest.json").read_text())["test"])
        t = session_table(w[w["session_id"].isin(test)])
        oof_parts.append(t.assign(fold=f))
    oof = pd.concat(oof_parts, ignore_index=True)
    w0 = scored_windows("round5b", "fold_00", [])
    known = set(oof["session_id"])
    holdout = session_table(w0[~w0["session_id"].isin(known)])
    misuse = session_table(scored_windows("round5b", "fold_00", [ROOT / "dataset/real/pipeline/telemetry_misuse_v1.csv"],
                                          tag="_misuse"))
    extra = pd.concat([holdout, misuse], ignore_index=True)
    report = {"n_oof_sessions": len(oof), "n_extra_sessions": len(extra), "results": evaluate(oof, extra)}
    oof.to_csv(args.out.with_suffix(".oof.csv"), index=False)
    extra.to_csv(args.out.with_suffix(".extra.csv"), index=False)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    thr = conformal_threshold(oof[~oof["attack"]]["R"].to_numpy(), args.alpha)
    gate = {"statistic": "R (longest run of consecutive Stage1 candidate windows)",
            "alpha": args.alpha, "min_run_windows": int(math.floor(thr)) + 1 if math.isfinite(thr) else None,
            "n_normal_sessions": int((~oof["attack"]).sum()), "stage1": "round5b",
            "calibration": "cross-conformal over round5b fold test splits (out-of-fold sessions)",
            "oof_normal_alarm": report["results"][f"R@{args.alpha}"]["oof_normal_alarm"]}
    SESSION_GATE.write_text(json.dumps(gate, indent=1))
    print("session gate:", gate)
    rows = []
    for name, m in report["results"].items():
        rows.append({"rule": name, "normal_alarm": m["oof_normal_alarm"],
                     "by_fold": "/".join(str(v) for v in m["oof_normal_alarm_by_fold"].values()),
                     "visible_attack": m["oof_visible_attack_alarm"], "new_serving": m["holdout_new_serving_alarm"],
                     "ltma_holdout": m["holdout_ltma_alarm"],
                     "misuse": json.dumps(m["misuse_alarm_by_variant"])})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

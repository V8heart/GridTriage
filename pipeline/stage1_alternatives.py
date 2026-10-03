"""Stage1 structural alternatives vs the official round5b scorer (plan_rag_pipeline_v4.md §6).

  S0   round5b: per-feature cohort tail-min + evidence surprisal -> Mondrian conformal
  SA   multivariate: kNN distance (or IsolationForest) in cohort robust-z space,
       tail p vs train normals (leave-one-session-out) + the same evidence surprisal
       -> the same Mondrian conformal on cal normals
  SB   conformal CUSUM over each GPU's non-overlapping window p-values (S0 p),
       alarm threshold set on cal normals only

All thresholds come from train/cal; test (fold split) and the 12-session holdout
are only scored. Results: dataset/eval/stage1_alternatives.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.stage1_v2 import (
    _select_u_scores,
    build_windows,
    conformal_pvalue,
    empirical_tail_p,
    load_truth_index,
    score_features,
    score_window,
)
from pipeline.v4_core import REFIT, ROOT, load_stage1

CACHE_DIR = ROOT / "dataset" / "eval"


def scored_windows(model: str, fold: str, telemetry: list[Path], *, tag: str = "") -> pd.DataFrame:
    cache = CACHE_DIR / f"stage1_alt_windows_{model}_{fold}{tag}.pkl"
    if cache.exists():
        cached = pd.read_pickle(cache)
        if cached.attrs.get("key") == f"{model}/{fold}":
            return cached
    s1 = load_stage1(model, fold)
    truth = load_truth_index(ROOT / "dataset/real/index/labels.csv", sessions_root=ROOT / "dataset/real/sessions")
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in telemetry], ignore_index=True)
    w = build_windows(df, window_s=30, stride_s=15, config=s1.config, sessions_root=ROOT / "dataset/real/sessions",
                      truth_index=truth, progress_source="raw").reset_index(drop=True)
    feats = sorted(score_features(s1.baseline, s1.config))
    out = []
    for _, row in w.iterrows():
        data = row.to_dict()
        sc = score_window(data, s1.baseline, s1.calibration, s1.config)
        z = s1.baseline.robust_z(data)
        info = truth[(str(data["session_id"]), int(data["gpu_id"]))]
        out.append({
            "session_id": data["session_id"], "gpu_id": int(data["gpu_id"]), "start": int(data["start"]),
            "role": info.get("gpu_role"), "variant": info.get("gt_variant") or info.get("session_gt_label"),
            "gt_label": data["gt_label"], "attack": bool(data["gt_is_attack"]),
            "family": str(data.get("declared_job_family")), "band": str(data.get("expected_band")),
            "n_procs": data.get("observed_n_procs"), "mean_w": data.get("mean_w"),
            "idle": sc["idle_status"], "p0": sc["p_value"], "cand0": sc["is_candidate"],
            "evidence_score": sc["evidence_score"],
            **{f"z_{k}": abs(z.get(k, np.nan)) for k in feats},
        })
    frame = pd.DataFrame(out)
    frame.attrs["key"] = f"{model}/{fold}"
    frame.to_pickle(cache)
    return frame


def _bucket(frame: pd.DataFrame) -> pd.Series:
    return frame["family"] + "|" + frame["band"]


def multivariate_scores(frame: pd.DataFrame, train_mask: pd.Series, method: str, k: int = 10) -> np.ndarray:
    """Per-cohort novelty score from train normals; train rows scored leave-one-session-out."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    zcols = [c for c in frame if c.startswith("z_")]
    X = frame[zcols].fillna(0.0).clip(upper=50).to_numpy()
    scores = np.full(len(frame), np.nan)
    buckets = _bucket(frame)
    for b in buckets.unique():
        ref_idx = np.where(train_mask & (buckets == b))[0]
        if len(ref_idx) < 30:  # same fallback idea as cohort baselines: family, then global
            fam = b.split("|")[0]
            ref_idx = np.where(train_mask & (frame["family"] == fam))[0]
        if len(ref_idx) < 30:
            ref_idx = np.where(train_mask)[0]
        rows = np.where(buckets == b)[0]
        sess_all = frame["session_id"].to_numpy()
        fam = b.split("|")[0]
        for sess in np.unique(sess_all[rows]):
            tgt = rows[sess_all[rows] == sess]
            # leave the scored session out; widen bucket -> family -> global if too few remain
            ref = ref_idx[sess_all[ref_idx] != sess]
            if len(ref) < 30:
                ref = np.where(train_mask & (frame["family"] == fam) & (sess_all != sess))[0]
            if len(ref) < 30:
                ref = np.where(train_mask & (sess_all != sess))[0]
            if method == "knn":
                nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(X[ref])
                scores[tgt] = nn.kneighbors(X[tgt])[0].mean(axis=1)
            else:
                forest = IsolationForest(n_estimators=200, random_state=0).fit(X[ref])
                scores[tgt] = -forest.score_samples(X[tgt])
    return scores


def conformal_from_scores(frame: pd.DataFrame, raw: np.ndarray, train_mask, cal_mask, alpha: float,
                          *, return_p: bool = False):
    """Feature tail vs train normals + evidence surprisal -> Mondrian conformal on cal normals."""
    buckets = _bucket(frame).to_numpy()
    ev = frame["evidence_score"].to_numpy()
    u = np.empty(len(frame))
    for b in np.unique(buckets):
        ref = raw[train_mask.to_numpy() & (buckets == b)]
        if len(ref) < 30:
            ref = raw[train_mask.to_numpy()]
        for i in np.where(buckets == b)[0]:
            u[i] = -np.log(max(empirical_tail_p(raw[i], ref), 1e-12)) + ev[i]
    p = np.empty(len(frame))
    for b in np.unique(buckets):
        refs = u[cal_mask.to_numpy() & (buckets == b)]
        if len(refs) < 30:
            refs = u[cal_mask.to_numpy()]
        for i in np.where(buckets == b)[0]:
            p[i] = conformal_pvalue(u[i], list(refs))
    cand = p <= alpha
    idle = (frame["idle"] == "idle").to_numpy()
    cand[idle] = False
    return (cand, p) if return_p else cand


def conformal_cusum(frame: pd.DataFrame, cal_mask: pd.Series, *, eps: float = 0.92, fp_target: float = 0.05,
                    p_col: str = "p0") -> np.ndarray:
    """Conformal CUSUM S_n = max(1, S_{n-1}) * eps * p_n^(eps-1) over non-overlapping windows."""
    stat = np.zeros(len(frame))
    for _, part in frame.groupby(["session_id", "gpu_id"]):
        order = part.sort_values("start").index.to_numpy()
        s = 1.0
        for j, i in enumerate(order):
            if j % 2 == 0:  # stride 15 s, window 30 s: every other window is disjoint
                p = max(float(frame.at[i, p_col]), 1e-6)
                s = max(1.0, s) * eps * p ** (eps - 1.0)
            stat[i] = s
    normal_cal = (cal_mask & ~frame["attack"] & (frame["role"] == "target")).to_numpy()
    threshold = float(np.quantile(stat[normal_cal], 1 - fp_target))
    cand = stat >= threshold
    cand[(frame["idle"] == "idle").to_numpy()] = False
    return cand, threshold


def summarize(frame: pd.DataFrame, cand: np.ndarray, mask: pd.Series) -> dict:
    sub = frame[mask & (frame["role"] == "target")]
    c = pd.Series(cand, index=frame.index)[sub.index]
    normals = sub[~sub["attack"]]
    attacks = sub[sub["attack"]]
    per_variant = c[attacks.index].groupby(attacks["variant"]).mean().round(3).to_dict()
    normal_by_variant = c[normals.index].groupby(normals["variant"]).mean().round(3).to_dict()
    visible = attacks[~attacks["variant"].isin(["swma_piggyback", "swma_mimicry"])]
    return {"normal_fp": round(float(c[normals.index].mean()), 3), "attack_recall": round(float(c[attacks.index].mean()), 3),
            "visible_recall": round(float(c[visible.index].mean()), 3), "n_normal": len(normals), "n_attack": len(attacks),
            "per_variant": per_variant, "normal_by_variant": normal_by_variant}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="round5b")
    parser.add_argument("--fold", default="fold_00")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage1_alternatives.json")
    args = parser.parse_args()
    frame = scored_windows(args.model, args.fold, [ROOT / "dataset/real/pipeline/telemetry_v4_inst.csv",
                                                   ROOT / "dataset/real/pipeline/telemetry_holdout_v2.csv"])
    manifest = json.loads((REFIT / args.model / args.fold / "split_manifest.json").read_text())
    split = frame["session_id"].map({s: k for k in ("train", "cal", "test") for s in manifest.get(k, [])}).fillna("holdout")
    normal_target = ~frame["attack"] & (frame["role"] == "target") & ~frame["gt_label"].astype(str).eq("normal_idle")
    train_mask = (split == "train") & normal_target
    cal_mask = split == "cal"
    alpha = load_stage1(args.model, args.fold).config["alpha"]

    results = {"S0_round5b": frame["cand0"].to_numpy()}
    for method in ("knn", "iforest"):
        raw = multivariate_scores(frame, train_mask, method)
        results[f"SA_{method}"], frame[f"p_{method}"] = conformal_from_scores(
            frame, raw, train_mask, cal_mask & normal_target, alpha, return_p=True)
    sb, thr = conformal_cusum(frame, cal_mask)
    results["SB_cusum"] = sb
    results["SAB_knn_cusum"], _ = conformal_cusum(frame, cal_mask, p_col="p_knn")
    results["SAB_iforest_cusum"], _ = conformal_cusum(frame, cal_mask, p_col="p_iforest")
    report = {"alpha": alpha, "cusum_threshold": thr, "results": {}}
    for name, cand in results.items():
        report["results"][name] = {sp: summarize(frame, cand, split == sp) for sp in ("cal", "test", "holdout")}
        r = report["results"][name]
        print(f"{name:12s} test FP {r['test']['normal_fp']:.3f} recall {r['test']['attack_recall']:.3f} "
              f"(visible {r['test']['visible_recall']:.3f}) | holdout FP {r['holdout']['normal_fp']:.3f} "
              f"LTMA {r['holdout']['per_variant'].get('ltma_basic')} serving {r['holdout']['normal_by_variant'].get('bursty')}")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

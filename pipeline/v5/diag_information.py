"""Is the ~60% Stage2 accuracy a data limit or a pipeline limit?

Supervised, leave-one-session-out classifiers give an *upper bound* on how much class
information a feature set carries (labels are used, so this is a ceiling, not a detector):

  bundle      the v4 evidence bundle (pattern numbers + evidence rates + declared family):
              exactly what the LLM judge sees
  power       raw power + instant power windows
  power_util  + GPU utilisation
  raw         + framebuffer memory + process count (channels v4 drops)
  raw_decl    + declared job family

If ``bundle`` is far below ``raw``, the bundle throws information away (pipeline issue).
If even ``raw`` confuses two classes, power telemetry cannot separate them (data issue).

Open-set check (the 'unknown' problem): each attack class is removed from training; we
ask whether its segments look *novel* (kNN distance / low max-probability) relative to
held-out sessions of known classes, and what a closed-set model calls them.

  python -m pipeline.v5.diag_information            # -> dataset/eval/v5/diag_information.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pipeline.v5.raw_features import ALL_RAW, MEM_FEATURES, POWER_FEATURES, PROC_FEATURES, UTIL_FEATURES, build_windows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "v5"
SEGMENTS = [ROOT / "dataset/eval/stage2_v4_segments.jsonl", ROOT / "dataset/eval/stage2_v4_pristine_segments.jsonl"]
FAMILIES = ["training", "inference", "interactive", "hpo", "data", "evaluation"]
ATTACKS = ["swma", "ltma", "cryptojacking", "llmjacking", "hidden_ml_training", "sponge_attack", "burst_ramp_attack"]


def _clf(seed: int = 0) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)


def _decl(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"decl_{f}": (df["declared_job_family"] == f).astype(float) for f in FAMILIES}, index=df.index)


def loso_segments(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seg: pd.Series) -> pd.DataFrame:
    """Leave-one-session-out; window probabilities averaged per segment (session x gpu)."""
    rows = []
    for g in groups.unique():
        test = groups == g
        clf = _clf().fit(X[~test], y[~test])
        proba = pd.DataFrame(clf.predict_proba(X[test]), columns=clf.classes_, index=X.index[test])
        for s, p in proba.groupby(seg[test]):
            mean = p.mean()
            rows.append({"seg": s, "session_id": g, "gt": y[test][seg[test] == s].iloc[0], "pred": mean.idxmax(),
                         "max_p": float(mean.max())})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    res = res.assign(ok=res["gt"] == res["pred"])
    per = res.groupby("gt")["ok"].mean().round(3).to_dict()
    return {"n_segments": int(len(res)), "accuracy": round(float(res.ok.mean()), 3),
            "macro_recall": round(float(np.mean(list(per.values()))), 3), "per_class_recall": per,
            "confusion": {gt: g.pred.value_counts().to_dict() for gt, g in res.groupby("gt")}}


def bundle_frame() -> pd.DataFrame:
    rows = []
    for path in SEGMENTS:
        for line in path.read_text().splitlines():
            s = json.loads(line)
            if s["scope"] != "all":
                continue
            pat = {f"pat_{k}": v for k, v in (s["bundle"].get("pattern") or {}).items() if isinstance(v, (int, float))}
            rates = {f"ev_{k}": float(v) for k, v in (s.get("evidence_rates") or {}).items()}
            fam = (s["bundle"].get("declared_context") or {}).get("job_family")
            u = s["bundle"].get("unexplainedness") or {}
            rows.append({"seg": s["seg_id"], "session_id": s["session_id"], "cls": s["gt_card"],
                         "declared_job_family": fam, "u_score": u.get("u_score"), "p_value": u.get("p_value"),
                         **pat, **rates})
    return pd.DataFrame(rows)


def loso_bundle(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c.startswith(("pat_", "ev_")) or c in ("u_score", "p_value")]
    X = pd.concat([df[feats].astype(float).fillna(0.0), _decl(df)], axis=1)
    rows = []
    for g in df.session_id.unique():
        test = df.session_id == g
        clf = _clf().fit(X[~test], df.cls[~test])
        for i, p in zip(df.index[test], clf.predict_proba(X[test])):
            rows.append({"seg": df.seg[i], "session_id": g, "gt": df.cls[i], "pred": clf.classes_[np.argmax(p)],
                         "max_p": float(np.max(p))})
    return pd.DataFrame(rows)


def open_set(win: pd.DataFrame, feats: list[str], k: int = 5) -> dict:
    """Remove one attack class from training; is it novel relative to held-out known sessions?"""
    X = win[feats].astype(float).fillna(0.0)
    seg = win.session_id + ":" + win.gpu_id.astype(str)
    out = {}
    sessions = win.session_id.unique()
    rng = np.random.default_rng(0)
    for novel in [c for c in ATTACKS if c in set(win.cls)]:
        nov_mask = win.cls == novel
        known_sessions = win.loc[~nov_mask, "session_id"].unique()
        # held-out known sessions: 5 folds over known sessions, so known test is out-of-sample too
        folds = np.array_split(rng.permutation(known_sessions), 5)
        seg_scores = []
        closed_preds = {}
        for fold in folds:
            test_known = win.session_id.isin(fold)
            train = ~nov_mask & ~test_known
            scaler = StandardScaler().fit(X[train])
            Xtr = scaler.transform(X[train])
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xtr)
            ref = np.sort(nn.kneighbors(Xtr)[0][:, 1:].mean(axis=1))  # leave-self-out train distances
            clf = _clf().fit(X[train], win.cls[train])
            for mask, is_novel in ((test_known, 0), (nov_mask, 1)):
                if is_novel and fold is not folds[0]:
                    continue  # score the novel class once
                if not mask.any():
                    continue
                d = nn.kneighbors(scaler.transform(X[mask]), n_neighbors=k)[0].mean(axis=1)
                pct = np.searchsorted(ref, d) / len(ref)  # distance percentile vs training
                proba = clf.predict_proba(X[mask])
                frame = pd.DataFrame({"seg": seg[mask].values, "pct": pct, "maxp": proba.max(axis=1),
                                      "pred": clf.classes_[proba.argmax(axis=1)]})
                for s, g in frame.groupby("seg"):
                    seg_scores.append({"seg": s, "novel": is_novel, "knn_pct": float(g.pct.median()),
                                       "one_minus_maxp": float(1 - g.maxp.mean())})
                    if is_novel:
                        closed_preds[s] = g.pred.value_counts().idxmax()
        sc = pd.DataFrame(seg_scores)
        out[novel] = {
            "n_novel_segments": int(sc.novel.sum()), "n_known_segments": int((sc.novel == 0).sum()),
            "auroc_knn": round(float(roc_auc_score(sc.novel, sc.knn_pct)), 3),
            "auroc_maxp": round(float(roc_auc_score(sc.novel, sc.one_minus_maxp)), 3),
            "novel_flagged_at_known_fpr10": round(float(
                (sc[sc.novel == 1].knn_pct > np.quantile(sc[sc.novel == 0].knn_pct, 0.90)).mean()), 3),
            "closed_set_called_it": pd.Series(closed_preds).value_counts().to_dict(),
        }
    return out


def to_markdown(rep: dict) -> str:
    lines = ["# v5 정보 상한 진단", "", "세션 단위 교차검증(leave-one-session-out) 지도학습 상한. 라벨을 쓰므로 탐지기가 아니라 **정보량 상한**이다.", "",
             "| 특징 집합 | 세그먼트 | 정확도 | 매크로 재현율 |", "|---|---|---|---|"]
    for name, r in rep["closed_set"].items():
        lines.append(f"| {name} | {r['n_segments']} | {r['accuracy']} | {r['macro_recall']} |")
    classes = sorted({c for r in rep["closed_set"].values() for c in r["per_class_recall"]})
    lines += ["", "## 클래스별 재현율", "", "| 클래스 | " + " | ".join(rep["closed_set"]) + " |",
              "|---|" + "---|" * len(rep["closed_set"])]
    for c in classes:
        lines.append(f"| {c} | " + " | ".join(str(r["per_class_recall"].get(c, "-")) for r in rep["closed_set"].values()) + " |")
    lines += ["", "## 혼동 (raw)", ""]
    for gt, row in rep["closed_set"]["raw"]["confusion"].items():
        lines.append(f"- {gt}: {row}")
    lines += ["", "## 처음 보는 공격의 새로움 (해당 클래스를 학습에서 제거)", "",
              "| 제거 클래스 | 특징 | AUROC kNN | AUROC 1-maxP | 정상 FPR 10%에서 새로움 검출 | 닫힌 분류기가 부른 이름 |", "|---|---|---|---|---|---|"]
    for feats, block in rep["open_set"].items():
        for c, r in block.items():
            lines.append(f"| {c} | {feats} | {r['auroc_knn']} | {r['auroc_maxp']} | {r['novel_flagged_at_known_fpr10']} | {r['closed_set_called_it']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "windows_raw.pkl"
    if cache.exists() and not args.rebuild:
        win = pd.read_pickle(cache)
    else:
        win = build_windows()
        win.to_pickle(cache)
    print("windows", len(win), "sessions", win.session_id.nunique(), win.groupby("cls").session_id.nunique().to_dict(), flush=True)
    seg = win.session_id + ":" + win.gpu_id.astype(str)
    sets = {"power": POWER_FEATURES, "power_util": POWER_FEATURES + UTIL_FEATURES, "raw": ALL_RAW}
    rep = {"closed_set": {}, "open_set": {}}
    bundle = bundle_frame()
    rep["closed_set"]["bundle(LLM 입력)"] = summarize(loso_bundle(bundle))
    print("bundle", rep["closed_set"]["bundle(LLM 입력)"]["accuracy"], flush=True)
    for name, feats in sets.items():
        rep["closed_set"][name] = summarize(loso_segments(win[feats].fillna(0.0), win.cls, win.session_id, seg))
        print(name, rep["closed_set"][name]["accuracy"], flush=True)
    Xd = pd.concat([win[ALL_RAW].fillna(0.0), _decl(win)], axis=1)
    rep["closed_set"]["raw_decl"] = summarize(loso_segments(Xd, win.cls, win.session_id, seg))
    for name in ("power", "raw"):
        rep["open_set"][name] = open_set(win, sets[name])
        print("open_set", name, {c: r["auroc_knn"] for c, r in rep["open_set"][name].items()}, flush=True)
    (OUT / "diag_information.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    (OUT / "diag_information.md").write_text(to_markdown(rep))
    print(to_markdown(rep))


if __name__ == "__main__":
    main()

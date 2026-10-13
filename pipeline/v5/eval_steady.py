"""Steady-state check: v4 vs v5 after the analyst has enrolled a few cases ("a few days into operation").

No retraining and no new threat cards: enrolled sessions only add 30 s feature windows to the case memory.
Fixed before running (sessions chosen by sorted session id, never by result):

  S1 (preparation)  1 LTMA session + 1 session of every normal variant with >= 2 evaluation sessions
  S2 (after a few days)  S1 + 1 session of each misuse threat (sponge, llmjacking, hidden ML training, burst)

All systems are compared on the same segments: the evaluation set of eval_v5 P1 minus every session
enrolled in S1 or S2.
  v4        stored A5 rows (v4 has no case memory, so enrolment cannot change it)
  v5_day0   stored eval_v5 P1 rows, gate (a)+(b) (memory = seed only)
  v5_S1 / v5_S2   re-judged with the enlarged memory, gates (c)+(a)+(b), enrolled classes = enrolled labels
'(b)-off' variants are re-derived from the stored pre-gate LLM verdict (no extra LLM call).

  python -m pipeline.v5.eval_steady
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline.kb.query import render_query
from pipeline.kb.retriever import HybridRetriever
from pipeline.stage2_rag_judge import is_correct
from pipeline.v5.eval_v5 import MODEL, eval_segments, harmful, md, memory_sessions, v4_baseline, windows_of
from pipeline.v5.exemplar_kb import CaseMemory
from pipeline.v5.judge_v5 import CHANNELS, judge_v5, open_set_gate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "v5"
MISUSE = ("sponge_attack", "llmjacking", "hidden_ml_training", "burst_ramp_attack")


def enrolment_plan(segs: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """scenario -> [(session_id, label)]; first session id per type, decided before any judgment."""
    frame = pd.DataFrame([{"sid": s["session_id"], "cls": s["gt_card"], "var": s["gt_variant"] or s["gt_label"]}
                          for s in segs]).drop_duplicates("sid")
    s1 = [(sorted(frame[frame.cls == "ltma"].sid)[0], "ltma")]
    for var, g in frame[frame.cls == "normal"].groupby("var"):
        if g.sid.nunique() >= 2:
            s1.append((sorted(g.sid)[0], "normal"))
    s2 = s1 + [(sorted(frame[frame.cls == c].sid)[0], c) for c in MISUSE]
    return {"S1": s1, "S2": s2}


def run(scenario: str, enrolled: list[tuple[str, str]], segs: list[dict], win: pd.DataFrame, seed_win: pd.DataFrame,
        retriever, hosts: list[str], out: Path) -> None:
    done = set()
    if out.exists():
        done = {(json.loads(l)["system"], json.loads(l)["seg_id"]) for l in out.read_text().splitlines() if l.strip()}
    add = pd.concat([win[win.session_id == sid].assign(cls=label) for sid, label in enrolled], ignore_index=True)
    memory = CaseMemory(pd.concat([seed_win, add], ignore_index=True))
    labels = frozenset(label for _, label in enrolled)
    system = f"v5_{scenario}"
    jobs = [s for s in segs if (system, s["seg_id"]) not in done]
    print(f"[{scenario}] enrolled={len(enrolled)} classes={sorted(labels)} memory={memory.classes} jobs={len(jobs)}", flush=True)
    lock = threading.Lock()

    def work(i, seg):
        hits = retriever.search(render_query(seg["bundle"]), k=8, mode="hybrid_rerank")
        part = windows_of(win, seg)
        fam = (seg["bundle"].get("declared_context") or {}).get("job_family")
        case = memory.score(part, fam)
        channels = memory.normal_quantiles(part, list(CHANNELS))
        j = judge_v5(seg["bundle"], [h.doc for h in hits], case, channels, model=MODEL, host=hosts[i % len(hosts)],
                     n_windows=seg.get("n_windows"), normal_rule=True, enrolled=labels)
        row = {"system": system, "seg_id": seg["seg_id"], "gt_card": seg["gt_card"], "gt_variant": seg["gt_variant"],
               "accept": seg["accept"], "case": case, **j.to_dict()}
        with lock, out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        for f in [pool.submit(work, i, s) for i, s in enumerate(jobs)]:
            f.result()


def without_b(row: dict) -> dict:
    """Same row with gate (b) switched off (stored pre-gate LLM verdict, gate (a) re-applied)."""
    g = row.get("gate") or {}
    if g.get("rule") != "b_unexplained_anomaly":
        return row
    r = dict(row, verdict=g["llm_verdict"], threat_card=g["llm_card"])
    fired = open_set_gate(r["verdict"], r["threat_card"], row["case"], normal_rule=False)
    return dict(r, verdict="unknown", threat_card=None) if fired else r


def report(paths: list[Path], keep: set[str], plan: dict) -> str:
    rows = [json.loads(l) for p in paths if p.exists() for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["seg_id"] in keep]
    accept = {r["seg_id"]: r["accept"] for r in rows}
    base = v4_baseline()
    rows += [dict(base[s], system="v4", accept=accept[s]) for s in keep if s in base and s in accept]
    day0 = [json.loads(l) for l in (OUT / "eval_v5_p1.jsonl").read_text().splitlines() if l.strip()]
    from pipeline.v5.eval_v5 import with_gate_b

    # stored day-0 rows were judged with gate (a) only; (b) is re-applied to their pre-gate LLM verdict
    day0 = [r for r in day0 if r["system"] == "v5" and r["seg_id"] in keep]
    rows += [dict(r, system="v5_day0_no_b") for r in day0]
    rows += [dict(without_b(r), system=r["system"] + "_no_b") for r in list(rows) if r["system"] in ("v5_S1", "v5_S2")]
    rows += [dict(with_gate_b(dict(r, system="v5")), system="v5_day0") for r in day0]
    df = pd.DataFrame(rows)
    df["correct"] = [is_correct(r, r["gt_card"], r["accept"]) for r in df.to_dict("records")]
    df["harmful"] = [harmful(r["gt_card"], r["verdict"], r.get("threat_card"), r["accept"]) for r in df.to_dict("records")]
    order = ["v4", "v5_day0", "v5_day0_no_b", "v5_S1", "v5_S1_no_b", "v5_S2", "v5_S2_no_b"]
    summary = []
    for sys_ in order:
        g = df[df.system == sys_]
        if g.empty:
            continue
        atk = g[~g.gt_card.isin(["normal", "unobservable"])]
        nrm = g[g.gt_card == "normal"]
        summary.append({"system": sys_, "n": len(g), "accuracy": round(g.correct.mean(), 3),
                        "harmful_err": round(g.harmful.mean(), 3), "review_rate(unknown)": round((g.verdict == "unknown").mean(), 3),
                        "attack_caught": round(atk.verdict.isin(["attack", "unknown"]).mean(), 3),
                        "normal_correct": round(nrm.correct.mean(), 3), "normal_fp": round((nrm.verdict == "attack").mean(), 3),
                        "invalid": round((g.verdict == "invalid").mean(), 3)})
    per = df.pivot_table(index="gt_card", columns="system", values="correct", aggfunc="mean").round(2)
    per = per[[c for c in order if c in per.columns]]
    lines = ["# 정상 운영 상태 비교 (사례 등록 후)", "",
             "등록: " + "; ".join(f"{k}: " + ", ".join(f"{lab}({sid[:10]})" for sid, lab in v) for k, v in plan.items()), "",
             md(pd.DataFrame(summary), index=False), "", "## 클래스별 정확도", "", md(per), ""]
    for sys_ in ("v5_S1", "v5_S2"):
        s = df[df.system == sys_]
        if len(s):
            lines += [f"## {sys_} 혼동", "", md(pd.crosstab(s.gt_card, s.verdict + ":" + s.threat_card.fillna("-"))), ""]
            rules = s.gate.map(lambda g: (g or {}).get("rule") or "-").value_counts().to_dict()
            lines += [f"게이트 발동: {rules}", ""]
    text = "\n".join(lines)
    (OUT / "eval_steady.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", nargs="+", default=["http://127.0.0.1:11434"])
    parser.add_argument("--workers-per-host", type=int, default=2)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    mem = memory_sessions()
    segs = eval_segments(mem)
    plan = enrolment_plan(segs)
    enrolled_all = {sid for v in plan.values() for sid, _ in v}
    keep = [s for s in segs if s["session_id"] not in enrolled_all]
    paths = [OUT / f"eval_steady_{k}.jsonl" for k in plan]
    if not args.report_only:
        win = pd.read_pickle(OUT / "windows_raw.pkl")
        seed_win = win[win.session_id.isin(mem)]
        retriever = HybridRetriever()
        hosts = [h for h in args.hosts for _ in range(args.workers_per_host)]
        for (name, enrolled), path in zip(plan.items(), paths):
            run(name, enrolled, keep, win, seed_win, retriever, hosts, path)
    print(report(paths, {s["seg_id"] for s in keep}, plan))


if __name__ == "__main__":
    main()

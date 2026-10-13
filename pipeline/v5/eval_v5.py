"""Evaluate Stage2 v5 against v4 on the segments outside the case memory.

Case memory = round5b fold_00 train+cal sessions (the same sessions Stage1 was fitted on).
Evaluation = every other 'all'-scope segment (fold_00 test, holdout, misuse captures).

  P1 closed   full KB. v5 vs v4 A5 (v4 rows re-read from the existing ablation JSONLs).
  P2 open     leave-one-class-out: class c's card and its cases are removed; c's segments
              are correct only as 'unknown'. v4 (A5 with the card removed) is re-run as baseline.

Metrics: strict accuracy, harmful errors (attack->normal, normal->attack, wrong attack card),
unknown rate. Gate (b) is evaluated post hoc from the stored pre-gate LLM verdict.

  python -m pipeline.v5.eval_v5 --protocol p1 p2
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline.kb.build_kb import build
from pipeline.kb.query import render_query
from pipeline.kb.retriever import HybridRetriever, load_kb
from pipeline.stage2_rag_judge import is_correct, judge
from pipeline.v5.exemplar_kb import CaseMemory
from pipeline.v5.judge_v5 import CHANNELS, open_set_gate, judge_v5

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "v5"
MANIFEST = ROOT / "dataset/real/pipeline/refit/round5b/fold_00/split_manifest.json"
SEGMENTS = [ROOT / "dataset/eval/stage2_v4_segments.jsonl", ROOT / "dataset/eval/stage2_v4_pristine_segments.jsonl"]
V4_RESULTS = [ROOT / "dataset/eval/stage2_v4_ablation.jsonl", ROOT / "dataset/eval/stage2_v4_pristine_all.jsonl"]
LOCO = ["swma", "cryptojacking", "ltma", "sponge_attack", "hidden_ml_training", "llmjacking", "burst_ramp_attack"]
MODEL = "gemma3:12b"


def memory_sessions() -> set[str]:
    m = json.loads(MANIFEST.read_text())
    return set(m["train"]) | set(m["cal"])


def eval_segments(mem_sessions: set[str]) -> list[dict]:
    seen, out = set(), []
    for path in SEGMENTS:  # dev file first: overlapping holdout segments keep the dev bundle
        for line in path.read_text().splitlines():
            s = json.loads(line)
            if s["scope"] == "all" and s["session_id"] not in mem_sessions and s["seg_id"] not in seen:
                seen.add(s["seg_id"])
                s["_source"] = path.name
                out.append(s)
    return out


def v4_baseline() -> dict:
    rows = {}
    for path in V4_RESULTS:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["model"] == MODEL and r["config"] == "A5_rerank" and r["seg_id"] not in rows:
                rows[r["seg_id"]] = r
    return rows


def windows_of(win: pd.DataFrame, seg: dict) -> pd.DataFrame:
    return win[(win.session_id == seg["session_id"]) & (win.gpu_id == int(seg["gpu_id"]))]


def harmful(gt: str, verdict: str, card: str | None, accept: list[str]) -> bool:
    if verdict in ("unknown", "invalid"):
        return False
    if gt == "normal":
        return verdict == "attack"
    if gt == "unobservable":
        return False
    return not (verdict == "attack" and card in accept)


def run(protocol: str, segs: list[dict], win: pd.DataFrame, mem_win: pd.DataFrame, hosts: list[str],
        out: Path) -> None:
    done = set()
    if out.exists():
        done = {(json.loads(l)["system"], json.loads(l)["seg_id"], json.loads(l)["held_out"]) for l in out.read_text().splitlines() if l.strip()}
    lock = threading.Lock()
    groups = [(None, segs)] if protocol == "p1" else [(c, [s for s in segs if s["gt_card"] == c]) for c in LOCO]
    for held, group in groups:
        if not group:
            continue
        docs = build(exclude_cards=(held,)) if held else load_kb()
        retriever = HybridRetriever(docs)
        memory = CaseMemory(mem_win[mem_win.cls != held] if held else mem_win)
        systems = ["v5"] + (["v4"] if held else [])
        jobs = [(sys_, s) for sys_ in systems for s in group if (sys_, s["seg_id"], held) not in done]
        print(f"[{protocol}] held_out={held} segments={len(group)} jobs={len(jobs)} memory={memory.classes}", flush=True)

        def work(i, sys_, seg):
            query = render_query(seg["bundle"])
            hits = retriever.search(query, k=8, mode="hybrid_rerank")
            hdocs = [h.doc for h in hits]
            host = hosts[i % len(hosts)]
            row = {"system": sys_, "protocol": protocol, "held_out": held, "seg_id": seg["seg_id"],
                   "gt_card": seg["gt_card"], "gt_variant": seg["gt_variant"], "accept": seg["accept"],
                   "query": query, "doc_ids": [d["doc_id"] for d in hdocs]}
            if sys_ == "v4":
                j = judge(seg["bundle"], hdocs, model=MODEL, host=host)
            else:
                part = windows_of(win, seg)
                fam = (seg["bundle"].get("declared_context") or {}).get("job_family")
                case = memory.score(part, fam)
                channels = memory.normal_quantiles(part, list(CHANNELS))
                j = judge_v5(seg["bundle"], hdocs, case, channels, model=MODEL, host=host, n_windows=seg.get("n_windows"))
                row.update(case=case, channels=channels)
            row.update(j.to_dict())
            with lock, out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            for f in [pool.submit(work, i, *job) for i, job in enumerate(jobs)]:
                f.result()


def correct_open(row: dict) -> bool:
    if row["held_out"]:
        return row["verdict"] == "unknown"
    return is_correct(row, row["gt_card"], row["accept"])


def with_gate_b(row: dict) -> dict:
    """Re-apply the gate with rule (b) to the stored pre-gate LLM verdict."""
    g = row.get("gate") or {}
    if row["system"] != "v5" or not g or row["verdict"] == "invalid":
        return row
    r = dict(row, verdict=g["llm_verdict"], threat_card=g["llm_card"])
    fired = open_set_gate(r["verdict"], r["threat_card"], row["case"], normal_rule=True)
    if fired:
        r.update(verdict="unknown", threat_card=None)
    return r


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional tabulate dependency."""
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, g in rows.groupby(["protocol", "system"]):
        atk = g[~g.gt_card.isin(["normal", "unobservable"])]
        out.append({"protocol": keys[0], "system": keys[1], "n": len(g),
                    "accuracy": round(g.correct.mean(), 3),
                    "macro_acc": round(g.groupby("gt_card").correct.mean().mean(), 3),
                    "harmful_err": round(g.harmful.mean(), 3),
                    "unknown_rate": round((g.verdict == "unknown").mean(), 3),
                    "attack_detected": round(atk.verdict.isin(["attack", "unknown"]).mean(), 3) if len(atk) else None,
                    "normal_fp": round((g[g.gt_card == "normal"].verdict == "attack").mean(), 3)})
    return pd.DataFrame(out)


def report(paths: list[Path]) -> str:
    rows = [json.loads(l) for p in paths if p.exists() for l in p.read_text().splitlines() if l.strip()]
    base = v4_baseline()
    p1_ids = {r["seg_id"] for r in rows if r["protocol"] == "p1"}
    accept = {r["seg_id"]: r["accept"] for r in rows}
    rows += [dict(base[s], system="v4", protocol="p1", held_out=None, accept=accept[s]) for s in p1_ids if s in base]
    rows += [dict(with_gate_b(r), system="v5+b") for r in rows if r["system"] == "v5"]
    rows += [dict(r, system="v5_llm_only", verdict=r["gate"]["llm_verdict"], threat_card=r["gate"]["llm_card"])
             for r in rows if r["system"] == "v5" and r.get("gate")]
    df = pd.DataFrame(rows)
    df["correct"] = [correct_open(r) for r in df.to_dict("records")]
    df["harmful"] = [harmful(r["gt_card"], r["verdict"], r.get("threat_card"), r["accept"]) for r in df.to_dict("records")]
    lines = ["# Stage2 v5 평가", "", "## 요약", "", md(summarize(df), index=False), ""]
    p1 = df[df.protocol == "p1"]
    if len(p1):
        lines += ["## P1 클래스별 정확도", "", md(p1.pivot_table(index="gt_card", columns="system", values="correct",
                                                              aggfunc="mean").round(2)), ""]
        lines += ["## P1 v5 혼동", ""]
        v5 = p1[p1.system == "v5"]
        lines += [md(pd.crosstab(v5.gt_card, v5.verdict + ":" + v5.threat_card.fillna("-"))), ""]
    p2 = df[df.protocol == "p2"]
    if len(p2):
        lines += ["## P2 클래스 하나 빼기: unknown 판정률 (정답 = unknown)", "",
                  md(p2.pivot_table(index="held_out", columns="system", values="correct", aggfunc="mean").round(2)), "",
                  "### P2 v4 / v5 판정 내역", ""]
        for sys_ in ("v4", "v5"):
            s = p2[p2.system == sys_]
            lines += [f"**{sys_}**", "", md(pd.crosstab(s.held_out, s.verdict + ":" + s.threat_card.fillna("-"))), ""]
    text = "\n".join(lines)
    (OUT / "eval_v5.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", nargs="+", default=["p1", "p2"])
    parser.add_argument("--hosts", nargs="+", default=["http://127.0.0.1:11434"])
    parser.add_argument("--workers-per-host", type=int, default=2)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {p: OUT / f"eval_v5_{p}.jsonl" for p in args.protocol}
    if not args.report_only:
        mem = memory_sessions()
        win = pd.read_pickle(OUT / "windows_raw.pkl")
        segs = eval_segments(mem)
        hosts = [h for h in args.hosts for _ in range(args.workers_per_host)]
        for p in args.protocol:
            run(p, segs, win, win[win.session_id.isin(mem)], hosts, paths[p])
    print(report(list(paths.values())))


if __name__ == "__main__":
    main()

"""Stage2 v4 ablation: does retrieval improve the LLM's final verdict?

Configs (plan_rag_pipeline_v4.md §4.3):
  A0_no_docs     card names only, no document content
  A1_legacy      pre-v4 description query + TF-IDF over card chunks, top-3
  A2_tfidf_kb    v4 query + TF-IDF over the full KB (1,000+ docs), top-k
  A3_all_sigs    every card's observation-signature section (small-KB upper bound)
  A4_hybrid      v4 query + BM25/dense RRF, top-k
  A5_rerank      v4 query + hybrid + cross-encoder rerank, top-k   (proposed)
  A6_random      k random KB docs (placebo)
  A7_multi       legacy description + v4 query, multi-query hybrid + max rerank, top-k
  A9_union       A8 structured cards + A5 text results (deduplicated)
  A8_profile     slot observation profile: structured card match (top-3 cards' profile+signature)
                 + 2 ATT&CK/ATLAS docs by text; the LLM also sees the profile slots

Results are appended to a JSONL so interrupted runs resume.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pipeline.kb.query import render_query
from pipeline.kb.retriever import HybridRetriever, load_kb
from pipeline.stage2_rag_judge import is_correct, judge

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ("A0_no_docs", "A1_legacy", "A2_tfidf_kb", "A3_all_sigs", "A4_hybrid", "A5_rerank", "A6_random", "A7_multi", "A8_profile", "A9_union", "A9b_union_noprof")
CARD_NAMES = sorted(p.stem for p in (ROOT / "corpus").glob("*.md"))


def contexts(segments: list[dict], configs, k: int) -> dict[tuple[str, str], dict]:
    docs = load_kb()
    if any(c in configs for c in ("A8_profile", "A9_union", "A9b_union_noprof")):
        from pipeline.kb.build_kb import build

        docs = build(with_profiles=True)  # slot-profile experiments need the profile chunks
    cards_only = [d for d in docs if d["layer"] == "card"]
    sigs = [d for d in cards_only if d["doc_id"].endswith("#관측 시그니처")]
    full = HybridRetriever(docs)
    small = HybridRetriever(cards_only, load_models=False)
    out = {}
    for seg in segments:
        query = render_query(seg["bundle"])
        rng = random.Random(seg["seg_id"])
        for cfg in configs:
            if cfg == "A0_no_docs":
                ctx = {"docs": [], "all_cards": CARD_NAMES, "show_docs": False, "query": None}
            elif cfg == "A1_legacy":
                hits = small.search(seg["legacy_query"] or query, k=3, mode="tfidf")
                ctx = {"docs": [h.doc for h in hits], "query": seg["legacy_query"]}
            elif cfg == "A2_tfidf_kb":
                ctx = {"docs": [h.doc for h in full.search(query, k=k, mode="tfidf")], "query": query}
            elif cfg == "A3_all_sigs":
                ctx = {"docs": sigs, "query": None}
            elif cfg == "A4_hybrid":
                ctx = {"docs": [h.doc for h in full.search(query, k=k, mode="hybrid")], "query": query}
            elif cfg == "A5_rerank":
                ctx = {"docs": [h.doc for h in full.search(query, k=k, mode="hybrid_rerank")], "query": query}
            elif cfg == "A7_multi":
                hits = full.search_multi([seg["legacy_query"], query], k=k)
                ctx = {"docs": [h.doc for h in hits], "query": f"{seg['legacy_query']} | {query}"}
            elif cfg == "A8_profile":
                from pipeline.kb.profile import profile_slots, render_profile

                slots = profile_slots(seg["bundle"])
                pq = render_profile(slots)
                ctx = {"docs": [h.doc for h in full.search_profile(slots, pq)], "query": pq,
                       "bundle": {**seg["bundle"], "observation_profile": slots}}
            elif cfg in ("A9_union", "A9b_union_noprof"):
                from pipeline.kb.profile import profile_slots, render_profile

                slots = profile_slots(seg["bundle"])
                pq = render_profile(slots)
                text_docs = [h.doc for h in full.search(query, k=k, mode="hybrid_rerank")]
                struct_docs = [h.doc for h in full.search_profile(slots, pq, n_attack=0)]
                seen, docs = set(), []
                for d in struct_docs + text_docs:  # structured cards first, then text results
                    if d["doc_id"] not in seen:
                        seen.add(d["doc_id"])
                        docs.append(d)
                ctx = {"docs": docs, "query": f"{query} | {pq}",
                       "bundle": seg["bundle"] if cfg == "A9b_union_noprof" else {**seg["bundle"], "observation_profile": slots}}
            elif cfg == "A6_random":
                ctx = {"docs": rng.sample(docs, k), "query": None}
            else:
                raise ValueError(cfg)
            out[(seg["seg_id"], cfg)] = ctx
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=ROOT / "dataset/eval/stage2_v4_segments.jsonl")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--models", nargs="+", default=["gemma3:12b", "deepseek-r1:14b"])
    parser.add_argument("--hosts", nargs="+", default=["http://127.0.0.1:11434"], help="one or more Ollama servers (parallel workers)")
    parser.add_argument("--workers-per-host", type=int, default=2)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage2_v4_ablation.jsonl")
    args = parser.parse_args()

    segments = [json.loads(l) for l in args.segments.read_text().splitlines() if l.strip()]
    segments = [s for s in segments if s["scope"] == args.scope]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seg_id"], r["config"], r["model"]))
    ctx = contexts(segments, args.configs, args.k)
    jobs = [(s, c, m) for m in args.models for c in args.configs for s in segments if (s["seg_id"], c, m) not in done]
    print(f"segments={len(segments)} jobs={len(jobs)} (done {len(done)})", flush=True)
    lock = threading.Lock()
    hosts = [h for h in args.hosts for _ in range(args.workers_per_host)]

    def work(i, seg, cfg, model):
        c = ctx[(seg["seg_id"], cfg)]
        j = judge(c.get("bundle", seg["bundle"]), c["docs"], model=model, all_cards=c.get("all_cards"),
                  show_docs=c.get("show_docs", True), host=hosts[i % len(hosts)])
        row = {"seg_id": seg["seg_id"], "config": cfg, "model": model, "split": seg["split"],
               "gt_card": seg["gt_card"], "gt_variant": seg["gt_variant"], "observability": seg["observability"],
               "query": c.get("query"), "doc_ids": [d["doc_id"] for d in c["docs"]],
               "gt_retrieved": any(d.get("card") in seg["accept"] for d in c["docs"]),
               "correct": is_correct(j, seg["gt_card"], seg["accept"]), **j.to_dict()}
        with lock, args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futs = [pool.submit(work, i, *job) for i, job in enumerate(jobs)]
        for n, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if n % 25 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    report(args.out)


def report(path: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    df["is_attack_gt"] = ~df["gt_card"].isin(["normal", "unobservable"])
    rows = []
    for (model, cfg), g in df.groupby(["model", "config"]):
        atk = g[g.is_attack_gt]
        nrm = g[g.gt_card == "normal"]
        rows.append({
            "model": model, "config": cfg, "n": len(g),
            "accuracy": round(g.correct.mean(), 3),
            "attack_detected": round((atk.verdict == "attack").mean(), 3),
            "attack_type_acc": round(atk.correct.mean(), 3),
            "normal_acc": round(nrm.correct.mean(), 3),
            "normal_fp": round((nrm.verdict == "attack").mean(), 3),
            "macro_acc": round(g.groupby("gt_card").correct.mean().mean(), 3),
            "gt_card_in_context": round(atk.gt_retrieved.mean(), 3),
            "invalid": round((g.verdict == "invalid").mean(), 3),
            "latency_s": round(g.latency_s.median(), 1),
        })
    table = pd.DataFrame(rows).sort_values(["model", "config"])
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()

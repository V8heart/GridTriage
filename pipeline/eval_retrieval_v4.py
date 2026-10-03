"""Retrieval-only evaluation over the v4 KB (no LLM).

For each segment query, find the rank of the first chunk of the ground-truth
card among all KB documents (cards + ATT&CK + ATLAS distractors).
Normal segments count as hits if any benign card is retrieved.
Unobservable segments have no target card and are skipped here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.kb.query import render_query
from pipeline.kb.retriever import MODES, HybridRetriever, load_kb

ROOT = Path(__file__).resolve().parent.parent
BENIGN = {"normal_workloads", "benign_periodic"}


def first_rank(hits, cards: set[str]) -> int | None:
    for hit in hits:
        if hit.doc.get("card") in cards:
            return hit.rank
    return None


def run(segments: list[dict], retriever: HybridRetriever, modes, *, depth: int = 50, query_kind: str = "v4") -> pd.DataFrame:
    rows = []
    for seg in segments:
        if seg["gt_card"] == "unobservable":
            continue
        target = BENIGN if seg["gt_card"] == "normal" else {seg["gt_card"]}
        if query_kind == "profile":
            from pipeline.kb.profile import profile_slots, render_profile

            slots = profile_slots(seg["bundle"])
            query = render_profile(slots)
            runs = {"profile_struct": retriever.search_profile(slots, query, n_cards=10, n_attack=0,
                                                              sections=("관측 프로필",))}
        elif query_kind == "multi":
            query = [seg["legacy_query"], render_query(seg["bundle"])]
            runs = {m: retriever.search_multi(query, k=depth, rerank=m == "hybrid_rerank", pool=max(depth, 50))
                    for m in modes if m in ("hybrid", "hybrid_rerank")}
        else:
            query = render_query(seg["bundle"]) if query_kind == "v4" else seg["legacy_query"]
            runs = {m: retriever.search(query, k=depth, mode=m, pool=max(depth, 50)) for m in modes}
        for mode, hits in runs.items():
            rank = first_rank(hits, target)
            rows.append({"seg_id": seg["seg_id"], "scope": seg["scope"], "split": seg["split"],
                         "gt_card": seg["gt_card"], "mode": mode, "query": query_kind,
                         "rank": rank if rank is not None else np.inf,
                         "top1": hits[0].doc_id if hits else None,
                         "top1_score": hits[0].score if hits else None})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        r = g["rank"]
        return pd.Series({"n": len(g), "R@1": (r <= 1).mean(), "R@3": (r <= 3).mean(),
                          "R@5": (r <= 5).mean(), "R@10": (r <= 10).mean(),
                          "MRR": (1.0 / r).mean()})
    return frame.groupby(["query", "mode"]).apply(agg, include_groups=False).round(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=ROOT / "dataset/eval/stage2_v4_segments.jsonl")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--cards-only", action="store_true", help="index only the 10 cards (no distractors)")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/retrieval_v4.json")
    args = parser.parse_args()
    segments = [json.loads(l) for l in args.segments.read_text().splitlines() if l.strip()]
    segments = [s for s in segments if s["scope"] == args.scope]
    docs = load_kb()
    if args.cards_only:
        docs = [d for d in docs if d["layer"] == "card"]
    retriever = HybridRetriever(docs)
    frame = pd.concat([run(segments, retriever, args.modes, query_kind=q) for q in ("legacy", "v4", "profile")])
    table = summarize(frame)
    print(f"KB docs: {len(docs)}  segments: {frame.seg_id.nunique()}")
    print(table.to_string())
    per_card = frame[frame["mode"].isin(["hybrid_rerank", "profile_struct"])].groupby(["query", "gt_card"])["rank"].apply(lambda r: (r <= 3).mean()).round(3)
    print("\nR@3 by card (hybrid_rerank / profile_struct):\n" + per_card.to_string())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kb_docs": len(docs), "scope": args.scope,
                                    "summary": table.reset_index().to_dict("records"),
                                    "rows": frame.replace({np.inf: None}).to_dict("records")},
                                   ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

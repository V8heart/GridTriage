"""Remove the Bit2Watt cards (SWMA, LTMA) and check that research recovers them.

Uses the official Stage2 path (v4 query + hybrid_rerank + LLM judge), unlike eval_loo.py (profile retrieval).
  full    judge SWMA/LTMA segments with the complete KB (reference)
  before  same, with both cards removed: what does the judge say, does 'unknown' fire?
          (the judge's requirement gate turns a forced nearest-card 'attack' into 'unknown')
  draft   research once per class from the most common profile of its *triggered* segments
          (arXiv / Semantic Scholar); falls back to all segments only if none triggered
  after   judge again with the drafts indexed: retrieved? chosen? do the drafts cite Bit2Watt?
Drafts go to dataset/eval/research_bit2watt/drafts (never into corpus/drafts).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from pipeline.kb.build_kb import build
from pipeline.kb.profile import profile_slots
from pipeline.kb.retriever import HybridRetriever
from pipeline.research.agent import research
from pipeline.v4_core import analyze_segment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "research_bit2watt"
BIT2WATT = ("swma", "ltma")
BIT2WATT_ARXIV = "2607.05993"


def run(segs, docs, model, host):
    retriever = HybridRetriever(docs)
    rows = []
    for seg in segs:
        out = analyze_segment(seg["bundle"], retriever, model=model, host=host, gate=True)
        j = out["judgment"]
        rows.append({"seg_id": seg["seg_id"], "gt_card": seg["gt_card"], "verdict": j["verdict"],
                     "threat_card": j.get("threat_card"), "gate": j.get("gate"), "retrieved_cards": [h["card"] for h in out["hits"]],
                     "explanation": (j.get("explanation") or "")[:300]})
    return rows


def summary(rows, drafts=()):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gt_card"]].append(r)
    out = {}
    for cls, rs in by.items():
        out[cls] = {
            "n": len(rs),
            "verdicts": dict(collections.Counter(f"{r['verdict']}:{r['threat_card']}" for r in rs)),
            "attack": sum(r["verdict"] == "attack" for r in rs),
            "correct_card": sum(r["threat_card"] == cls for r in rs),
            "unknown": sum(r["verdict"] == "unknown" for r in rs),
            "unknown_by_gate": sum(bool(r["gate"]) for r in rs),
            "draft_retrieved": sum(any(c in drafts for c in r["retrieved_cards"]) for r in rs),
            "draft_chosen": sum(r["threat_card"] in drafts for r in rs),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=ROOT / "dataset/eval/stage2_v4_segments.jsonl")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    segs = [s for s in map(json.loads, args.segments.read_text().splitlines())
            if s["scope"] == "all" and s["gt_card"] in BIT2WATT]
    report = {"n_segments": len(segs)}

    report["full"] = summary(run(segs, build(), args.model, args.host))
    print("full", json.dumps(report["full"], ensure_ascii=False), flush=True)

    docs = build(exclude_cards=BIT2WATT)
    before = run(segs, docs, args.model, args.host)
    report["before"] = summary(before)
    print("before", json.dumps(report["before"], ensure_ascii=False), flush=True)

    known = {a for d in docs for a in d.get("attack_ids", [])}
    drafts_dir = OUT / "drafts"
    report["research"] = {}
    unknown_ids = {r["seg_id"] for r in before if r["verdict"] == "unknown"}
    for cls in BIT2WATT:
        pool = [s for s in segs if s["gt_card"] == cls and s["seg_id"] in unknown_ids]
        forced = not pool
        pool = pool or [s for s in segs if s["gt_card"] == cls]
        slots = collections.Counter(json.dumps(profile_slots(s["bundle"]), sort_keys=True)
                                    for s in pool).most_common(1)[0][0]
        same = [s for s in pool if json.dumps(profile_slots(s["bundle"]), sort_keys=True) == slots]
        observed = sorted(set.intersection(*[{k for k, v in s["bundle"]["unexplainedness"]["evidence"].items() if v}
                                             for s in same]))
        res = research(json.loads(slots), model=args.model, host=args.host, known_ids=known, drafts=drafts_dir,
                       observed=observed)
        titles = [p["title"] for p in res["papers"]]
        report["research"][cls] = {
            "triggered_segments": 0 if forced else len(pool), "forced": forced, "observed_evidence": observed,
            "slots": json.loads(slots), "queries": res["queries"], "n_papers": res["n_papers"],
            "paper_titles": titles[:10],
            "bit2watt_fetched": any(BIT2WATT_ARXIV in p["url"] or "bit2watt" in p["title"].lower() for p in res["papers"]),
            "draft": Path(res["draft_path"]).stem if res["draft_path"] else None,
            "draft_sources": res["card"].get("sources"),
            "draft_cites_bit2watt": any(BIT2WATT_ARXIV in u for u in res["card"].get("sources") or []),
            "card_name": res["card"].get("name"),
            "draft_evidence": res["card"].get("evidence"),
        }
        print("research", cls, json.dumps(report["research"][cls], ensure_ascii=False), flush=True)

    drafts = {r["draft"] for r in report["research"].values() if r["draft"]}
    if drafts:
        after = run(segs, build(exclude_cards=BIT2WATT, drafts_dir=drafts_dir), args.model, args.host)
        report["after"] = summary(after, drafts)
        print("after", json.dumps(report["after"], ensure_ascii=False), flush=True)
        report["after_rows"] = after
    report["before_rows"] = before
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

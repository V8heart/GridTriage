"""Leave-one-card-out check of the research loop (plan_rag_pipeline_v4.md §9.3).

For each threat class that has real sessions but is removed from the KB:
  before  judge its segments with the card absent (A8 profile retrieval)
  trigger does the research agent fire (unknown verdict or no matching card)?
  draft   research once from the class's most common observation profile
  after   judge again with the draft indexed: is the draft retrieved and used?
Drafts go to dataset/eval/research_loo/<class>/ (never into corpus/drafts).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from pipeline.kb.build_kb import build
from pipeline.kb.profile import match_score, profile_slots, render_profile
from pipeline.kb.retriever import HybridRetriever
from pipeline.research.agent import research, should_research
from pipeline.stage2_rag_judge import judge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "research_loo"


def judge_segments(segs, retriever, model, host):
    rows = []
    for seg in segs:
        slots = profile_slots(seg["bundle"])
        hits = retriever.search_profile(slots, render_profile(slots))
        cards = [h.doc for h in hits if h.doc.get("profile")]
        j = judge({**seg["bundle"], "observation_profile": slots}, [h.doc for h in hits], model=model, host=host)
        scores = [match_score(slots, d["profile"]) for d in cards]
        rows.append({"seg_id": seg["seg_id"], "slots": slots, "verdict": j.verdict, "threat_card": j.threat_card,
                     "retrieved_cards": [d["card"] for d in cards], "best_profile_score": max(scores, default=None),
                     "trigger": should_research(j.to_dict(), scores), "explanation": j.explanation[:300]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=ROOT / "dataset/eval/stage2_v4_pristine_segments.jsonl")
    parser.add_argument("--classes", nargs="+", default=["hidden_ml_training", "sponge_attack", "llmjacking", "burst_ramp_attack"])
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    segs_all = [json.loads(l) for l in args.segments.read_text().splitlines() if l.strip()]
    report = {}
    for cls in args.classes:
        segs = [s for s in segs_all if s["scope"] == "all" and s["gt_card"] == cls]
        docs = build(exclude_cards=(cls,), with_profiles=True)
        known = {a for d in docs for a in d.get("attack_ids", [])}
        before = judge_segments(segs, HybridRetriever(docs), args.model, args.host)
        common = collections.Counter(json.dumps(r["slots"], sort_keys=True) for r in before).most_common(1)[0][0]
        drafts = OUT / cls
        out = research(json.loads(common), model=args.model, host=args.host, known_ids=known, drafts=drafts)
        after = []
        if out["draft_path"]:
            after = judge_segments(segs, HybridRetriever(build(exclude_cards=(cls,), drafts_dir=drafts, with_profiles=True)), args.model, args.host)
        draft = Path(out["draft_path"]).stem if out["draft_path"] else None
        report[cls] = {
            "n_segments": len(segs),
            "before_verdicts": collections.Counter(f"{r['verdict']}:{r['threat_card']}" for r in before),
            "triggered": sum(r["trigger"] for r in before),
            "research": {k: v for k, v in out.items() if k != "papers"},
            "paper_titles": [p["title"] for p in out["papers"]][:8],
            "after_verdicts": collections.Counter(f"{r['verdict']}:{r['threat_card']}" for r in after),
            "draft_retrieved": sum(draft in r["retrieved_cards"] for r in after) if draft else 0,
            "draft_used_in_verdict": sum(r["threat_card"] == draft for r in after) if draft else 0,
        }
        print(cls, json.dumps({k: report[cls][k] for k in ("before_verdicts", "triggered", "after_verdicts",
                                                        "draft_retrieved", "draft_used_in_verdict")}, default=str),
              "| draft:", draft, "| sources:", report[cls]["research"]["card"].get("sources"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

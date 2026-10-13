"""P3: does unknown-time research name the right mechanism? v4 research loop vs v5.

For each held-out class of P2 (its card and cases removed), up to two of its segments are researched
by both loops (forced, so retrieval quality is compared independently of the trigger). The v5 inputs
(case fit, channels) are the ones P2 computed with the class removed from memory.

A result is a hit when the regex of the true mechanism (fixed below before any run) matches:
  hyp_hit   top hypothesis (v5: name+mechanism; v4: drafted card name+summary+mechanism)
  lit_hit   a supporting paper of the top hypothesis (v5: graded 'correct'; v4: the draft's cited sources)
  any_hit   any of the v5 hypotheses / v4 papers fetched

  python -m pipeline.v5.eval_research_v5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from pipeline.kb.profile import profile_slots
from pipeline.kb.retriever import load_kb
from pipeline.research.agent import research as research_v4
from pipeline.v5.research_v5 import research_v5, write_queue_item

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "v5"

TRUTH = {
    "swma": r"bit2watt|load[- ]altering|power (?:modulation|oscillation)|grid (?:stability|oscillation|frequency)|workload modulation|power grid",
    "ltma": r"bit2watt|load[- ]altering|power (?:modulation|oscillation)|grid (?:stability|oscillation|frequency)|workload modulation|power grid",
    "burst_ramp_attack": r"load[- ]altering|power (?:virus|attack|spike)|grid (?:stability|frequency)|power grid|bit2watt|demand",
    "cryptojacking": r"crypto|mining|miner",
    "hidden_ml_training": r"(?:unauthori[sz]ed|hidden|covert|illicit|undeclared|rogue|stealth).{0,60}(?:training|fine[- ]?tun)|compute governance|training detection",
    "llmjacking": r"llm ?jack|(?:api|credential|key|endpoint|account).{0,40}(?:abuse|theft|hijack|stolen|misuse)|model (?:hijack|theft)|(?:inference|llm) (?:service|endpoint) (?:abuse|hijack|misuse)",
    "sponge_attack": r"sponge|energy[- ]latency|resource[- ]exhaustion|(?:denial[- ]of[- ]service|dos).{0,60}(?:model|llm|inference|neural)|energy (?:attack|consumption attack)",
}


def hit(cls: str, text: str) -> bool:
    return bool(re.search(TRUTH[cls], text or "", flags=re.I))


def v4_texts(out: dict) -> tuple[str, str, str]:
    card = out.get("card") or {}
    hyp = " ".join(str(card.get(k, "")) for k in ("name", "summary", "mechanism"))
    titles = {p["url"]: p["title"] + " " + p.get("abstract", "")[:300] for p in out.get("papers", [])}
    lit = " ".join(titles.get(u, "") for u in card.get("sources", []))
    anyp = " ".join(titles.values())
    return hyp, lit, anyp


def v5_texts(out: dict) -> tuple[str, str, str]:
    hyps = out.get("hypotheses") or []
    if not hyps:
        return "", "", ""
    top = hyps[0]
    hyp = top["name"].replace("_", " ") + " " + top["mechanism"]
    lit = " ".join(p["title"] + " " + p["abstract"][:300] for p in top["papers"] if p["grade"] == "correct")
    anyp = " ".join(h["name"].replace("_", " ") + " " + h["mechanism"] + " " +
                    " ".join(p["title"] for p in h["papers"] if p["grade"] != "incorrect") for h in hyps)
    return hyp, lit, anyp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    p2 = [json.loads(l) for l in (OUT / "eval_v5_p2.jsonl").read_text().splitlines() if l.strip()]
    segs = {json.loads(l)["seg_id"]: json.loads(l) for f in ("stage2_v4_segments.jsonl", "stage2_v4_pristine_segments.jsonl")
            for l in (ROOT / "dataset/eval" / f).read_text().splitlines()}
    out_path = OUT / "eval_research_v5.jsonl"
    done = {(json.loads(l)["seg_id"], json.loads(l)["system"]) for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    known_ids = {a for d in load_kb() for a in d.get("attack_ids", [])}
    for cls in TRUTH:
        rows = [r for r in p2 if r["system"] == "v5" and r["held_out"] == cls][: args.per_class]
        for r in rows:
            seg = segs[r["seg_id"]]
            for system in ("v4", "v5"):
                if (r["seg_id"], system) in done:
                    continue
                if system == "v4":
                    out = research_v4(profile_slots(seg["bundle"]), model=args.model, host=args.host, known_ids=known_ids,
                                      drafts=OUT / "research_v4_cmp" / cls,
                                      observed=[k for k, v in seg["bundle"]["unexplainedness"]["evidence"].items() if v])
                    texts = v4_texts(out)
                    summary = {"queries": out["queries"], "draft": (out.get("card") or {}).get("name"),
                               "papers": [p["title"] for p in out.get("papers", [])][:10]}
                else:
                    out = research_v5(seg["bundle"], r["channels"], r["case"], model=args.model, host=args.host)
                    write_queue_item(r["seg_id"], out, OUT / "research_queue" / cls)
                    texts = v5_texts(out)
                    summary = {"tokens": out["observed_tokens"],
                               "hypotheses": [(h["name"], h["score"], h["support"], h["queries"],
                                               [p["title"] for p in h["papers"] if p["grade"] == "correct"])
                                              for h in out["hypotheses"]]}
                row = {"seg_id": r["seg_id"], "held_out": cls, "system": system,
                       "hyp_hit": hit(cls, texts[0]), "lit_hit": hit(cls, texts[1]), "any_hit": hit(cls, texts[2]),
                       "summary": summary}
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(cls, system, {k: row[k] for k in ("hyp_hit", "lit_hit", "any_hit")}, flush=True)
    df = pd.DataFrame([json.loads(l) for l in out_path.read_text().splitlines()])
    table = df.groupby(["held_out", "system"])[["hyp_hit", "lit_hit", "any_hit"]].mean().round(2).unstack("system")
    total = df.groupby("system")[["hyp_hit", "lit_hit", "any_hit"]].mean().round(3)
    text = "# P3 연구 루프 비교 (정답 메커니즘을 찾았는가)\n\n```\n" + table.to_string() + "\n\n" + total.to_string() + "\n```\n"
    (OUT / "eval_research_v5.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

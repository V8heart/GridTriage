"""Handbook reachability table. Not a judge. Uses checklist_satisfied only."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.corpus_schema import checklist_satisfied, parse_corpus_v2
from pipeline.rag_analyzer import CORPUS_DIR


def _variant(row: dict) -> str:
    for key in ("gt_variant", "ground_truth", "gt_label"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "nan", "None"}:
            return str(value)
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-source", default="")
    args = parser.parse_args()
    rows = json.loads(args.stage2_json.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("results") or []
    docs = [parse_corpus_v2(path) for path in sorted(CORPUS_DIR.glob("*.md"))]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_by_variant: dict[str, int] = defaultdict(int)
    for row in rows:
        variant = _variant(row)
        n_by_variant[variant] += 1
        evidence = ((row.get("stage2_evidence_bundle") or {}).get("unexplainedness") or {}).get("evidence") or {}
        if not evidence and isinstance(row.get("evidence_bool"), dict):
            evidence = row["evidence_bool"]
        for doc in docs:
            ok, _, _ = checklist_satisfied(doc, evidence)
            if ok:
                counts[variant][doc.name] += 1
    lines = [
        "# Stage2 handbook reachability (stub checklist, not a verdict)",
        "",
        f"source={args.stage2_json} progress_source={args.progress_source or 'n/a'} segments={len(rows)}",
        "",
        "| variant | n | " + " | ".join(doc.name for doc in docs) + " |",
        "|---|" + "---:|" * (1 + len(docs)),
    ]
    for variant in sorted(n_by_variant):
        n = n_by_variant[variant]
        cells = [str(counts[variant][doc.name]) for doc in docs]
        lines.append(f"| {variant} | {n} | " + " | ".join(cells) + " |")
    payload = {
        "progress_source": args.progress_source,
        "n_segments": len(rows),
        "n_by_variant": dict(n_by_variant),
        "required_met": {k: dict(v) for k, v in counts.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

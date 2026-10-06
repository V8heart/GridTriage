"""Build the Stage2 retrieval knowledge base as JSONL.

Layers:
  L1 card    corpus/*.md threat cards, one chunk per ``##`` section (the judgment units)
  L2 attack  MITRE ATT&CK Enterprise + ICS techniques (STIX), MITRE ATLAS techniques
             and case studies. Background knowledge and, for retrieval, distractors.

Sources are pinned by the version recorded in each dump (ATT&CK 19.2, ATLAS 5.6.0).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "kb" / "raw"
CITATION = re.compile(r"\(Citation:[^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MAX_CHARS = 1500


def _clean(text: str) -> str:
    text = CITATION.sub("", text or "")
    text = LINK.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def _draft_card(path: Path):
    """Research drafts lack the approved evidence checklist, so parse them without the schema gate."""
    from types import SimpleNamespace

    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---\n", 2)
    meta = yaml.safe_load(fm) or {}
    return SimpleNamespace(name=path.stem, category=meta.get("category", "cyber_physical_attack"), body=body, meta=meta)


def card_rules(meta: dict) -> dict:
    """Required / exclusion evidence names, carried on every card doc for the judge's requirement gate."""
    evidence = [e for e in meta.get("evidence") or [] if isinstance(e, dict) and e.get("name")]
    return {"requires": sorted(e["name"] for e in evidence if e.get("necessity") == "required"),
            "excludes": sorted(e["name"] for e in evidence if e.get("necessity") == "exclusion")}


def card_docs(corpus_dir: Path, *, exclude: tuple[str, ...] = (), drafts_dir: Path | None = None,
              with_profiles: bool = False) -> list[dict]:
    from pipeline.corpus_schema import parse_corpus_v2

    docs = []
    paths = [(p, False) for p in sorted(corpus_dir.glob("*.md")) if p.stem not in exclude]
    if drafts_dir is not None and drafts_dir.exists():
        paths += [(p, True) for p in sorted(drafts_dir.glob("*.md"))]
    for path, is_draft in paths:
        card = _draft_card(path) if is_draft else parse_corpus_v2(path)
        sections = re.split(r"^## ", card.body, flags=re.M)
        title = sections[0].strip().splitlines()[0].lstrip("# ").strip() if sections[0].strip() else card.name
        profile = card.meta.get("profile")
        rules = card_rules(card.meta)
        # Slot profiles were not adopted for judgment (RESULTS_v4.md §5.2); index them only on request.
        if profile and with_profiles:
            from pipeline.kb.profile import render_card_profile

            docs.append({
                "doc_id": f"card:{card.name}#관측 프로필",
                "layer": "card",
                "source": f"corpus/{path.name}",
                "card": card.name,
                "category": card.category,
                "title": f"{title} — 관측 프로필",
                "text": render_card_profile(profile),
                "attack_ids": list(card.meta.get("attack_ids") or []),
                "profile": profile,
            })
        for section in sections[1:]:
            heading, _, text = section.partition("\n")
            text = _clean(text)
            if not text:
                continue
            docs.append({
                "doc_id": f"card:{card.name}#{heading.strip()}",
                "layer": "card",
                "source": f"corpus/{path.name}",
                "card": card.name,
                "category": card.category,
                "title": f"{title} — {heading.strip()}",
                "text": text,
                "attack_ids": list(card.meta.get("attack_ids") or []),
                **rules,
            })
    return docs


def _stix_techniques(path: Path, source: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for obj in data["objects"]:
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ext = [r["external_id"] for r in obj.get("external_references", []) if r.get("source_name") in {"mitre-attack", "mitre-ics-attack"}]
        if not ext:
            continue
        tactics = sorted({p["phase_name"] for p in obj.get("kill_chain_phases", [])})
        docs.append({
            "doc_id": f"{source}:{ext[0]}",
            "layer": "attack",
            "source": source,
            "card": None,
            "category": "technique",
            "title": f"{ext[0]} {obj['name']}",
            "text": _clean(f"{ext[0]} {obj['name']}. Tactics: {', '.join(tactics)}. {obj.get('description', '')}"),
            "attack_ids": [ext[0]],
        })
    return docs


def atlas_docs(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    docs = []
    for tech in data["matrices"][0].get("techniques", []):
        docs.append({
            "doc_id": f"atlas:{tech['id']}",
            "layer": "attack",
            "source": "atlas",
            "card": None,
            "category": "technique",
            "title": f"{tech['id']} {tech['name']}",
            "text": _clean(f"{tech['id']} {tech['name']}. {tech.get('description', '')}"),
            "attack_ids": [tech["id"]],
        })
    for case in data.get("case-studies", []):
        used = sorted({step.get("technique") for step in case.get("procedure", []) if step.get("technique")})
        docs.append({
            "doc_id": f"atlas:{case['id']}",
            "layer": "attack",
            "source": "atlas-case-study",
            "card": None,
            "category": "case_study",
            "title": f"{case['id']} {case['name']}",
            "text": _clean(f"{case['name']}. {case.get('summary', '')}"),
            "attack_ids": used,
        })
    return docs


def build(corpus_dir: Path = ROOT / "corpus", raw: Path = RAW, *, cards_only: bool = False,
          exclude_cards: tuple[str, ...] = (), drafts_dir: Path | None = None,
          with_profiles: bool = False) -> list[dict]:
    docs = card_docs(corpus_dir, exclude=exclude_cards, drafts_dir=drafts_dir, with_profiles=with_profiles)
    if not cards_only:
        docs += _stix_techniques(raw / "enterprise-attack.json", "attack-enterprise")
        docs += _stix_techniques(raw / "ics-attack.json", "attack-ics")
        docs += atlas_docs(raw / "ATLAS.yaml")
    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_id in KB")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "kb" / "kb_docs.jsonl")
    parser.add_argument("--include-drafts", action="store_true", help="index corpus/drafts (unapproved research cards)")
    args = parser.parse_args()
    docs = build(drafts_dir=ROOT / "corpus" / "drafts" if args.include_drafts else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    by_source: dict[str, int] = {}
    for doc in docs:
        by_source[doc["source"].split("/")[0]] = by_source.get(doc["source"].split("/")[0], 0) + 1
    print(json.dumps({"docs": len(docs), "by_source": by_source, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

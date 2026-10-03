"""Curated sources only (plan_rag_pipeline_v4.md §9.3): arXiv, Semantic Scholar, MITRE ATT&CK/ATLAS.

No general web search engine. Fetched text is untrusted data: it is only passed to the
extractor as quoted material and never executed or followed as instructions.
Outgoing queries carry generic technical keywords only (no session ids or telemetry values).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

UA = "GridPulse-research/0.1 (curated threat research; contact: local)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    source: str
    title: str
    abstract: str
    url: str
    year: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _get(url: str, *, timeout: int = 30, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _arxiv_term(t: str) -> str:
    t = t.strip()
    return f'all:"{t}"' if " " in t else f"all:{t}"


def arxiv_search(query: str, max_results: int = 5) -> list[Paper]:
    """``query`` = 'required;alt1;alt2' -> required AND (alt1 OR alt2)."""
    parts = [t for t in query.split(";") if t.strip()]
    terms = _arxiv_term(parts[0]) if parts else query
    if len(parts) > 1:
        terms += " AND (" + " OR ".join(_arxiv_term(t) for t in parts[1:]) + ")"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terms, "max_results": max_results, "sortBy": "relevance"})
    root = ET.fromstring(_get(url))
    out = []
    for entry in root.findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", "", ATOM) or "").split())
        abstract = " ".join((entry.findtext("a:summary", "", ATOM) or "").split())
        link = entry.findtext("a:id", "", ATOM)
        year = (entry.findtext("a:published", "", ATOM) or "")[:4]
        out.append(Paper("arxiv", title, abstract, link, int(year) if year.isdigit() else None))
    time.sleep(3)  # arXiv API etiquette: one request every 3 s
    return out


def semantic_scholar_search(query: str, limit: int = 5, *, retries: int = 2) -> list[Paper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": query.replace(";", " "), "limit": limit, "fields": "title,abstract,url,year"})
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    for attempt in range(retries + 1):
        try:
            data = json.loads(_get(url, headers=headers))
            return [Paper("semantic_scholar", p.get("title") or "", p.get("abstract") or "", p.get("url") or "", p.get("year"))
                    for p in data.get("data", []) if p.get("abstract")]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries:
                return []
            time.sleep(5 * (attempt + 1))
    return []


def attack_versions() -> dict:
    """Latest published ATT&CK domain versions (to compare with the local KB dump)."""
    index = json.loads(_get("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/index.json"))
    return {c["name"]: c["versions"][0]["version"] for c in index.get("collections", [])}

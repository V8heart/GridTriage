"""Hybrid retriever over the Stage2 KB: BM25 + bge-m3 dense, RRF, cross-encoder rerank.

Modes (for ablation):
  tfidf          char 2-4gram TF-IDF (the pre-v4 retriever) over the whole KB
  bm25           lexical only
  dense          bge-m3 only
  hybrid         BM25 + dense fused by reciprocal rank fusion
  hybrid_rerank  hybrid top-N re-scored by bge-reranker-v2-m3 (default)

Scores rank documents for the LLM; they are not attack probabilities.
Models run on a dedicated pipeline GPU (assumed separate from the monitored
GPUs): GRIDPULSE_RETRIEVER_DEVICE, else the last CUDA device, else CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "dataset" / "kb" / "kb_docs.jsonl"
INDEX_DIR = ROOT / "dataset" / "kb" / "index"
DENSE_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MODES = ("tfidf", "bm25", "dense", "hybrid", "hybrid_rerank")
RRF_K = 60
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[가-힣]+")


def default_device() -> str:
    env = os.environ.get("GRIDPULSE_RETRIEVER_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.device_count() - 1}"
    except ImportError:
        pass
    return "cpu"


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; Hangul words also yield char bigrams (particles vary)."""
    out = []
    for tok in TOKEN.findall(text.lower()):
        out.append(tok)
        if "가" <= tok[0] <= "힣" and len(tok) > 2:
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return out


@dataclass
class Hit:
    doc: dict
    score: float
    rank: int
    stage_scores: dict

    @property
    def doc_id(self) -> str:
        return self.doc["doc_id"]


def load_kb(path: Path = KB_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _doc_text(doc: dict) -> str:
    return f"{doc['title']}. {doc['text']}"


class HybridRetriever:
    def __init__(self, docs: list[dict] | None = None, *, device: str | None = None, load_models: bool = True):
        self.docs = docs if docs is not None else load_kb()
        self.device = device or default_device()
        texts = [_doc_text(d) for d in self.docs]
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        self._tfidf = None
        self._dense = None
        self._emb = None
        self._reranker = None
        if load_models:
            self._load_dense(texts)

    # -- lazy components -------------------------------------------------
    def _load_dense(self, texts: list[str]) -> None:
        from sentence_transformers import SentenceTransformer

        self._dense = SentenceTransformer(DENSE_MODEL, device=self.device)
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        cache = INDEX_DIR / f"dense_{digest}.npy"
        if cache.exists():
            self._emb = np.load(cache)
        else:
            self._emb = self._dense.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, self._emb)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if self._tfidf is None:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            mat = normalize(vec.fit_transform([_doc_text(d) for d in self.docs]))
            self._tfidf = (vec, mat)
        vec, mat = self._tfidf
        return (mat @ normalize(vec.transform([query])).T).toarray().ravel()

    def _rerank(self, query: str, idx: list[int]) -> np.ndarray:
        import torch
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=512)
        pairs = [(query, _doc_text(self.docs[i])) for i in idx]
        return np.asarray(self._reranker.predict(pairs, batch_size=16, activation_fn=torch.nn.Sigmoid()), dtype=float)

    # -- search ----------------------------------------------------------
    def _diverse(self, order, k: int, max_per_card: int | None) -> list[int]:
        """Top-k with at most ``max_per_card`` chunks of the same threat card."""
        if not max_per_card:
            return [int(i) for i in order[:k]]
        out, per = [], {}
        for i in order:
            card = self.docs[int(i)].get("card")
            if card is not None:
                if per.get(card, 0) >= max_per_card:
                    continue
                per[card] = per.get(card, 0) + 1
            out.append(int(i))
            if len(out) == k:
                break
        return out

    def search(self, query: str, k: int = 8, *, mode: str = "hybrid_rerank", pool: int = 50,
               max_per_card: int | None = 2) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        n = len(self.docs)
        stage: dict[int, dict] = {}
        if mode == "tfidf":
            scores = self._tfidf_scores(query)
            top = self._diverse(np.argsort(-scores), k, max_per_card)
            return [Hit(self.docs[i], float(scores[i]), r + 1, {"tfidf": float(scores[i])}) for r, i in enumerate(top)]
        bm25 = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        dense = None
        if mode != "bm25":
            q = self._dense.encode([query], normalize_embeddings=True)[0]
            dense = self._emb @ q
        if mode == "bm25":
            fused, order = bm25, np.argsort(-bm25)
        elif mode == "dense":
            fused, order = dense, np.argsort(-dense)
        else:
            rank_b = np.empty(n, dtype=int)
            rank_b[np.argsort(-bm25)] = np.arange(1, n + 1)
            rank_d = np.empty(n, dtype=int)
            rank_d[np.argsort(-dense)] = np.arange(1, n + 1)
            fused = 1.0 / (RRF_K + rank_b) + 1.0 / (RRF_K + rank_d)
            order = np.argsort(-fused)
        for i in order[:pool]:
            stage[int(i)] = {"bm25": float(bm25[i]), "fused": float(fused[i])}
            if dense is not None:
                stage[int(i)]["dense"] = float(dense[i])
        if mode != "hybrid_rerank":
            top = self._diverse(order[:max(pool, k)], k, max_per_card)
            return [Hit(self.docs[i], float(fused[i]), r + 1, stage[i]) for r, i in enumerate(top)]
        cand = [int(i) for i in order[:pool]]
        rr = self._rerank(query, cand)
        for i, s in zip(cand, rr):
            stage[i]["rerank"] = float(s)
        top = self._diverse([cand[j] for j in np.argsort(-rr)], k, max_per_card)
        return [Hit(self.docs[i], stage[i]["rerank"], r + 1, stage[i]) for r, i in enumerate(top)]

    def _fused_ranking(self, query: str) -> np.ndarray:
        """BM25 + dense RRF score for every document."""
        n = len(self.docs)
        bm25 = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        dense = self._emb @ self._dense.encode([query], normalize_embeddings=True)[0]
        rank_b = np.empty(n, dtype=int)
        rank_b[np.argsort(-bm25)] = np.arange(1, n + 1)
        rank_d = np.empty(n, dtype=int)
        rank_d[np.argsort(-dense)] = np.arange(1, n + 1)
        return 1.0 / (RRF_K + rank_b) + 1.0 / (RRF_K + rank_d)

    def search_multi(self, queries: list[str], k: int = 8, *, rerank: bool = True, pool: int = 50,
                     max_per_card: int | None = 2) -> list[Hit]:
        """Multi-query retrieval: RRF across per-query hybrid rankings, then max-over-queries rerank.

        One query describes the generic power shape, the other the discriminative evidence;
        a document only needs to match one of them well.
        """
        queries = [q for q in queries if q]
        n = len(self.docs)
        fused = np.zeros(n)
        for q in queries:
            score = self._fused_ranking(q)
            rank = np.empty(n, dtype=int)
            rank[np.argsort(-score)] = np.arange(1, n + 1)
            fused += 1.0 / (RRF_K + rank)
        order = np.argsort(-fused)
        if not rerank:
            top = self._diverse(order[:max(pool, k)], k, max_per_card)
            return [Hit(self.docs[i], float(fused[i]), r + 1, {"fused": float(fused[i])}) for r, i in enumerate(top)]
        cand = [int(i) for i in order[:pool]]
        rr = np.max(np.vstack([self._rerank(q, cand) for q in queries]), axis=0)
        top = self._diverse([cand[j] for j in np.argsort(-rr)], k, max_per_card)
        scores = dict(zip(cand, rr))
        return [Hit(self.docs[i], float(scores[i]), r + 1, {"rerank": float(scores[i]), "fused": float(fused[i])})
                for r, i in enumerate(top)]

    def search_profile(self, slots: dict, query: str, *, n_cards: int = 3, n_attack: int = 2,
                       sections: tuple[str, ...] = ("관측 프로필", "관측 시그니처")) -> list[Hit]:
        """Slot-structured card ranking + text retrieval for ATT&CK/ATLAS background.

        Cards are ranked by the fraction of their profile slots the observation satisfies;
        ties are broken by the cross-encoder score of the query against the card profile.
        """
        from pipeline.kb.profile import match_score

        profiles = [(i, d) for i, d in enumerate(self.docs) if d.get("profile")]
        structured = {d["card"]: match_score(slots, d["profile"]) for _, d in profiles}
        tie = dict(zip([d["card"] for _, d in profiles], self._rerank(query, [i for i, _ in profiles])))
        ranked = sorted(structured, key=lambda c: (structured[c], tie[c]), reverse=True)[:n_cards]
        out: list[Hit] = []
        for card in ranked:
            for sec in sections:
                for i, d in enumerate(self.docs):
                    if d.get("card") == card and d["doc_id"].endswith("#" + sec):
                        out.append(Hit(d, structured[card], len(out) + 1,
                                       {"profile_match": structured[card], "tie_rerank": float(tie[card])}))
        if n_attack:
            fused = self._fused_ranking(query)
            background = [int(i) for i in np.argsort(-fused) if self.docs[int(i)]["layer"] != "card"][:n_attack]
            for i in background:
                out.append(Hit(self.docs[i], float(fused[i]), len(out) + 1, {"fused": float(fused[i])}))
        return out

    def card_rank(self, hits_or_query, card: str, *, mode: str = "hybrid_rerank", depth: int = 50) -> int | None:
        """1-based rank of the first chunk of ``card`` in the ranking, None if absent."""
        hits = hits_or_query if isinstance(hits_or_query, list) else self.search(hits_or_query, k=depth, mode=mode, pool=max(depth, 50))
        for hit in hits:
            if hit.doc.get("card") == card:
                return hit.rank
        return None

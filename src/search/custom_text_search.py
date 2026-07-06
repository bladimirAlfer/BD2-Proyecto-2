from __future__ import annotations

import pickle
import time
from collections import defaultdict

import pandas as pd

from src.config import PMC_ARTICLES_CSV, PMC_CHUNKS_CSV, TEXT_INDEX_PATH, TEXT_VECTORIZER_PATH


class CustomTextSearch:
    def __init__(self):
        if not TEXT_INDEX_PATH.exists():
            raise FileNotFoundError(TEXT_INDEX_PATH)
        if not TEXT_VECTORIZER_PATH.exists():
            raise FileNotFoundError(TEXT_VECTORIZER_PATH)
        self.chunks = pd.read_csv(PMC_CHUNKS_CSV).fillna("")
        self.chunk_by_id = self.chunks.set_index("chunk_id").to_dict(orient="index")
        self.article_by_id = {}
        if PMC_ARTICLES_CSV.exists():
            articles = pd.read_csv(PMC_ARTICLES_CSV).fillna("")
            self.article_by_id = articles.set_index("article_id").to_dict(orient="index")
        with open(TEXT_INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(TEXT_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def search(self, query: str, top_k: int = 10):
        t0 = time.perf_counter()
        q = self.vectorizer.transform([query])
        scores = defaultdict(float)

        row = q.getrow(0)
        for term_id, q_weight in zip(row.indices, row.data):
            postings = self.index.get(int(term_id), [])
            for chunk_id, doc_weight in postings:
                scores[chunk_id] += float(q_weight) * float(doc_weight)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        results = []
        for chunk_id, score in ranked:
            meta = self.chunk_by_id.get(chunk_id, {})
            article = self.article_by_id.get(meta.get("article_id", ""), {})
            results.append({
                "chunk_id": chunk_id,
                "article_id": meta.get("article_id", ""),
                "title": article.get("title", ""),
                "doi": article.get("doi", ""),
                "citation": article.get("citation", ""),
                "score": float(score),
                "text": meta.get("chunk_text", "")[:500],
            })
        return {"method": "custom_spimi", "latency_ms": elapsed, "results": results}

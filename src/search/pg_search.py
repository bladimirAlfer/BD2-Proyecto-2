from __future__ import annotations

import time
from pathlib import Path

from src.database.connection import fetch_all
from src.search.custom_image_search import CustomImageSearch
from src.search.custom_audio_search import CustomAudioSearch
from src.utils.vectors import vector_to_pg_literal


def pg_text_search(query: str, top_k: int = 10):
    t0 = time.perf_counter()
    rows = fetch_all("""
        SELECT c.chunk_id, c.article_id, d.title, d.doi, d.citation, c.chunk_text,
               ts_rank(c.search_vector, plainto_tsquery('english', %(q)s)) AS score
        FROM text_chunks c
        JOIN documents d ON d.article_id = c.article_id
        WHERE c.search_vector @@ plainto_tsquery('english', %(q)s)
        ORDER BY score DESC
        LIMIT %(top_k)s;
    """, {"q": query, "top_k": top_k})
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "method": "postgres_gin",
        "latency_ms": elapsed,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "article_id": r["article_id"],
                "title": r["title"],
                "doi": r["doi"],
                "citation": r["citation"],
                "score": float(r["score"]),
                "text": (r["chunk_text"] or "")[:500],
            }
            for r in rows
        ],
    }


def pg_image_search(image_path: str | Path, top_k: int = 10):
    engine = CustomImageSearch()
    qhist = engine.image_to_histogram(image_path)
    qvec = vector_to_pg_literal(qhist)
    t0 = time.perf_counter()
    rows = fetch_all("""
        SELECT ih.image_id, ih.article_id, i.image_path,
               1 - (ih.histogram <=> %(qvec)s::vector) AS score
        FROM image_histograms ih
        JOIN images i ON i.image_id = ih.image_id
        ORDER BY ih.histogram <=> %(qvec)s::vector
        LIMIT %(top_k)s;
    """, {"qvec": qvec, "top_k": top_k})
    elapsed = (time.perf_counter() - t0) * 1000
    return {"method": "postgres_pgvector_image", "latency_ms": elapsed, "results": [dict(r) for r in rows]}


def pg_audio_search(audio_path: str | Path, top_k: int = 10):
    engine = CustomAudioSearch()
    qhist = engine.audio_to_histogram(audio_path)
    qvec = vector_to_pg_literal(qhist)
    t0 = time.perf_counter()
    rows = fetch_all("""
        SELECT ah.track_id, s.title, s.artist_name, s.genre_top, s.audio_path,
               1 - (ah.histogram <=> %(qvec)s::vector) AS score
        FROM audio_histograms ah
        JOIN songs s ON s.track_id = ah.track_id
        ORDER BY ah.histogram <=> %(qvec)s::vector
        LIMIT %(top_k)s;
    """, {"qvec": qvec, "top_k": top_k})
    elapsed = (time.perf_counter() - t0) * 1000
    return {"method": "postgres_pgvector_audio", "latency_ms": elapsed, "results": [dict(r) for r in rows]}


def explain_json(sql: str, params=None):
    rows = fetch_all("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params or {})
    plan = rows[0]["QUERY PLAN"][0]
    return plan

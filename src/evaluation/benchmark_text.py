from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PMC_CHUNKS_CSV, TEXT_INDEX_PATH, TEXT_TFIDF_MATRIX_PATH, TEXT_VECTORIZER_PATH
from src.database.connection import bulk_insert
from src.evaluation.benchmark_utils import (
    explain_io,
    file_size_bytes,
    peak_memory_mb,
    pg_relation_sizes,
    throughput_qps,
)
from src.evaluation.metrics import ids_from_results, overlap_at_k
from src.search.custom_text_search import CustomTextSearch
from src.search.pg_search import pg_text_search

OUT = Path("results/tables/benchmark_text.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "machine learning",
    "deep learning",
    "neural network",
    "database indexing",
    "information retrieval",
]


def main(top_k: int = 10):
    custom = CustomTextSearch()
    dataset_size = len(custom.chunks)
    custom_index_size = file_size_bytes(TEXT_INDEX_PATH, TEXT_VECTORIZER_PATH, TEXT_TFIDF_MATRIX_PATH)
    custom_table_size = file_size_bytes(PMC_CHUNKS_CSV)
    pg_index_size, pg_table_size = pg_relation_sizes("text_chunks")
    rows = []
    for qid, q in enumerate(QUERIES):
        c = custom.search(q, top_k)
        g = pg_text_search(q, top_k)
        hit_blocks, read_blocks, execution_time_ms = explain_io(
            """
            SELECT c.chunk_id, c.article_id, c.chunk_text,
                   ts_rank(c.search_vector, plainto_tsquery('english', %(q)s)) AS score
            FROM text_chunks c
            WHERE c.search_vector @@ plainto_tsquery('english', %(q)s)
            ORDER BY score DESC
            LIMIT %(top_k)s;
            """,
            {"q": q, "top_k": top_k},
        )
        custom_ids = ids_from_results(c["results"], "chunk_id")
        gin_ids = ids_from_results(g["results"], "chunk_id")
        ov = overlap_at_k(custom_ids, gin_ids, top_k)

        rows.append({
            "modality": "text",
            "application": "document_search",
            "method": "custom_spimi",
            "query_id": str(qid),
            "query_text": q,
            "dataset_size": dataset_size,
            "top_k": top_k,
            "latency_ms": c["latency_ms"],
            "throughput_qps": throughput_qps(c["latency_ms"]),
            "recall_at_k": ov,
            "overlap_at_k": ov,
            "execution_time_ms": c["latency_ms"],
            "memory_peak_mb": peak_memory_mb(),
            "shared_hit_blocks": 0,
            "shared_read_blocks": 0,
            "index_size_bytes": custom_index_size,
            "table_size_bytes": custom_table_size,
        })
        rows.append({
            "modality": "text",
            "application": "document_search",
            "method": "postgres_gin",
            "query_id": str(qid),
            "query_text": q,
            "dataset_size": dataset_size,
            "top_k": top_k,
            "latency_ms": g["latency_ms"],
            "throughput_qps": throughput_qps(g["latency_ms"]),
            "recall_at_k": ov,
            "overlap_at_k": ov,
            "execution_time_ms": execution_time_ms,
            "memory_peak_mb": peak_memory_mb(),
            "shared_hit_blocks": hit_blocks,
            "shared_read_blocks": read_blocks,
            "index_size_bytes": pg_index_size,
            "table_size_bytes": pg_table_size,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    pg_rows = [(
        r["modality"], r["application"], r["method"], r["query_id"], r["query_text"], r["dataset_size"],
        r["top_k"], r["latency_ms"], r["throughput_qps"], r["recall_at_k"], r["overlap_at_k"],
        r["execution_time_ms"], r["shared_hit_blocks"], r["shared_read_blocks"],
        r["index_size_bytes"], r["table_size_bytes"]
    ) for r in rows]

    bulk_insert("""
        INSERT INTO benchmark_results(
            modality, application, method, query_id, query_text, dataset_size, top_k,
            latency_ms, throughput_qps, recall_at_k, overlap_at_k, execution_time_ms,
            shared_hit_blocks, shared_read_blocks, index_size_bytes, table_size_bytes
        ) VALUES %s
    """, pg_rows)

    print(df)
    print("Guardado:", OUT)


if __name__ == "__main__":
    main()

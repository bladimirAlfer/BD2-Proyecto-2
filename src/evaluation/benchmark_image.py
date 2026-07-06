from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PMC_IMAGES_CSV, PMC_VISUAL_HIST_CSV, PMC_VISUAL_INDEX_PATH, PMC_VISUAL_KMEANS_PATH
from src.database.connection import bulk_insert
from src.evaluation.benchmark_utils import (
    explain_io,
    file_size_bytes,
    peak_memory_mb,
    pg_relation_sizes,
    throughput_qps,
)
from src.evaluation.metrics import ids_from_results, overlap_at_k
from src.search.custom_image_search import CustomImageSearch
from src.search.pg_search import pg_image_search
from src.utils.vectors import vector_to_pg_literal

OUT = Path("results/tables/benchmark_image.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main(top_k: int = 10, n_queries: int = 10):
    images = pd.read_csv(PMC_IMAGES_CSV).dropna(subset=["image_path"])
    images = images[images["image_path"].apply(lambda p: Path(str(p)).exists())].head(n_queries)
    custom = CustomImageSearch()
    dataset_size = len(custom.hist)
    custom_index_size = file_size_bytes(PMC_VISUAL_INDEX_PATH, PMC_VISUAL_KMEANS_PATH, PMC_VISUAL_HIST_CSV)
    custom_table_size = file_size_bytes(PMC_IMAGES_CSV)
    pg_index_size, pg_table_size = pg_relation_sizes("image_histograms")
    rows = []

    for i, r in images.iterrows():
        image_path = r["image_path"]
        c = custom.search(image_path, top_k)
        p = pg_image_search(image_path, top_k)
        qvec = vector_to_pg_literal(custom.image_to_histogram(image_path))
        hit_blocks, read_blocks, execution_time_ms = explain_io(
            """
            SELECT ih.image_id, ih.article_id, i.image_path,
                   1 - (ih.histogram <=> %(qvec)s::vector) AS score
            FROM image_histograms ih
            JOIN images i ON i.image_id = ih.image_id
            ORDER BY ih.histogram <=> %(qvec)s::vector
            LIMIT %(top_k)s;
            """,
            {"qvec": qvec, "top_k": top_k},
        )
        custom_ids = ids_from_results(c["results"], "image_id")
        pg_ids = ids_from_results(p["results"], "image_id")
        ov = overlap_at_k(custom_ids, pg_ids, top_k)
        rows.append({
            "modality": "image", "application": "document_search", "method": "custom_visual_index",
            "query_id": str(r["image_id"]), "dataset_size": dataset_size, "top_k": top_k,
            "latency_ms": c["latency_ms"], "throughput_qps": throughput_qps(c["latency_ms"]),
            "recall_at_k": ov, "overlap_at_k": ov, "execution_time_ms": c["latency_ms"],
            "memory_peak_mb": peak_memory_mb(), "shared_hit_blocks": 0, "shared_read_blocks": 0,
            "index_size_bytes": custom_index_size, "table_size_bytes": custom_table_size,
        })
        rows.append({
            "modality": "image", "application": "document_search", "method": "postgres_pgvector",
            "query_id": str(r["image_id"]), "dataset_size": dataset_size, "top_k": top_k,
            "latency_ms": p["latency_ms"], "throughput_qps": throughput_qps(p["latency_ms"]),
            "recall_at_k": ov, "overlap_at_k": ov, "execution_time_ms": execution_time_ms,
            "memory_peak_mb": peak_memory_mb(), "shared_hit_blocks": hit_blocks, "shared_read_blocks": read_blocks,
            "index_size_bytes": pg_index_size, "table_size_bytes": pg_table_size,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pg_rows = [(
        x["modality"], x["application"], x["method"], x["query_id"], None, x["dataset_size"],
        x["top_k"], x["latency_ms"], x["throughput_qps"], x["recall_at_k"], x["overlap_at_k"],
        x["execution_time_ms"], x["shared_hit_blocks"], x["shared_read_blocks"],
        x["index_size_bytes"], x["table_size_bytes"]
    ) for x in rows]
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

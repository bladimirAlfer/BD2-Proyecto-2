from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FMA_AUDIO_HIST_CSV, FMA_AUDIO_INDEX_PATH, FMA_AUDIO_KMEANS_PATH, FMA_TRACKS_CSV
from src.database.connection import bulk_insert
from src.evaluation.benchmark_utils import (
    explain_io,
    file_size_bytes,
    peak_memory_mb,
    pg_relation_sizes,
    throughput_qps,
)
from src.evaluation.metrics import ids_from_results, overlap_at_k
from src.search.custom_audio_search import CustomAudioSearch
from src.search.pg_search import pg_audio_search
from src.utils.vectors import vector_to_pg_literal

OUT = Path("results/tables/benchmark_audio.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main(top_k: int = 10, n_queries: int = 10):
    tracks = pd.read_csv(FMA_TRACKS_CSV).dropna(subset=["audio_path"])
    tracks = tracks[tracks["audio_path"].apply(lambda p: Path(str(p)).exists())].head(n_queries)
    custom = CustomAudioSearch()
    dataset_size = len(custom.hist)
    custom_index_size = file_size_bytes(FMA_AUDIO_INDEX_PATH, FMA_AUDIO_KMEANS_PATH, FMA_AUDIO_HIST_CSV)
    custom_table_size = file_size_bytes(FMA_TRACKS_CSV)
    pg_index_size, pg_table_size = pg_relation_sizes("audio_histograms")
    rows = []

    for _, r in tracks.iterrows():
        audio_path = r["audio_path"]
        c = custom.search(audio_path, top_k)
        p = pg_audio_search(audio_path, top_k)
        qvec = vector_to_pg_literal(custom.audio_to_histogram(audio_path))
        hit_blocks, read_blocks, execution_time_ms = explain_io(
            """
            SELECT ah.track_id, s.title, s.artist_name, s.genre_top, s.audio_path,
                   1 - (ah.histogram <=> %(qvec)s::vector) AS score
            FROM audio_histograms ah
            JOIN songs s ON s.track_id = ah.track_id
            ORDER BY ah.histogram <=> %(qvec)s::vector
            LIMIT %(top_k)s;
            """,
            {"qvec": qvec, "top_k": top_k},
        )
        custom_ids = ids_from_results(c["results"], "track_id")
        pg_ids = ids_from_results(p["results"], "track_id")
        ov = overlap_at_k(custom_ids, pg_ids, top_k)
        rows.append({
            "modality": "audio", "application": "music_search", "method": "custom_audio_index",
            "query_id": str(r["track_id"]), "dataset_size": dataset_size, "top_k": top_k,
            "latency_ms": c["latency_ms"], "throughput_qps": throughput_qps(c["latency_ms"]),
            "recall_at_k": ov, "overlap_at_k": ov, "execution_time_ms": c["latency_ms"],
            "memory_peak_mb": peak_memory_mb(), "shared_hit_blocks": 0, "shared_read_blocks": 0,
            "index_size_bytes": custom_index_size, "table_size_bytes": custom_table_size,
        })
        rows.append({
            "modality": "audio", "application": "music_search", "method": "postgres_pgvector",
            "query_id": str(r["track_id"]), "dataset_size": dataset_size, "top_k": top_k,
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

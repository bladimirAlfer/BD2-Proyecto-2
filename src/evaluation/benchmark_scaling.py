"""Benchmark de escalamiento multimodal.

Para cada modalidad (texto, imagen, audio) y para varios tamaños de corpus N,
compara tres métodos sobre EL MISMO subconjunto de N elementos:

  1. sequential_cosine : KNN secuencial exacto por coseno (baseline / ground-truth).
  2. custom_inverted    : índice invertido propio (SPIMI / BoVW / BoAW).
  3. postgres           : GIN full-text (texto) o pgvector HNSW (imagen/audio).

Métricas por consulta: latencia, throughput, recall@k (contra el secuencial
exacto), overlap@k (custom vs postgres), memoria del índice, bloques I/O y
tamaños de índice/tabla. Los resultados se guardan por modalidad en
results/tables/benchmark_<modality>.csv y se insertan en benchmark_results.

Diseño: la consulta usa el histograma / vector TF-IDF YA calculado, de modo que
se mide el costo del ÍNDICE y no el de decodificar audio/imagen.
"""
from __future__ import annotations

# Medición de latencia en un solo hilo: para operaciones pequeñas el despacho de
# hilos de BLAS (OpenBLAS/MKL) domina y distorsiona los tiempos. Debe fijarse
# ANTES de importar numpy.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle

from src.config import (
    PMC_CHUNKS_CSV,
    PMC_VISUAL_HIST_CSV,
    FMA_AUDIO_HIST_CSV,
    TEXT_TFIDF_MATRIX_PATH,
    TEXT_VECTORIZER_PATH,
    TOP_K,
)
from src.database.connection import bulk_insert, execute, fetch_all
from src.evaluation.benchmark_utils import explain_io, throughput_qps
from src.search.sequential_knn import SequentialKNN

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_QUERIES = [
    "machine learning",
    "deep learning",
    "neural network",
    "gene expression",
    "cancer treatment",
    "medical imaging",
    "protein structure",
    "clinical trial",
]

N_MEDIA_QUERIES = 12  # nº de consultas para imagen/audio


# --------------------------------------------------------------------------- #
# Utilidades comunes
# --------------------------------------------------------------------------- #
def recall_at_k(method_ids, baseline_ids, k):
    base = set(baseline_ids[:k])
    if not base:
        return 0.0
    return len(set(method_ids[:k]) & base) / len(base)


def overlap_at_k(a, b, k):
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(k)


def build_inverted_index(matrix_dense, ids):
    """Índice invertido word_id -> [(item_id, weight)] a partir de histogramas densos."""
    index = defaultdict(list)
    nz_rows, nz_cols = np.nonzero(matrix_dense)
    for r, c in zip(nz_rows, nz_cols):
        index[int(c)].append((ids[r], float(matrix_dense[r, c])))
    return dict(index)


def inverted_search(index, query_vec, top_k):
    """Búsqueda por índice invertido = producto punto disperso (= coseno si normalizado)."""
    scores = defaultdict(float)
    q = np.asarray(query_vec).ravel()
    for word_id in np.nonzero(q)[0]:
        for item_id, w in index.get(int(word_id), ()):  # postings del término
            scores[item_id] += float(q[word_id]) * w
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [i for i, _ in ranked]


def scales_for(n_total, proposed):
    """Recorta la lista de escalas al total disponible y añade el total como escala máxima."""
    scales = sorted({s for s in proposed if 0 < s < n_total} | {n_total})
    return scales


# --------------------------------------------------------------------------- #
# PostgreSQL: tablas por escala
# --------------------------------------------------------------------------- #
def pg_setup_text(ids):
    execute("DROP TABLE IF EXISTS bench_text CASCADE;")
    execute("""
        CREATE TABLE bench_text (
            chunk_id TEXT PRIMARY KEY,
            chunk_text TEXT,
            search_vector tsvector GENERATED ALWAYS AS
                (to_tsvector('english', coalesce(chunk_text, ''))) STORED
        );
    """)
    execute(
        "INSERT INTO bench_text(chunk_id, chunk_text) "
        "SELECT chunk_id, chunk_text FROM text_chunks WHERE chunk_id = ANY(%(ids)s);",
        {"ids": list(ids)},
    )
    execute("CREATE INDEX bench_text_gin ON bench_text USING GIN(search_vector);")
    execute("ANALYZE bench_text;")


def pg_query_text(q, k):
    sql = """
        SELECT chunk_id
        FROM bench_text
        WHERE search_vector @@ plainto_tsquery('english', %(q)s)
        ORDER BY ts_rank(search_vector, plainto_tsquery('english', %(q)s)) DESC
        LIMIT %(k)s;
    """
    rows = fetch_all(sql, {"q": q, "k": k})
    # latencia = tiempo de ejecución del lado del servidor (aísla el índice; excluye
    # el costo de abrir conexión/red que penalizaría injustamente a PostgreSQL)
    hit, read, exec_ms = explain_io(sql, {"q": q, "k": k})
    return [r["chunk_id"] for r in rows], exec_ms, hit, read


def pg_setup_vector(table, id_col, id_type, source_table, ids):
    execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    execute(f"""
        CREATE TABLE {table} (
            {id_col} {id_type} PRIMARY KEY,
            histogram vector(256)
        );
    """)
    execute(
        f"INSERT INTO {table}({id_col}, histogram) "
        f"SELECT {id_col}, histogram FROM {source_table} WHERE {id_col} = ANY(%(ids)s);",
        {"ids": list(ids)},
    )
    execute(f"CREATE INDEX {table}_hnsw ON {table} USING hnsw (histogram vector_cosine_ops);")
    execute(f"ANALYZE {table};")


def pg_query_vector(table, id_col, qvec_literal, k):
    sql = f"""
        SELECT {id_col}
        FROM {table}
        ORDER BY histogram <=> %(qvec)s::vector
        LIMIT %(k)s;
    """
    rows = fetch_all(sql, {"qvec": qvec_literal, "k": k})
    hit, read, exec_ms = explain_io(sql, {"qvec": qvec_literal, "k": k})
    return [r[id_col] for r in rows], exec_ms, hit, read


def pg_sizes(table):
    rows = fetch_all(
        "SELECT pg_indexes_size(%(t)s::regclass) AS idx, "
        "pg_total_relation_size(%(t)s::regclass) AS tab;",
        {"t": table},
    )
    return int(rows[0]["idx"]), int(rows[0]["tab"])


def vec_literal(vec):
    arr = np.asarray(vec, dtype="float32").ravel()
    return "[" + ",".join(f"{float(v):.8f}" for v in arr) + "]"


# --------------------------------------------------------------------------- #
# Motor genérico de escalamiento para modalidades vectoriales (imagen/audio)
# --------------------------------------------------------------------------- #
def run_vector_modality(modality, application, ids, matrix, queries_idx,
                        pg_table, id_col, id_type, source_table, top_k, pg_method_name):
    rows = []
    scales = scales_for(len(ids), PROPOSED_SCALES[modality])
    print(f"\n[{modality}] N total={len(ids)}  escalas={scales}")

    for N in scales:
        subset_ids = ids[:N]
        subset_mat = matrix[:N]

        # Baseline exacto (secuencial)
        seq = SequentialKNN(subset_mat, subset_ids)

        # Índice invertido propio (con medición de memoria)
        tracemalloc.start()
        inv = build_inverted_index(subset_mat, subset_ids)
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        inv_mem_mb = mem_peak / (1024 * 1024)

        # PostgreSQL: tabla + índice HNSW del tamaño N
        pg_setup_vector(pg_table, id_col, id_type, source_table, subset_ids)
        pg_idx_bytes, pg_tab_bytes = pg_sizes(pg_table)

        for qi in queries_idx:
            qvec = matrix[qi]
            qlit = vec_literal(qvec)

            t0 = time.perf_counter()
            seq_res = [i for i, _ in seq.search(qvec, top_k)]
            seq_lat = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            cust_res = inverted_search(inv, qvec, top_k)
            cust_lat = (time.perf_counter() - t0) * 1000

            pg_res, pg_lat, hit, read = pg_query_vector(pg_table, id_col, qlit, top_k)

            qid = str(ids[qi])
            rows.append(_row(modality, application, "sequential_cosine", qid, N, top_k,
                             seq_lat, recall_at_k(seq_res, seq_res, top_k),
                             overlap_at_k(seq_res, pg_res, top_k), seq_lat, 0.0, 0, 0, 0, 0))
            rows.append(_row(modality, application, "custom_inverted", qid, N, top_k,
                             cust_lat, recall_at_k(cust_res, seq_res, top_k),
                             overlap_at_k(cust_res, pg_res, top_k), cust_lat, inv_mem_mb, 0, 0, 0, 0))
            rows.append(_row(modality, application, pg_method_name, qid, N, top_k,
                             pg_lat, recall_at_k(pg_res, seq_res, top_k),
                             overlap_at_k(cust_res, pg_res, top_k), pg_lat, 0.0,
                             hit or 0, read or 0, pg_idx_bytes, pg_tab_bytes))

        execute(f"DROP TABLE IF EXISTS {pg_table} CASCADE;")
        print(f"  N={N:6d}  seq/cust/pg listos")
    return rows


def run_text_modality(top_k):
    rows = []
    chunks = pd.read_csv(PMC_CHUNKS_CSV)
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    with open(TEXT_TFIDF_MATRIX_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(TEXT_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    q_vecs = vectorizer.transform(TEXT_QUERIES)  # (nq, V) L2-normalizado
    scales = scales_for(len(chunk_ids), PROPOSED_SCALES["text"])
    print(f"\n[text] N total={len(chunk_ids)}  escalas={scales}")

    for N in scales:
        subset_ids = chunk_ids[:N]
        subset_mat = tfidf[:N]

        seq = SequentialKNN(subset_mat, subset_ids)

        # Índice invertido (SPIMI) sobre el subconjunto, con memoria
        tracemalloc.start()
        inv = defaultdict(list)
        coo = subset_mat.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            inv[int(c)].append((subset_ids[r], float(v)))
        inv = dict(inv)
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        inv_mem_mb = mem_peak / (1024 * 1024)

        pg_setup_text(subset_ids)
        pg_idx_bytes, pg_tab_bytes = pg_sizes("bench_text")

        for qi, qtext in enumerate(TEXT_QUERIES):
            qrow = q_vecs.getrow(qi)
            qdense = np.asarray(qrow.todense()).ravel()

            t0 = time.perf_counter()
            seq_res = [i for i, _ in seq.search(qdense, top_k)]
            seq_lat = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            scores = defaultdict(float)
            for term_id, w in zip(qrow.indices, qrow.data):
                for cid, dw in inv.get(int(term_id), ()):  # postings
                    scores[cid] += float(w) * dw
            cust_res = [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            cust_lat = (time.perf_counter() - t0) * 1000

            pg_res, pg_lat, hit, read = pg_query_text(qtext, top_k)

            rows.append(_row("text", "document_search", "sequential_cosine", str(qi), N, top_k,
                             seq_lat, 1.0, overlap_at_k(seq_res, pg_res, top_k), seq_lat, 0.0, 0, 0, 0, 0,
                             query_text=qtext))
            rows.append(_row("text", "document_search", "custom_inverted", str(qi), N, top_k,
                             cust_lat, recall_at_k(cust_res, seq_res, top_k),
                             overlap_at_k(cust_res, pg_res, top_k), cust_lat, inv_mem_mb, 0, 0, 0, 0,
                             query_text=qtext))
            rows.append(_row("text", "document_search", "postgres_gin", str(qi), N, top_k,
                             pg_lat, recall_at_k(pg_res, seq_res, top_k),
                             overlap_at_k(cust_res, pg_res, top_k), pg_lat, 0.0,
                             hit or 0, read or 0, pg_idx_bytes, pg_tab_bytes, query_text=qtext))

        execute("DROP TABLE IF EXISTS bench_text CASCADE;")
        print(f"  N={N:6d}  seq/cust/pg listos")
    return rows


def _row(modality, application, method, qid, N, top_k, latency, recall, overlap,
         exec_ms, mem_mb, hit, read, idx_bytes, tab_bytes, query_text=None):
    return {
        "modality": modality, "application": application, "method": method,
        "query_id": qid, "query_text": query_text, "dataset_size": N, "top_k": top_k,
        "latency_ms": latency, "throughput_qps": throughput_qps(latency),
        "recall_at_k": recall, "overlap_at_k": overlap, "execution_time_ms": exec_ms,
        "memory_peak_mb": mem_mb, "shared_hit_blocks": hit, "shared_read_blocks": read,
        "index_size_bytes": idx_bytes, "table_size_bytes": tab_bytes,
    }


def save(rows, modality):
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"benchmark_{modality}.csv"
    df.to_csv(out, index=False)
    pg_rows = [(
        r["modality"], r["application"], r["method"], r["query_id"], r["query_text"],
        r["dataset_size"], r["top_k"], r["latency_ms"], r["throughput_qps"],
        r["recall_at_k"], r["overlap_at_k"], r["execution_time_ms"],
        r["shared_hit_blocks"], r["shared_read_blocks"], r["index_size_bytes"], r["table_size_bytes"]
    ) for r in rows]
    bulk_insert("""
        INSERT INTO benchmark_results(
            modality, application, method, query_id, query_text, dataset_size, top_k,
            latency_ms, throughput_qps, recall_at_k, overlap_at_k, execution_time_ms,
            shared_hit_blocks, shared_read_blocks, index_size_bytes, table_size_bytes
        ) VALUES %s
    """, pg_rows)
    print(f"Guardado {out}  ({len(df)} filas)")


PROPOSED_SCALES = {
    "text": [500, 1000, 2000, 5000],
    "image": [200, 400, 800],
    "audio": [1000, 2000, 4000],
}


def load_vector_csv(csv_path, id_col, prefix):
    df = pd.read_csv(csv_path).fillna(0)
    hist_cols = sorted([c for c in df.columns if c.startswith(prefix)],
                       key=lambda x: int(x.split("_")[-1]))
    ids = df[id_col].tolist()
    matrix = df[hist_cols].to_numpy(dtype="float64")
    return ids, matrix


def main(top_k: int = TOP_K, modalities=("text", "image", "audio")):
    execute("TRUNCATE benchmark_results;")

    if "text" in modalities:
        save(run_text_modality(top_k), "text")

    if "image" in modalities:
        ids, matrix = load_vector_csv(PMC_VISUAL_HIST_CSV, "image_id", "vw_")
        nq = min(N_MEDIA_QUERIES, min(PROPOSED_SCALES["image"]))
        qidx = list(np.linspace(0, nq - 1, nq, dtype=int))
        rows = run_vector_modality("image", "document_search", ids, matrix, qidx,
                                   "bench_img", "image_id", "TEXT", "image_histograms",
                                   top_k, "postgres_hnsw")
        save(rows, "image")

    if "audio" in modalities:
        ids, matrix = load_vector_csv(FMA_AUDIO_HIST_CSV, "track_id", "aw_")
        nq = min(N_MEDIA_QUERIES, min(PROPOSED_SCALES["audio"]))
        qidx = list(np.linspace(0, nq - 1, nq, dtype=int))
        rows = run_vector_modality("audio", "music_search", ids, matrix, qidx,
                                   "bench_audio", "track_id", "INT", "audio_histograms",
                                   top_k, "postgres_hnsw")
        save(rows, "audio")

    print("\nBenchmark de escalamiento completo.")


if __name__ == "__main__":
    mods = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ("text", "image", "audio")
    main(modalities=mods)

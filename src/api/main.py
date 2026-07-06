from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import ROOT_DIR, TOP_K
from src.search.custom_text_search import CustomTextSearch
from src.search.custom_image_search import CustomImageSearch
from src.search.custom_audio_search import CustomAudioSearch
from src.search.pg_search import pg_text_search, pg_image_search, pg_audio_search
from src.search.query_parser import parse_sql, QueryParseError
from src.database.connection import fetch_all

app = FastAPI(title="Sistema Multimodal de Recuperación y Búsqueda - BD2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_text_engine = None
_image_engine = None
_audio_engine = None


def get_text_engine():
    global _text_engine
    if _text_engine is None:
        _text_engine = CustomTextSearch()
    return _text_engine


def get_image_engine():
    global _image_engine
    if _image_engine is None:
        _image_engine = CustomImageSearch()
    return _image_engine


def get_audio_engine():
    global _audio_engine
    if _audio_engine is None:
        _audio_engine = CustomAudioSearch()
    return _audio_engine


@app.get("/health")
def health():
    return {"status": "ok", "project": "multimodal-search-bd2"}


def _one_value(sql: str, default=0):
    rows = fetch_all(sql)
    if not rows:
        return default
    return next(iter(rows[0].values()))


@app.get("/datasets/stats")
def dataset_stats():
    tables = {
        "documents": "documents",
        "text_chunks": "text_chunks",
        "images": "images",
        "songs": "songs",
        "text_codewords": "text_codebook",
        "visual_codewords": "visual_codebook",
        "audio_codewords": "audio_codebook",
        "image_histograms": "image_histograms",
        "audio_histograms": "audio_histograms",
        "text_postings": "text_inverted_index",
        "visual_postings": "visual_inverted_index",
        "audio_postings": "audio_inverted_index",
    }
    counts = {name: int(_one_value(f"SELECT COUNT(*) AS n FROM {table};")) for name, table in tables.items()}
    genre_rows = fetch_all("""
        SELECT coalesce(genre_top, 'Unknown') AS genre, COUNT(*) AS n
        FROM songs
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 12;
    """)
    return {
        "counts": counts,
        "music_genres": [dict(r) for r in genre_rows],
        "scale_hint": {
            "small_chunks": 1000,
            "medium_chunks": 5000,
            "large_chunks": 40000,
        },
    }


@app.get("/documents/search/text")
def search_documents_text(
    q: str = Query(..., description="Consulta textual"),
    method: str = Query("custom", description="custom | gin"),
    top_k: int = TOP_K,
):
    if method == "gin":
        return pg_text_search(q, top_k)
    return get_text_engine().search(q, top_k)


@app.get("/documents/search/text/compare")
def compare_documents_text(q: str = Query(...), top_k: int = TOP_K):
    custom = get_text_engine().search(q, top_k)
    postgres = pg_text_search(q, top_k)
    custom_ids = [r["chunk_id"] for r in custom["results"]]
    postgres_ids = [r["chunk_id"] for r in postgres["results"]]
    overlap = len(set(custom_ids) & set(postgres_ids)) / max(1, top_k)
    return {
        "query": q,
        "top_k": top_k,
        "overlap_at_k": overlap,
        "custom": custom,
        "postgres": postgres,
    }


@app.get("/documents/examples/images")
def document_image_examples(limit: int = 12):
    rows = fetch_all("""
        SELECT i.image_id, i.article_id, i.image_path, i.original_filename, d.title
        FROM images i
        JOIN documents d ON d.article_id = i.article_id
        WHERE i.image_path IS NOT NULL AND i.image_path <> ''
        ORDER BY i.article_id, i.image_id
        LIMIT %(limit)s;
    """, {"limit": limit})
    return {"results": [dict(r) for r in rows]}


@app.get("/documents/{article_id}")
def document_detail(article_id: str, chunk_limit: int = 5, image_limit: int = 8):
    documents = fetch_all("""
        SELECT article_id, pmcid, title, doi, citation, license_code, raw_path
        FROM documents
        WHERE article_id = %(article_id)s;
    """, {"article_id": article_id})
    chunks = fetch_all("""
        SELECT chunk_id, chunk_order, chunk_text
        FROM text_chunks
        WHERE article_id = %(article_id)s
        ORDER BY chunk_order
        LIMIT %(chunk_limit)s;
    """, {"article_id": article_id, "chunk_limit": chunk_limit})
    images = fetch_all("""
        SELECT image_id, image_path, original_filename
        FROM images
        WHERE article_id = %(article_id)s
        ORDER BY image_id
        LIMIT %(image_limit)s;
    """, {"article_id": article_id, "image_limit": image_limit})
    return {
        "document": dict(documents[0]) if documents else None,
        "chunks": [dict(r) for r in chunks],
        "images": [dict(r) for r in images],
    }


@app.post("/documents/search/image")
def search_documents_image(
    file: UploadFile = File(...),
    method: str = Query("custom", description="custom | pgvector"),
    top_k: int = TOP_K,
):
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        if method == "pgvector":
            return pg_image_search(tmp_path, top_k)
        return get_image_engine().search(tmp_path, top_k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/music/search/audio")
def search_music_audio(
    file: UploadFile = File(...),
    method: str = Query("custom", description="custom | pgvector"),
    top_k: int = TOP_K,
):
    suffix = Path(file.filename or "query.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        if method == "pgvector":
            return pg_audio_search(tmp_path, top_k)
        return get_audio_engine().search(tmp_path, top_k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/music/search/metadata")
def search_music_metadata(q: str, top_k: int = TOP_K):
    t0 = time.perf_counter()
    rows = fetch_all("""
        SELECT track_id, title, artist_name, genre_top,
               ts_rank(search_vector, plainto_tsquery('english', %(q)s)) AS score
        FROM songs
        WHERE search_vector @@ plainto_tsquery('english', %(q)s)
        ORDER BY score DESC
        LIMIT %(top_k)s;
    """, {"q": q, "top_k": top_k})
    elapsed = (time.perf_counter() - t0) * 1000
    return {"method": "postgres_gin_music_metadata", "latency_ms": elapsed, "results": [dict(r) for r in rows]}


@app.get("/music/examples/audio")
def music_audio_examples(limit: int = 12):
    rows = fetch_all("""
        SELECT ah.track_id, s.title, s.artist_name, s.genre_top, s.audio_path
        FROM audio_histograms ah
        JOIN songs s ON s.track_id = ah.track_id
        WHERE s.audio_path IS NOT NULL AND s.audio_path <> ''
        ORDER BY ah.track_id
        LIMIT %(limit)s;
    """, {"limit": limit})
    return {"results": [dict(r) for r in rows]}


def _music_metadata_search(q: str, top_k: int):
    t0 = time.perf_counter()
    rows = fetch_all("""
        SELECT track_id, title, artist_name, genre_top,
               ts_rank(search_vector, plainto_tsquery('english', %(q)s)) AS score
        FROM songs
        WHERE search_vector @@ plainto_tsquery('english', %(q)s)
        ORDER BY score DESC
        LIMIT %(top_k)s;
    """, {"q": q, "top_k": top_k})
    elapsed = (time.perf_counter() - t0) * 1000
    return {"method": "postgres_gin_music_metadata", "latency_ms": elapsed,
            "results": [dict(r) for r in rows]}


@app.get("/query/sql")
def query_sql(sql: str = Query(..., description="Consulta SQL multimodal")):
    """ParserSQL: ejecuta una consulta tipo SQL sobre texto, imagen o audio."""
    try:
        plan = parse_sql(sql)
    except QueryParseError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        if plan.modality == "text":
            data = (pg_text_search if plan.method == "postgres" else get_text_engine().search)(
                plan.operand, plan.top_k)
        elif plan.modality == "music_meta":
            data = _music_metadata_search(plan.operand, plan.top_k)
        elif plan.modality == "image":
            if not Path(plan.operand).exists():
                return {"ok": False, "error": f"No existe la imagen: {plan.operand}"}
            data = (pg_image_search if plan.method == "postgres" else get_image_engine().search)(
                plan.operand, plan.top_k)
        elif plan.modality == "audio":
            if not Path(plan.operand).exists():
                return {"ok": False, "error": f"No existe el audio: {plan.operand}"}
            data = (pg_audio_search if plan.method == "postgres" else get_audio_engine().search)(
                plan.operand, plan.top_k)
        else:
            return {"ok": False, "error": "Modalidad no soportada."}
    except Exception as exc:  # errores de ejecución (archivo ilegible, etc.)
        return {"ok": False, "error": f"Error al ejecutar: {exc}"}

    return {
        "ok": True,
        "plan": {
            "modality": plan.modality, "method": plan.method,
            "operand": plan.operand, "top_k": plan.top_k, "notes": plan.notes,
        },
        "latency_ms": data.get("latency_ms"),
        "results": data.get("results", []),
    }


@app.get("/benchmarks/results")
def benchmark_results(limit: int = 100):
    rows = fetch_all("""
        SELECT * FROM benchmark_results
        ORDER BY created_at DESC
        LIMIT %(limit)s;
    """, {"limit": limit})
    return {"results": [dict(r) for r in rows]}


@app.post("/benchmarks/run/{modality}")
def run_benchmark(modality: str):
    if modality == "plots":
        cmd = [sys.executable, "src/evaluation/plot_results.py"]
    elif modality in {"text", "image", "audio"}:
        cmd = [sys.executable, "src/evaluation/benchmark_scaling.py", modality]
    else:
        return {"ok": False, "error": "modality debe ser: text, image, audio o plots"}

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "ok": proc.returncode == 0,
        "modality": modality,
        "elapsed_ms": elapsed,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "returncode": proc.returncode,
    }

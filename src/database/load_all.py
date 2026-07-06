from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ROOT_DIR,
    PMC_ARTICLES_CSV,
    PMC_CHUNKS_CSV,
    PMC_IMAGES_CSV,
    PMC_TEXT_CODEBOOK_CSV,
    PMC_VISUAL_HIST_CSV,
    FMA_TRACKS_CSV,
    FMA_AUDIO_HIST_CSV,
    PMC_VISUAL_KMEANS_PATH,
    FMA_AUDIO_KMEANS_PATH,
    TEXT_INDEX_CSV,
    PMC_VISUAL_INDEX_CSV,
    FMA_AUDIO_INDEX_CSV,
)
from src.database.connection import bulk_insert, execute, get_connection
from src.utils.vectors import vector_to_pg_literal


def read_sql_file(path: str):
    return (ROOT_DIR / path).read_text(encoding="utf-8")


def init_schema():
    print("Inicializando extensiones, esquema e índices...")
    for path in ["database/01_extensions.sql", "database/02_schema.sql"]:
        execute(read_sql_file(path))


def load_documents():
    if not PMC_ARTICLES_CSV.exists():
        print("No existe", PMC_ARTICLES_CSV)
        return
    df = pd.read_csv(PMC_ARTICLES_CSV).fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("article_id", "")), str(r.get("pmcid", "")), str(r.get("version", "")),
            str(r.get("title", "")), str(r.get("doi", "")), str(r.get("citation", "")),
            str(r.get("license_code", "")), str(r.get("raw_path", "")), json.dumps({})
        ))
    bulk_insert("""
        INSERT INTO documents(article_id, pmcid, version, title, doi, citation, license_code, raw_path, metadata)
        VALUES %s
        ON CONFLICT (article_id) DO UPDATE SET title = EXCLUDED.title
    """, rows)
    print("documents:", len(rows))


def load_text_chunks():
    if not PMC_CHUNKS_CSV.exists():
        print("No existe", PMC_CHUNKS_CSV)
        return
    df = pd.read_csv(PMC_CHUNKS_CSV).fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("chunk_id", "")), str(r.get("article_id", "")), int(r.get("chunk_order", 0)),
            None if str(r.get("page_number", "")) == "" else int(r.get("page_number", 0)),
            str(r.get("chunk_text", "")), str(r.get("clean_text", r.get("chunk_text", "")))
        ))
    bulk_insert("""
        INSERT INTO text_chunks(chunk_id, article_id, chunk_order, page_number, chunk_text, clean_text)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET chunk_text = EXCLUDED.chunk_text, clean_text = EXCLUDED.clean_text
    """, rows)
    print("text_chunks:", len(rows))


def load_images():
    if not PMC_IMAGES_CSV.exists():
        print("No existe", PMC_IMAGES_CSV)
        return
    df = pd.read_csv(PMC_IMAGES_CSV).fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("image_id", "")), str(r.get("article_id", "")), str(r.get("image_path", "")),
            str(r.get("original_filename", "")), json.dumps({})
        ))
    bulk_insert("""
        INSERT INTO images(image_id, article_id, image_path, original_filename, metadata)
        VALUES %s
        ON CONFLICT (image_id) DO UPDATE SET image_path = EXCLUDED.image_path
    """, rows)
    print("images:", len(rows))


def load_text_codebook():
    if not PMC_TEXT_CODEBOOK_CSV.exists():
        print("No existe", PMC_TEXT_CODEBOOK_CSV)
        return
    df = pd.read_csv(PMC_TEXT_CODEBOOK_CSV).fillna("")
    term_col = "term" if "term" in df.columns else df.columns[0]
    if "term_id" not in df.columns:
        df["term_id"] = range(len(df))
    rows = [(int(r["term_id"]), str(r[term_col])) for _, r in df.iterrows()]
    bulk_insert("""
        INSERT INTO text_codebook(term_id, term)
        VALUES %s
        ON CONFLICT (term_id) DO UPDATE SET term = EXCLUDED.term
    """, rows)
    print("text_codebook:", len(rows))


def load_codebook_from_kmeans(path: Path, table: str, id_col: str):
    if not path.exists():
        print("No existe", path)
        return
    import pickle
    with open(path, "rb") as f:
        kmeans = pickle.load(f)
    centroids = np.asarray(kmeans.cluster_centers_, dtype="float64")
    rows = [(i, list(map(float, c))) for i, c in enumerate(centroids)]
    bulk_insert(f"""
        INSERT INTO {table}({id_col}, centroid)
        VALUES %s
        ON CONFLICT ({id_col}) DO UPDATE SET centroid = EXCLUDED.centroid
    """, rows)
    print(table, len(rows))


def load_image_histograms():
    if not PMC_VISUAL_HIST_CSV.exists():
        print("No existe", PMC_VISUAL_HIST_CSV)
        return
    df = pd.read_csv(PMC_VISUAL_HIST_CSV).fillna(0)
    rows = []
    for _, r in df.iterrows():
        hist_cols = sorted([c for c in df.columns if c.startswith("vw_")], key=lambda x: int(x.split("_")[-1]))
        vec = [float(r[c]) for c in hist_cols]
        rows.append((str(r["image_id"]), str(r["article_id"]), vector_to_pg_literal(vec)))
    bulk_insert("""
        INSERT INTO image_histograms(image_id, article_id, histogram)
        VALUES %s
        ON CONFLICT (image_id) DO UPDATE SET histogram = EXCLUDED.histogram
    """, rows)
    print("image_histograms:", len(rows))


def load_songs():
    if not FMA_TRACKS_CSV.exists():
        print("No existe", FMA_TRACKS_CSV)
        return
    df = pd.read_csv(FMA_TRACKS_CSV).fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            int(r["track_id"]), str(r.get("title", "")), str(r.get("artist_name", "")),
            str(r.get("album_title", "")), str(r.get("genre_top", "")), str(r.get("subset", "")),
            str(r.get("split", "")), str(r.get("audio_path", "")), json.dumps({})
        ))
    bulk_insert("""
        INSERT INTO songs(track_id, title, artist_name, album_title, genre_top, subset, split, audio_path, metadata)
        VALUES %s
        ON CONFLICT (track_id) DO UPDATE SET title = EXCLUDED.title, audio_path = EXCLUDED.audio_path
    """, rows)
    print("songs:", len(rows))


def load_audio_histograms():
    if not FMA_AUDIO_HIST_CSV.exists():
        print("No existe", FMA_AUDIO_HIST_CSV)
        return
    df = pd.read_csv(FMA_AUDIO_HIST_CSV).fillna(0)
    hist_cols = sorted([c for c in df.columns if c.startswith("aw_")], key=lambda x: int(x.split("_")[-1]))
    rows = []
    for _, r in df.iterrows():
        vec = [float(r[c]) for c in hist_cols]
        rows.append((int(r["track_id"]), vector_to_pg_literal(vec)))
    bulk_insert("""
        INSERT INTO audio_histograms(track_id, histogram)
        VALUES %s
        ON CONFLICT (track_id) DO UPDATE SET histogram = EXCLUDED.histogram
    """, rows)
    print("audio_histograms:", len(rows))


def load_inverted_indexes():
    text_csv = TEXT_INDEX_CSV
    if text_csv.exists():
        df = pd.read_csv(text_csv)
        rows = [(int(r.term_id), str(r.chunk_id), float(r.weight)) for r in df.itertuples(index=False)]
        bulk_insert("""
            INSERT INTO text_inverted_index(term_id, chunk_id, weight)
            VALUES %s
            ON CONFLICT (term_id, chunk_id) DO UPDATE SET weight = EXCLUDED.weight
        """, rows)
        print("text_inverted_index:", len(rows))

    visual_csv = PMC_VISUAL_INDEX_CSV
    if visual_csv.exists():
        df = pd.read_csv(visual_csv)
        rows = [(int(r.visual_word_id), str(r.image_id), float(r.frequency)) for r in df.itertuples(index=False)]
        bulk_insert("""
            INSERT INTO visual_inverted_index(visual_word_id, image_id, frequency)
            VALUES %s
            ON CONFLICT (visual_word_id, image_id) DO UPDATE SET frequency = EXCLUDED.frequency
        """, rows)
        print("visual_inverted_index:", len(rows))

    audio_csv = FMA_AUDIO_INDEX_CSV
    if audio_csv.exists():
        df = pd.read_csv(audio_csv)
        rows = [(int(r.acoustic_word_id), int(r.track_id), float(r.frequency)) for r in df.itertuples(index=False)]
        bulk_insert("""
            INSERT INTO audio_inverted_index(acoustic_word_id, track_id, frequency)
            VALUES %s
            ON CONFLICT (acoustic_word_id, track_id) DO UPDATE SET frequency = EXCLUDED.frequency
        """, rows)
        print("audio_inverted_index:", len(rows))


def create_indexes():
    print("Creando índices PostgreSQL...")
    execute(read_sql_file("database/03_indexes.sql"))


def main():
    init_schema()
    load_documents()
    load_text_chunks()
    load_images()
    load_text_codebook()
    load_codebook_from_kmeans(PMC_VISUAL_KMEANS_PATH, "visual_codebook", "visual_word_id")
    load_codebook_from_kmeans(FMA_AUDIO_KMEANS_PATH, "audio_codebook", "acoustic_word_id")
    load_songs()
    load_image_histograms()
    load_audio_histograms()
    load_inverted_indexes()
    create_indexes()
    print("Carga completa en PostgreSQL.")


if __name__ == "__main__":
    main()

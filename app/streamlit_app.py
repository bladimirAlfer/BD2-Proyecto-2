from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(page_title="Multimodal Search BD2", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --border: #d9dee7;
        --ink: #18202b;
        --muted: #667085;
        --soft: #f6f8fb;
        --accent: #2f6f73;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #fff;
    }
    /* Forzar texto oscuro en las métricas: el fondo es blanco fijo, así que en
       tema oscuro el texto claro quedaría invisible (blanco sobre blanco). */
    [data-testid="stMetricValue"] { color: var(--ink) !important; }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * { color: var(--muted) !important; }
    [data-testid="stMetricDelta"] { color: var(--muted) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: 8px;
        box-shadow: none;
    }
    .result-title {
        font-weight: 650;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }
    .meta-line {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 0.4rem;
    }
    .score-pill {
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 650;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params: Any) -> dict[str, Any]:
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return {}


def api_post(path: str, **params: Any) -> dict[str, Any]:
    try:
        response = requests.post(f"{API_URL}{path}", params=params, timeout=900)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return {}


def api_post_file(path: str, file_name: str, file_bytes: bytes, mime: str, **params: Any) -> dict[str, Any]:
    try:
        files = {"file": (file_name, file_bytes, mime)}
        response = requests.post(f"{API_URL}{path}", params=params, files=files, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return {}


def local_bytes(path: str) -> bytes | None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        st.warning(f"No existe el archivo local: {p}")
        return None
    return p.read_bytes()


def score_text(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/d"


def render_text_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("Sin resultados.")
        return
    for rank, item in enumerate(results, start=1):
        with st.container(border=True):
            title = item.get("title") or item.get("article_id") or item.get("chunk_id")
            st.markdown(f"<div class='result-title'>{rank}. {title}</div>", unsafe_allow_html=True)
            meta = " | ".join(
                str(v)
                for v in [item.get("article_id"), item.get("chunk_id"), item.get("doi")]
                if v
            )
            st.markdown(
                f"<div class='meta-line'>{meta} &nbsp; <span class='score-pill'>score {score_text(item.get('score'))}</span></div>",
                unsafe_allow_html=True,
            )
            st.write(item.get("text", ""))


def render_image_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("Sin resultados visuales.")
        return
    cols = st.columns(4)
    for idx, item in enumerate(results):
        with cols[idx % 4]:
            with st.container(border=True):
                image_path = item.get("image_path")
                if image_path:
                    st.image(image_path, width="stretch")
                st.markdown(f"**{item.get('image_id', '')}**")
                st.caption(f"{item.get('article_id', '')} | score {score_text(item.get('score'))}")


def render_music_results(results: list[dict[str, Any]], show_audio: bool = True) -> None:
    if not results:
        st.info("Sin resultados musicales.")
        return
    for rank, item in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(
                f"<div class='result-title'>{rank}. {item.get('title') or 'Sin título'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='meta-line'>{item.get('artist_name', '')} | {item.get('genre_top', '')} "
                f"| track {item.get('track_id', '')} &nbsp; "
                f"<span class='score-pill'>score {score_text(item.get('score'))}</span></div>",
                unsafe_allow_html=True,
            )
            audio_path = item.get("audio_path")
            if show_audio and audio_path:
                audio_file = Path(audio_path)
                if not audio_file.is_absolute():
                    audio_file = PROJECT_ROOT / audio_file
                if audio_file.exists():
                    st.audio(str(audio_file))
                else:
                    st.caption(str(audio_path))


def stats_view() -> None:
    st.header("Estado del Sistema")
    stats = api_get("/datasets/stats")
    counts = stats.get("counts", {})
    if not counts:
        return

    cols = st.columns(6)
    metrics = [
        ("Documentos", counts.get("documents", 0)),
        ("Chunks texto", counts.get("text_chunks", 0)),
        ("Imágenes", counts.get("images", 0)),
        ("Canciones", counts.get("songs", 0)),
        ("Hist. imagen", counts.get("image_histograms", 0)),
        ("Hist. audio", counts.get("audio_histograms", 0)),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{int(value):,}")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Índices Construidos")
        index_df = pd.DataFrame(
            [
                {"Modalidad": "Texto", "Codebook": counts.get("text_codewords", 0), "Postings": counts.get("text_postings", 0), "Comparativo": "GIN"},
                {"Modalidad": "Imagen", "Codebook": counts.get("visual_codewords", 0), "Postings": counts.get("visual_postings", 0), "Comparativo": "pgvector HNSW"},
                {"Modalidad": "Audio", "Codebook": counts.get("audio_codewords", 0), "Postings": counts.get("audio_postings", 0), "Comparativo": "pgvector HNSW"},
            ]
        )
        st.dataframe(index_df, width="stretch", hide_index=True)
    with right:
        st.subheader("Distribución Musical")
        genres = pd.DataFrame(stats.get("music_genres", []))
        if not genres.empty:
            st.bar_chart(genres.set_index("genre")["n"])


def documents_view() -> None:
    st.header("Documentos")
    text_tab, image_tab, detail_tab = st.tabs(["Texto", "Imagen", "Documento"])

    with text_tab:
        q = st.text_input("Consulta textual", "machine learning", key="doc_text_q")
        col_a, col_b, col_c = st.columns([1, 1, 1])
        method = col_a.segmented_control("Índice", ["custom", "gin", "comparar"], default="comparar")
        top_k = col_b.slider("Top K", 1, 30, 10, key="doc_text_topk")
        run = col_c.button("Buscar", type="primary", width="stretch", key="doc_text_btn")

        if run:
            if method == "comparar":
                data = api_get("/documents/search/text/compare", q=q, top_k=top_k)
                if data:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Custom SPIMI", f"{data.get('custom', {}).get('latency_ms', 0):.2f} ms")
                    c2.metric("PostgreSQL GIN", f"{data.get('postgres', {}).get('latency_ms', 0):.2f} ms")
                    c3.metric("Overlap@K", f"{data.get('overlap_at_k', 0):.2f}")
                    left, right = st.columns(2)
                    with left:
                        st.subheader("Custom")
                        render_text_results(data.get("custom", {}).get("results", []))
                    with right:
                        st.subheader("GIN")
                        render_text_results(data.get("postgres", {}).get("results", []))
            else:
                data = api_get("/documents/search/text", q=q, method=method, top_k=top_k)
                if data:
                    st.metric("Latencia", f"{data.get('latency_ms', 0):.2f} ms")
                    render_text_results(data.get("results", []))

    with image_tab:
        examples = api_get("/documents/examples/images", limit=16).get("results", [])
        source = st.segmented_control("Query", ["ejemplo", "archivo"], default="ejemplo", key="img_source")
        col_a, col_b, col_c = st.columns([1.4, 1, 1])
        method = col_b.segmented_control("Índice", ["custom", "pgvector"], default="custom", key="img_method")
        top_k = col_c.slider("Top K", 1, 30, 12, key="img_topk")

        query_bytes = None
        query_name = "query.jpg"
        query_type = "image/jpeg"

        if source == "archivo":
            uploaded = col_a.file_uploader("Imagen", type=["jpg", "jpeg", "png"], key="image_file")
            if uploaded:
                query_bytes = uploaded.getvalue()
                query_name = uploaded.name
                query_type = uploaded.type or query_type
                st.image(query_bytes, caption=query_name, width=260)
        else:
            options = {f"{r['image_id']} | {r.get('title', '')[:60]}": r for r in examples}
            selected = col_a.selectbox("Imagen", list(options.keys()), key="image_sample") if options else None
            if selected:
                sample = options[selected]
                query_name = Path(sample["image_path"]).name
                query_bytes = local_bytes(sample["image_path"])
                st.image(sample["image_path"], caption=sample["image_id"], width=260)

        if st.button("Buscar imágenes similares", type="primary", key="img_btn") and query_bytes:
            data = api_post_file("/documents/search/image", query_name, query_bytes, query_type, method=method, top_k=top_k)
            if data:
                st.metric("Latencia", f"{data.get('latency_ms', 0):.2f} ms")
                render_image_results(data.get("results", []))

    with detail_tab:
        article_id = st.text_input("Article ID", "PMC13328751.1")
        if st.button("Abrir documento", key="doc_detail_btn"):
            data = api_get(f"/documents/{article_id}", chunk_limit=4, image_limit=8)
            doc = data.get("document")
            if doc:
                st.subheader(doc.get("title") or article_id)
                st.caption(" | ".join(str(v) for v in [doc.get("pmcid"), doc.get("doi"), doc.get("license_code")] if v))
                if doc.get("raw_path"):
                    st.code(doc["raw_path"], language="text")
                st.markdown("**Chunks**")
                for chunk in data.get("chunks", []):
                    with st.expander(chunk.get("chunk_id", "")):
                        st.write(chunk.get("chunk_text", "")[:1500])
                st.markdown("**Imágenes**")
                render_image_results(data.get("images", []))
            else:
                st.info("Documento no encontrado.")


def music_view() -> None:
    st.header("Música")
    text_tab, audio_tab = st.tabs(["Letra / metadata", "Audio"])

    with text_tab:
        q = st.text_input("Consulta musical", "Hip-Hop", key="music_text_q")
        top_k = st.slider("Top K", 1, 30, 10, key="music_text_topk")
        if st.button("Buscar canciones", type="primary", key="music_text_btn"):
            data = api_get("/music/search/metadata", q=q, top_k=top_k)
            if data:
                st.metric("Latencia GIN", f"{data.get('latency_ms', 0):.2f} ms")
                render_music_results(data.get("results", []), show_audio=False)

    with audio_tab:
        examples = api_get("/music/examples/audio", limit=16).get("results", [])
        source = st.segmented_control("Query", ["ejemplo", "archivo"], default="ejemplo", key="audio_source")
        col_a, col_b, col_c = st.columns([1.4, 1, 1])
        method = col_b.segmented_control("Índice", ["custom", "pgvector"], default="custom", key="audio_method")
        top_k = col_c.slider("Top K", 1, 30, 10, key="audio_topk")

        query_bytes = None
        query_name = "query.mp3"
        query_type = "audio/mpeg"

        if source == "archivo":
            uploaded = col_a.file_uploader("Audio", type=["mp3", "wav"], key="audio_file")
            if uploaded:
                query_bytes = uploaded.getvalue()
                query_name = uploaded.name
                query_type = uploaded.type or query_type
                st.audio(query_bytes)
        else:
            options = {f"{r['track_id']} | {r.get('title', '')} | {r.get('artist_name', '')}": r for r in examples}
            selected = col_a.selectbox("Canción", list(options.keys()), key="audio_sample") if options else None
            if selected:
                sample = options[selected]
                query_name = Path(sample["audio_path"]).name
                query_bytes = local_bytes(sample["audio_path"])
                if query_bytes:
                    st.audio(query_bytes)

        if st.button("Buscar canciones similares", type="primary", key="audio_btn") and query_bytes:
            data = api_post_file("/music/search/audio", query_name, query_bytes, query_type, method=method, top_k=top_k)
            if data:
                st.metric("Latencia", f"{data.get('latency_ms', 0):.2f} ms")
                render_music_results(data.get("results", []))


SQL_EXAMPLES = {
    "Texto — índice propio": "SELECT * FROM articles WHERE text @@ 'machine learning' LIMIT 10 USING custom;",
    "Texto — PostgreSQL GIN": "SELECT * FROM articles WHERE text @@ 'deep learning' LIMIT 10 USING postgres;",
    "Música — metadata/género": "SELECT * FROM songs WHERE genre @@ 'Hip-Hop' LIMIT 10;",
}


def sql_console_view() -> None:
    st.header("Consola SQL Multimodal")
    st.markdown(
        "<div class='small-muted'>Recupera texto, imágenes o audio con un dialecto SQL. "
        "Operadores: <code>@@</code> full-text · <code>&lt;-&gt;</code> vecino más cercano (KNN). "
        "Tablas: <code>articles</code>, <code>images</code>, <code>songs</code>. "
        "Cláusulas: <code>LIMIT n</code>, <code>USING custom|postgres</code>.</div>",
        unsafe_allow_html=True,
    )

    # Rutas de ejemplo reales para consultas de imagen/audio
    img_examples = api_get("/documents/examples/images", limit=1).get("results", [])
    audio_examples = api_get("/music/examples/audio", limit=1).get("results", [])
    examples = dict(SQL_EXAMPLES)
    if img_examples:
        p = img_examples[0]["image_path"]
        examples["Imagen — similitud (pgvector)"] = f"SELECT * FROM images WHERE image <-> '{p}' LIMIT 8 USING pgvector;"
    if audio_examples:
        p = audio_examples[0]["audio_path"]
        examples["Audio — similitud (custom)"] = f"SELECT * FROM songs WHERE audio <-> '{p}' LIMIT 10 USING custom;"

    cols = st.columns(len(examples))
    for col, (label, sql) in zip(cols, examples.items()):
        if col.button(label, width="stretch"):
            st.session_state["sql_text"] = sql

    sql = st.text_area(
        "Consulta SQL",
        value=st.session_state.get("sql_text", SQL_EXAMPLES["Texto — índice propio"]),
        height=90,
        key="sql_text",
    )

    if st.button("Ejecutar consulta", type="primary"):
        data = api_get("/query/sql", sql=sql)
        if not data:
            return
        if not data.get("ok"):
            st.error(data.get("error", "Error desconocido."))
            return
        plan = data.get("plan", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Modalidad", plan.get("modality", "-"))
        c2.metric("Método", plan.get("method", "-"))
        c3.metric("Latencia", f"{(data.get('latency_ms') or 0):.2f} ms")
        for note in plan.get("notes", []):
            st.info(note)
        results = data.get("results", [])
        modality = plan.get("modality")
        if modality == "image":
            render_image_results(results)
        elif modality in ("audio", "music_meta"):
            render_music_results(results, show_audio=(modality == "audio"))
        else:
            render_text_results(results)


st.sidebar.title("BD2 Multimodal")
st.sidebar.caption(API_URL)
view = st.sidebar.radio("Módulo", ["Estado", "Consola SQL", "Documentos", "Música"], label_visibility="collapsed")

health = api_get("/health")
if health.get("status") == "ok":
    st.sidebar.success("API conectada")
else:
    st.sidebar.warning("API no disponible")

st.title("Sistema Multimodal de Recuperación y Búsqueda")

if view == "Estado":
    stats_view()
elif view == "Consola SQL":
    sql_console_view()
elif view == "Documentos":
    documents_view()
elif view == "Música":
    music_view()

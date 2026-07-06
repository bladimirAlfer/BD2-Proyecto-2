"""ParserSQL para recuperación de texto y multimedia.

Traduce un dialecto SQL reducido a un plan de búsqueda multimodal. La idea es que
el usuario consulte el motor con una sintaxis declarativa familiar en lugar de
formularios, cubriendo texto (full-text), imagen y audio (vecinos por similitud).

Gramática soportada (case-insensitive):

    SELECT <cols> FROM <tabla>
    WHERE  <campo> <op> '<operando>'
    [LIMIT <k>] [USING <custom|postgres>]

Operadores:
    @@   coincidencia full-text        (texto / metadata musical)
    <->  vecino más cercano (KNN)       (imagen / audio por similitud)

Tablas y modalidad resultante:
    articles | documents | chunks   -> texto           (operando = consulta)
    images                          -> imagen           (operando = ruta de imagen)
    songs | music                   -> metadata musical si @@, audio si <->
    audio                           -> audio            (operando = ruta de audio)

Ejemplos:
    SELECT * FROM articles WHERE text @@ 'machine learning' LIMIT 10 USING custom;
    SELECT * FROM articles WHERE text @@ 'deep learning' USING postgres;
    SELECT * FROM images  WHERE image <-> 'data/.../fig.jpg' LIMIT 8 USING pgvector;
    SELECT * FROM songs   WHERE genre @@ 'Hip-Hop' LIMIT 10;
    SELECT * FROM songs   WHERE audio <-> 'data/.../123.mp3' LIMIT 5 USING custom;
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TEXT_TABLES = {"articles", "article", "documents", "document", "chunks", "text_chunks", "text"}
IMAGE_TABLES = {"images", "image", "figures", "figure"}
MUSIC_TABLES = {"songs", "song", "music", "tracks", "audio"}

CUSTOM_ALIASES = {"custom", "propio", "inverted", "spimi", "bovw", "boaw"}
POSTGRES_ALIASES = {"postgres", "postgresql", "pg", "gin", "pgvector", "hnsw"}


class QueryParseError(ValueError):
    """Error de sintaxis en la consulta SQL multimodal."""


@dataclass
class ParsedQuery:
    modality: str            # 'text' | 'image' | 'audio' | 'music_meta'
    operator: str            # '@@' | '<->'
    operand: str             # texto de consulta o ruta de archivo
    method: str              # 'custom' | 'postgres'
    top_k: int = 10
    table: str = ""
    columns: str = "*"
    raw: str = ""
    notes: list[str] = field(default_factory=list)


_QUERY_RE = re.compile(
    r"""^\s*SELECT\s+(?P<cols>.+?)\s+
        FROM\s+(?P<table>[a-zA-Z_][\w]*)\s+
        WHERE\s+(?P<field>[a-zA-Z_][\w]*)\s*
        (?P<op>@@|<->)\s*
        (?P<quote>['"])(?P<operand>.*?)(?P=quote)
        (?:\s+LIMIT\s+(?P<limit>\d+))?
        (?:\s+USING\s+(?P<method>[a-zA-Z_]+))?
        \s*;?\s*$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _resolve_method(raw_method: str | None, modality: str) -> str:
    if not raw_method:
        # Por defecto: índice propio para texto/imagen/audio; metadata siempre es GIN.
        return "postgres" if modality == "music_meta" else "custom"
    m = raw_method.lower()
    if m in CUSTOM_ALIASES:
        return "custom"
    if m in POSTGRES_ALIASES:
        return "postgres"
    raise QueryParseError(
        f"Método desconocido: '{raw_method}'. Usa USING custom o USING postgres."
    )


def parse_sql(sql: str) -> ParsedQuery:
    if not sql or not sql.strip():
        raise QueryParseError("Consulta vacía.")
    match = _QUERY_RE.match(sql.strip())
    if not match:
        raise QueryParseError(
            "Sintaxis inválida. Formato esperado:\n"
            "SELECT * FROM <tabla> WHERE <campo> <@@|<->> '<valor>' "
            "[LIMIT n] [USING custom|postgres];"
        )

    table = match.group("table").lower()
    op = match.group("op")
    operand = match.group("operand").strip()
    field_name = match.group("field").lower()
    limit = int(match.group("limit")) if match.group("limit") else 10
    if not operand:
        raise QueryParseError("El operando entre comillas no puede estar vacío.")
    if limit <= 0 or limit > 100:
        raise QueryParseError("LIMIT debe estar entre 1 y 100.")

    notes: list[str] = []

    # Determinar modalidad y validar operador
    if table in TEXT_TABLES:
        if op != "@@":
            raise QueryParseError("La búsqueda de texto usa el operador full-text '@@'.")
        modality = "text"
    elif table in IMAGE_TABLES:
        if op != "<->":
            raise QueryParseError("La búsqueda de imágenes usa el operador de similitud '<->'.")
        modality = "image"
    elif table in MUSIC_TABLES:
        if op == "@@":
            modality = "music_meta"   # búsqueda por metadata / género (GIN)
        else:
            modality = "audio"        # similitud acústica (KNN)
    else:
        raise QueryParseError(
            f"Tabla desconocida: '{table}'. Usa articles, images o songs."
        )

    method = _resolve_method(match.group("method"), modality)
    if modality == "music_meta" and method != "postgres":
        notes.append("La búsqueda por metadata musical siempre usa GIN (PostgreSQL).")
        method = "postgres"

    return ParsedQuery(
        modality=modality, operator=op, operand=operand, method=method,
        top_k=limit, table=table, columns=match.group("cols").strip(),
        raw=sql.strip(), notes=notes,
    )

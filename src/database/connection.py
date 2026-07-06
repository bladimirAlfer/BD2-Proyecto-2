from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def fetch_all(sql: str, params=None):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or {})
            return cur.fetchall()


def execute(sql: str, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
        conn.commit()


def bulk_insert(sql: str, rows, page_size: int = 1000):
    if not rows:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)
        conn.commit()

from __future__ import annotations

import resource
from pathlib import Path
from typing import Any

from src.database.connection import fetch_all


def throughput_qps(latency_ms: float | None) -> float | None:
    if latency_ms is None or latency_ms <= 0:
        return None
    return 1000.0 / latency_ms


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def file_size_bytes(*paths: str | Path) -> int:
    total = 0
    for path in paths:
        p = Path(path)
        if p.exists():
            total += p.stat().st_size
    return total


def pg_relation_sizes(table_name: str) -> tuple[int | None, int | None]:
    rows = fetch_all(
        """
        SELECT
            pg_indexes_size(%(table)s::regclass) AS index_size_bytes,
            pg_total_relation_size(%(table)s::regclass) AS table_size_bytes;
        """,
        {"table": table_name},
    )
    if not rows:
        return None, None
    return int(rows[0]["index_size_bytes"]), int(rows[0]["table_size_bytes"])


def _sum_plan_buffers(node: dict[str, Any]) -> tuple[int, int]:
    hit = int(node.get("Shared Hit Blocks", 0) or 0)
    read = int(node.get("Shared Read Blocks", 0) or 0)
    for child in node.get("Plans", []) or []:
        child_hit, child_read = _sum_plan_buffers(child)
        hit += child_hit
        read += child_read
    return hit, read


def explain_io(sql: str, params: dict[str, Any] | None = None) -> tuple[int | None, int | None, float | None]:
    rows = fetch_all("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params or {})
    if not rows:
        return None, None, None
    plan = rows[0]["QUERY PLAN"][0]
    hit, read = _sum_plan_buffers(plan["Plan"])
    return hit, read, float(plan.get("Execution Time", 0.0))

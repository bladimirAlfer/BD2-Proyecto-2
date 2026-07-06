from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def vector_to_pg_literal(vec) -> str:
    arr = np.asarray(vec, dtype="float32").ravel()
    return "[" + ",".join(f"{float(v):.8f}" for v in arr) + "]"


def row_to_vector(row, prefix: str) -> np.ndarray:
    cols = [c for c in row.index if c.startswith(prefix)]
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    return row[cols].to_numpy(dtype="float32")

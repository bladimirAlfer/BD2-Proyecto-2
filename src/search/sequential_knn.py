"""KNN secuencial (fuerza bruta) por Similitud de Coseno.

Sirve como baseline EXACTO para medir recall de los métodos aproximados/indexados
(índice invertido propio y pgvector HNSW) y como la variante "secuencial" que la
rúbrica exige comparar contra la variante "indexada" sobre los histogramas.

Todas las matrices se asumen normalizadas L2 por fila (TF-IDF de sklearn y los
histogramas BoVW/BoAW ya lo están); aun así se normaliza de forma defensiva, de
modo que el producto punto equivale a la similitud de coseno.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from scipy.sparse import csr_matrix


def _l2_normalize_rows(matrix):
    if issparse(matrix):
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        inv = csr_matrix((1.0 / norms, (range(len(norms)), range(len(norms)))))
        return inv.dot(matrix)
    matrix = np.asarray(matrix, dtype="float64")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SequentialKNN:
    """Escaneo lineal exacto por coseno sobre una matriz (dispersa o densa)."""

    def __init__(self, matrix, ids):
        self.ids = list(ids)
        self.matrix = _l2_normalize_rows(matrix)
        self.is_sparse = issparse(self.matrix)

    def search(self, query_vec, top_k: int = 10):
        q = np.asarray(query_vec, dtype="float64").ravel()
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        if self.is_sparse:
            sims = np.asarray(self.matrix.dot(q)).ravel()
        else:
            sims = self.matrix @ q
        # top-k por similitud descendente (argpartition + orden parcial)
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PMC_CHUNKS_CSV, TEXT_TFIDF_MATRIX_PATH, TEXT_INDEX_PATH, TEXT_INDEX_CSV

OUT_CSV = TEXT_INDEX_CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def build_spimi_index(tfidf: csr_matrix, chunk_ids, block_size_docs: int = 1000):
    """
    SPIMI simplificado:
    - Recorre documentos por bloques.
    - Para cada bloque crea postings term_id -> [(chunk_id, weight)].
    - Fusiona bloques en un índice final.
    """
    final_index = defaultdict(list)
    n_docs = tfidf.shape[0]

    for start in tqdm(range(0, n_docs, block_size_docs), desc="SPIMI blocks"):
        end = min(start + block_size_docs, n_docs)
        block = defaultdict(list)
        Xb = tfidf[start:end]

        for local_i in range(Xb.shape[0]):
            row = Xb.getrow(local_i)
            global_i = start + local_i
            chunk_id = chunk_ids[global_i]

            for term_id, weight in zip(row.indices, row.data):
                block[int(term_id)].append((chunk_id, float(weight)))

        for term_id, postings in block.items():
            final_index[term_id].extend(postings)

    return dict(final_index)


def main():
    if not PMC_CHUNKS_CSV.exists():
        raise FileNotFoundError(PMC_CHUNKS_CSV)
    if not TEXT_TFIDF_MATRIX_PATH.exists():
        raise FileNotFoundError(TEXT_TFIDF_MATRIX_PATH)

    chunks = pd.read_csv(PMC_CHUNKS_CSV)
    with open(TEXT_TFIDF_MATRIX_PATH, "rb") as f:
        tfidf = pickle.load(f)

    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    index = build_spimi_index(tfidf, chunk_ids)

    with open(TEXT_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    rows = []
    for term_id, postings in index.items():
        for chunk_id, weight in postings:
            rows.append({"term_id": term_id, "chunk_id": chunk_id, "weight": weight})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Términos indexados:", len(index))
    print("Postings:", len(rows))
    print("Archivo:", OUT_CSV)


if __name__ == "__main__":
    main()

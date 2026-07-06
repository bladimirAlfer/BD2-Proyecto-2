from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import PMC_IMAGES_CSV, PMC_VISUAL_HIST_CSV, PMC_VISUAL_INDEX_PATH, PMC_VISUAL_KMEANS_PATH
from src.utils.vectors import l2_normalize


class CustomImageSearch:
    def __init__(self):
        self.images = pd.read_csv(PMC_IMAGES_CSV).fillna("")
        self.image_by_id = self.images.set_index("image_id").to_dict(orient="index")
        self.hist = pd.read_csv(PMC_VISUAL_HIST_CSV).fillna(0)
        self.hist_cols = sorted([c for c in self.hist.columns if c.startswith("vw_")], key=lambda x: int(x.split("_")[-1]))
        self.hist_by_id = {
            str(r["image_id"]): r[self.hist_cols].to_numpy(dtype="float32")
            for _, r in self.hist.iterrows()
        }
        with open(PMC_VISUAL_INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(PMC_VISUAL_KMEANS_PATH, "rb") as f:
            self.kmeans = pickle.load(f)
        self.sift = cv2.SIFT_create()

    def image_to_histogram(self, image_path: str | Path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer imagen: {image_path}")
        img = cv2.resize(img, (512, 512))
        _, desc = self.sift.detectAndCompute(img, None)
        if desc is None:
            return np.zeros(self.kmeans.n_clusters, dtype="float32")
        labels = self.kmeans.predict(desc.astype("float32"))
        hist, _ = np.histogram(labels, bins=np.arange(self.kmeans.n_clusters + 1))
        return l2_normalize(hist)

    def search(self, image_path: str | Path, top_k: int = 10):
        t0 = time.perf_counter()
        qhist = self.image_to_histogram(image_path)
        scores = defaultdict(float)
        nonzero = np.where(qhist > 0)[0]
        for word_id in nonzero:
            for image_id, freq in self.index.get(int(word_id), []):
                scores[image_id] += float(qhist[word_id]) * float(freq)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        results = []
        for image_id, score in ranked:
            meta = self.image_by_id.get(image_id, {})
            results.append({
                "image_id": image_id,
                "article_id": meta.get("article_id", ""),
                "image_path": meta.get("image_path", ""),
                "score": float(score),
            })
        return {"method": "custom_visual_index", "latency_ms": elapsed, "results": results}

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PMC_IMAGES_CSV, PMC_IMAGE_DESCRIPTOR_MAP, PMC_VISUAL_HIST_CSV, PMC_VISUAL_KMEANS_PATH
from src.utils.vectors import l2_normalize

OUT = PMC_VISUAL_HIST_CSV.parent
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not PMC_IMAGES_CSV.exists():
        raise FileNotFoundError(PMC_IMAGES_CSV)
    if not PMC_IMAGE_DESCRIPTOR_MAP.exists():
        raise FileNotFoundError(PMC_IMAGE_DESCRIPTOR_MAP)
    if not PMC_VISUAL_KMEANS_PATH.exists():
        raise FileNotFoundError(PMC_VISUAL_KMEANS_PATH)

    images = pd.read_csv(PMC_IMAGES_CSV)
    with open(PMC_IMAGE_DESCRIPTOR_MAP, "rb") as f:
        descriptor_map = pickle.load(f)
    with open(PMC_VISUAL_KMEANS_PATH, "rb") as f:
        kmeans = pickle.load(f)

    rows = []
    for _, row in tqdm(images.iterrows(), total=len(images), desc="Visual histograms"):
        image_id = str(row["image_id"])
        if image_id not in descriptor_map:
            continue
        desc = descriptor_map[image_id]
        labels = kmeans.predict(desc)
        hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
        hist = l2_normalize(hist)
        out = {"image_id": image_id, "article_id": str(row["article_id"])}
        for i, v in enumerate(hist):
            out[f"vw_{i}"] = float(v)
        rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv(PMC_VISUAL_HIST_CSV, index=False)
    print("Histogramas visuales:", df.shape)


if __name__ == "__main__":
    main()

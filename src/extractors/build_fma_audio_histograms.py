import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import FMA_AUDIO_HIST_CSV, FMA_AUDIO_KMEANS_PATH, FMA_MFCC_MAP_PATH

MFCC_MAP_PATH = FMA_MFCC_MAP_PATH
KMEANS_PATH = FMA_AUDIO_KMEANS_PATH
OUT = FMA_AUDIO_HIST_CSV.parent
OUT.mkdir(parents=True, exist_ok=True)

with open(MFCC_MAP_PATH, "rb") as f:
    mfcc_map = pickle.load(f)

with open(KMEANS_PATH, "rb") as f:
    kmeans = pickle.load(f)

K_AUDIO = kmeans.n_clusters

rows = []

for track_id, mfcc in tqdm(mfcc_map.items(), desc="Construyendo histogramas"):
    labels = kmeans.predict(mfcc)

    hist, _ = np.histogram(labels, bins=np.arange(K_AUDIO + 1))
    hist = hist.astype("float32")

    # Normalización L2
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    row = {
        "track_id": track_id
    }

    for i, value in enumerate(hist):
        row[f"aw_{i}"] = value

    rows.append(row)

df_hist = pd.DataFrame(rows)
df_hist.to_csv(FMA_AUDIO_HIST_CSV, index=False)

print("Histogramas:", df_hist.shape)

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle

from src.config import FMA_ALL_MFCC_PATH, FMA_AUDIO_KMEANS_PATH, K_AUDIO

INPUT = FMA_ALL_MFCC_PATH
OUT = FMA_AUDIO_KMEANS_PATH.parent
OUT.mkdir(parents=True, exist_ok=True)

X = np.load(INPUT)

kmeans = MiniBatchKMeans(
    n_clusters=K_AUDIO,
    batch_size=4096,
    random_state=42,
    verbose=1
)

kmeans.fit(X)

with open(FMA_AUDIO_KMEANS_PATH, "wb") as f:
    pickle.dump(kmeans, f)

np.save(OUT / "fma_audio_codebook_centroids.npy", kmeans.cluster_centers_)

print("Codebook acústico:", kmeans.cluster_centers_.shape)

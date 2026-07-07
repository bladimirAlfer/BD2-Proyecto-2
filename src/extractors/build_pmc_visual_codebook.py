import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
import pickle

from src.config import K_VISUAL, PMC_SIFT_DESCRIPTORS_PATH

INPUT = PMC_SIFT_DESCRIPTORS_PATH
OUT = PMC_SIFT_DESCRIPTORS_PATH.parent
OUT.mkdir(parents=True, exist_ok=True)

X = np.load(INPUT)

MAX_DESC = int(os.getenv("PMC_MAX_VISUAL_DESCRIPTORS", "200000"))
if X.shape[0] > MAX_DESC:
    idx = np.random.default_rng(42).choice(X.shape[0], MAX_DESC, replace=False)
    X_train = X[idx]
else:
    X_train = X

kmeans = MiniBatchKMeans(
    n_clusters=K_VISUAL,
    batch_size=4096,
    random_state=42,
    verbose=1
)

kmeans.fit(X_train)

with open(OUT / "pmc_visual_codebook_kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

np.save(OUT / "pmc_visual_codebook_centroids.npy", kmeans.cluster_centers_)

print("Codebook visual:", kmeans.cluster_centers_.shape)

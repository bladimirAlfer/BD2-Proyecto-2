from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.config import FMA_AUDIO_HIST_CSV, FMA_AUDIO_INDEX_PATH, FMA_AUDIO_KMEANS_PATH, FMA_TRACKS_CSV
from src.utils.vectors import l2_normalize


class CustomAudioSearch:
    def __init__(self):
        self.tracks = pd.read_csv(FMA_TRACKS_CSV).fillna("")
        self.track_by_id = self.tracks.set_index("track_id").to_dict(orient="index")
        self.hist = pd.read_csv(FMA_AUDIO_HIST_CSV).fillna(0)
        with open(FMA_AUDIO_INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(FMA_AUDIO_KMEANS_PATH, "rb") as f:
            self.kmeans = pickle.load(f)

    def audio_to_histogram(self, audio_path: str | Path):
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
        mfcc = mfcc.T.astype("float32")
        labels = self.kmeans.predict(mfcc)
        hist, _ = np.histogram(labels, bins=np.arange(self.kmeans.n_clusters + 1))
        return l2_normalize(hist)

    def search(self, audio_path: str | Path, top_k: int = 10):
        t0 = time.perf_counter()
        qhist = self.audio_to_histogram(audio_path)
        scores = defaultdict(float)
        nonzero = np.where(qhist > 0)[0]
        for word_id in nonzero:
            for track_id, freq in self.index.get(int(word_id), []):
                scores[int(track_id)] += float(qhist[word_id]) * float(freq)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        results = []
        for track_id, score in ranked:
            meta = self.track_by_id.get(track_id, {})
            results.append({
                "track_id": int(track_id),
                "title": meta.get("title", ""),
                "artist_name": meta.get("artist_name", ""),
                "genre_top": meta.get("genre_top", ""),
                "audio_path": meta.get("audio_path", ""),
                "score": float(score),
            })
        return {"method": "custom_audio_index", "latency_ms": elapsed, "results": results}

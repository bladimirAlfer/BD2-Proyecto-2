import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pickle

from src.config import FMA_ALL_MFCC_PATH, FMA_TRACKS_CSV, FMA_MFCC_MAP_PATH

INPUT = FMA_TRACKS_CSV
OUT = FMA_MFCC_MAP_PATH.parent
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

mfcc_map = {}
all_mfcc = []

MAX_TRACKS = int(os.getenv("FMA_MAX_TRACKS", "1000"))
FMA_DURATION_SECONDS = float(os.getenv("FMA_DURATION_SECONDS", "30"))

if MAX_TRACKS > 0:
    df = df.head(MAX_TRACKS)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo MFCC"):
    track_id = int(row["track_id"])
    audio_path = row["audio_path"]

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=FMA_DURATION_SECONDS)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20,
            n_fft=2048,
            hop_length=512
        )

        # Forma original: (20, frames)
        # La transponemos a: (frames, 20)
        mfcc = mfcc.T.astype("float32")

        mfcc_map[track_id] = mfcc
        all_mfcc.append(mfcc)

    except Exception as e:
        print("Error en", track_id, e)

if all_mfcc:
    all_mfcc = np.vstack(all_mfcc)
else:
    all_mfcc = np.empty((0, 20), dtype="float32")

np.save(FMA_ALL_MFCC_PATH, all_mfcc)

with open(FMA_MFCC_MAP_PATH, "wb") as f:
    pickle.dump(mfcc_map, f)

print("Total vectores MFCC:", all_mfcc.shape)
print("Tracks procesados:", len(mfcc_map))

from __future__ import annotations

import pickle
from collections import defaultdict

import pandas as pd

from src.config import FMA_AUDIO_INDEX_CSV, FMA_AUDIO_INDEX_PATH, PMC_VISUAL_INDEX_CSV, PMC_VISUAL_INDEX_PATH


def build_visual_pickle():
    csv_path = PMC_VISUAL_INDEX_CSV
    out = PMC_VISUAL_INDEX_PATH
    if not csv_path.exists():
        print("No existe", csv_path)
        return
    df = pd.read_csv(csv_path)
    idx = defaultdict(list)
    for r in df.itertuples(index=False):
        idx[int(r.visual_word_id)].append((str(r.image_id), float(r.frequency)))
    with open(out, "wb") as f:
        pickle.dump(dict(idx), f)
    print("PKL visual:", out)


def build_audio_pickle():
    csv_path = FMA_AUDIO_INDEX_CSV
    out = FMA_AUDIO_INDEX_PATH
    if not csv_path.exists():
        print("No existe", csv_path)
        return
    df = pd.read_csv(csv_path)
    idx = defaultdict(list)
    for r in df.itertuples(index=False):
        idx[int(r.acoustic_word_id)].append((int(r.track_id), float(r.frequency)))
    with open(out, "wb") as f:
        pickle.dump(dict(idx), f)
    print("PKL audio:", out)


def main():
    build_visual_pickle()
    build_audio_pickle()


if __name__ == "__main__":
    main()

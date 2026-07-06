import pandas as pd

from src.config import FMA_TRACKS_CSV, RAW_DIR

RAW = RAW_DIR / "fma"
OUT = FMA_TRACKS_CSV.parent
OUT.mkdir(parents=True, exist_ok=True)

tracks_path = RAW / "fma_metadata" / "tracks.csv"
genres_path = RAW / "fma_metadata" / "genres.csv"

tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0)
genres = pd.read_csv(genres_path)

# Columnas útiles
df = pd.DataFrame(index=tracks.index)
df["track_id"] = tracks.index
df["title"] = tracks[("track", "title")]
df["artist_name"] = tracks[("artist", "name")]
df["album_title"] = tracks[("album", "title")]
df["genre_top"] = tracks[("track", "genre_top")]
df["subset"] = tracks[("set", "subset")]
df["split"] = tracks[("set", "split")]

# Solo FMA small
df_small = df[df["subset"] == "small"].copy()

def get_audio_path(track_id):
    tid = f"{int(track_id):06d}"
    return RAW / "fma_small" / tid[:3] / f"{tid}.mp3"

df_small["audio_path"] = df_small["track_id"].apply(get_audio_path)
df_small["exists"] = df_small["audio_path"].apply(lambda p: p.exists())

df_small = df_small[df_small["exists"]]

df_small["audio_path"] = df_small["audio_path"].astype(str)
df_small.to_csv(FMA_TRACKS_CSV, index=False)
genres.to_csv(OUT / "fma_genres.csv", index=False)

print("Tracks small encontrados:", len(df_small))
print(df_small.head())

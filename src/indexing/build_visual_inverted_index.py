from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.config import PMC_VISUAL_HIST_CSV, PMC_VISUAL_INDEX_CSV

HIST_CSV = PMC_VISUAL_HIST_CSV
OUT_CSV = PMC_VISUAL_INDEX_CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not HIST_CSV.exists():
        raise FileNotFoundError(HIST_CSV)

    df = pd.read_csv(HIST_CSV).fillna(0)
    hist_cols = sorted([c for c in df.columns if c.startswith("vw_")], key=lambda x: int(x.split("_")[-1]))

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Visual inverted index"):
        image_id = str(r["image_id"])
        for col in hist_cols:
            val = float(r[col])
            if val > 0:
                rows.append({
                    "visual_word_id": int(col.split("_")[-1]),
                    "image_id": image_id,
                    "frequency": val,
                })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Postings visuales:", len(rows))
    print("Archivo:", OUT_CSV)


if __name__ == "__main__":
    main()

import json
import shutil
import pandas as pd
from tqdm import tqdm

from src.config import PMC_ARTICLES_CSV, PMC_CHUNKS_CSV, PMC_IMAGES_CSV, RAW_DIR

RAW_DIR = RAW_DIR / "pmc" / "articles"
OUT_ARTICLES = PMC_ARTICLES_CSV.parent
OUT_CHUNKS = PMC_CHUNKS_CSV.parent
OUT_IMAGES = PMC_IMAGES_CSV.parent

OUT_ARTICLES.mkdir(parents=True, exist_ok=True)
OUT_CHUNKS.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

def split_text(text: str, chunk_size: int = 250, overlap: int = 50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= 40:
            chunks.append(" ".join(chunk))

    return chunks

records_articles = []
records_chunks = []
records_images = []

article_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]

for article_dir in tqdm(article_dirs, desc="Procesando PMC"):
    json_files = list(article_dir.glob("*.json"))
    txt_files = list(article_dir.glob("*.txt"))
    image_files = list(article_dir.glob("*.jpg")) + list(article_dir.glob("*.png")) + list(article_dir.glob("*.jpeg"))

    if not json_files or not txt_files:
        continue

    with open(json_files[0], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pmcid = metadata.get("pmcid")
    version = metadata.get("version")
    title = metadata.get("title")
    doi = metadata.get("doi")
    citation = metadata.get("citation")
    license_code = metadata.get("license_code")

    with open(txt_files[0], "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    article_id = f"{pmcid}.{version}"

    records_articles.append({
        "article_id": article_id,
        "pmcid": pmcid,
        "version": version,
        "title": title,
        "doi": doi,
        "citation": citation,
        "license_code": license_code,
        "raw_path": str(article_dir)
    })

    chunks = split_text(full_text)

    for idx, chunk in enumerate(chunks):
        records_chunks.append({
            "chunk_id": f"{article_id}_chunk_{idx}",
            "article_id": article_id,
            "chunk_order": idx,
            "chunk_text": chunk
        })

    for img_idx, img_path in enumerate(image_files):
        target_name = f"{article_id}_img_{img_idx}{img_path.suffix.lower()}"
        target_path = OUT_IMAGES / target_name
        shutil.copy(img_path, target_path)

        records_images.append({
            "image_id": f"{article_id}_img_{img_idx}",
            "article_id": article_id,
            "image_path": str(target_path),
            "original_filename": img_path.name
        })

df_articles = pd.DataFrame(records_articles)
df_chunks = pd.DataFrame(records_chunks)
df_images = pd.DataFrame(records_images)

df_articles.to_csv(PMC_ARTICLES_CSV, index=False)
df_chunks.to_csv(PMC_CHUNKS_CSV, index=False)
df_images.to_csv(PMC_IMAGES_CSV, index=False)

print("Artículos:", len(df_articles))
print("Chunks:", len(df_chunks))
print("Imágenes:", len(df_images))

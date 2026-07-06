import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from src.config import K_TEXT, PMC_CHUNKS_CSV

INPUT = PMC_CHUNKS_CSV
OUT = PMC_CHUNKS_CSV.parent
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=K_TEXT,
    min_df=2
)

X = vectorizer.fit_transform(df["chunk_text"].fillna(""))

with open(OUT / "pmc_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(OUT / "pmc_tfidf_matrix.pkl", "wb") as f:
    pickle.dump(X, f)

vocab = pd.DataFrame({
    "term": vectorizer.get_feature_names_out()
})
vocab["term_id"] = range(len(vocab))
vocab.to_csv(OUT / "pmc_text_codebook.csv", index=False)

print("Chunks:", X.shape[0])
print("Codebook textual:", X.shape[1])

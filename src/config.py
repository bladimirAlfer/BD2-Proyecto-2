import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        env_paths = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for env_path in env_paths:
            if not env_path.exists():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return True
        return False

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]


def project_path(env_name: str, default: str) -> Path:
    path = Path(os.getenv(env_name, default))
    if path.is_absolute():
        return path
    return ROOT_DIR / path

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "multimodal_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

K_TEXT = int(os.getenv("K_TEXT", "5000"))
K_VISUAL = int(os.getenv("K_VISUAL", "256"))
K_AUDIO = int(os.getenv("K_AUDIO", "256"))
TOP_K = int(os.getenv("TOP_K", "10"))

DATA_DIR = project_path("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PMC_ARTICLES_CSV = project_path("PMC_ARTICLES_CSV", "data/processed/pmc/articles/pmc_articles.csv")
PMC_CHUNKS_CSV = project_path("PMC_CHUNKS_CSV", "data/processed/pmc/text_chunks/pmc_text_chunks.csv")
PMC_IMAGES_CSV = project_path("PMC_IMAGES_CSV", "data/processed/pmc/images/pmc_images.csv")
PMC_TEXT_CODEBOOK_CSV = project_path("PMC_TEXT_CODEBOOK_CSV", "data/processed/pmc/text_chunks/pmc_text_codebook.csv")
PMC_VISUAL_HIST_CSV = project_path("PMC_VISUAL_HIST_CSV", "data/processed/pmc/images/pmc_visual_histograms.csv")

FMA_TRACKS_CSV = project_path("FMA_TRACKS_CSV", "data/processed/fma/metadata/fma_tracks_small.csv")
FMA_AUDIO_HIST_CSV = project_path("FMA_AUDIO_HIST_CSV", "data/processed/fma/mfcc/fma_audio_histograms.csv")

TEXT_VECTORIZER_PATH = project_path("TEXT_VECTORIZER_PATH", "data/processed/pmc/text_chunks/pmc_tfidf_vectorizer.pkl")
TEXT_TFIDF_MATRIX_PATH = project_path("TEXT_TFIDF_MATRIX_PATH", "data/processed/pmc/text_chunks/pmc_tfidf_matrix.pkl")
TEXT_INDEX_PATH = project_path("TEXT_INDEX_PATH", "data/processed/pmc/text_chunks/pmc_text_inverted_index.pkl")
TEXT_INDEX_CSV = project_path("TEXT_INDEX_CSV", "data/processed/pmc/text_chunks/pmc_text_inverted_index.csv")

PMC_IMAGE_DESCRIPTOR_MAP = project_path("PMC_IMAGE_DESCRIPTOR_MAP", "data/processed/pmc/images/pmc_image_descriptor_map.pkl")
PMC_VISUAL_KMEANS_PATH = project_path("PMC_VISUAL_KMEANS_PATH", "data/processed/pmc/images/pmc_visual_codebook_kmeans.pkl")
PMC_VISUAL_INDEX_PATH = project_path("PMC_VISUAL_INDEX_PATH", "data/processed/pmc/images/pmc_visual_inverted_index.pkl")
PMC_VISUAL_INDEX_CSV = project_path("PMC_VISUAL_INDEX_CSV", "data/processed/pmc/images/pmc_visual_inverted_index.csv")
PMC_SIFT_DESCRIPTORS_PATH = project_path("PMC_SIFT_DESCRIPTORS_PATH", "data/processed/pmc/images/pmc_sift_descriptors.npy")

FMA_MFCC_MAP_PATH = project_path("FMA_MFCC_MAP_PATH", "data/processed/fma/mfcc/fma_mfcc_map.pkl")
FMA_ALL_MFCC_PATH = project_path("FMA_ALL_MFCC_PATH", "data/processed/fma/mfcc/fma_all_mfcc.npy")
FMA_AUDIO_KMEANS_PATH = project_path("FMA_AUDIO_KMEANS_PATH", "data/processed/fma/mfcc/fma_audio_codebook_kmeans.pkl")
FMA_AUDIO_INDEX_PATH = project_path("FMA_AUDIO_INDEX_PATH", "data/processed/fma/mfcc/fma_audio_inverted_index.pkl")
FMA_AUDIO_INDEX_CSV = project_path("FMA_AUDIO_INDEX_CSV", "data/processed/fma/mfcc/fma_audio_inverted_index.csv")

RESULTS_DIR = project_path("RESULTS_DIR", "results")
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

DROP TABLE IF EXISTS benchmark_results CASCADE;
DROP TABLE IF EXISTS audio_inverted_index CASCADE;
DROP TABLE IF EXISTS visual_inverted_index CASCADE;
DROP TABLE IF EXISTS text_inverted_index CASCADE;
DROP TABLE IF EXISTS audio_histograms CASCADE;
DROP TABLE IF EXISTS image_histograms CASCADE;
DROP TABLE IF EXISTS text_codebook CASCADE;
DROP TABLE IF EXISTS audio_codebook CASCADE;
DROP TABLE IF EXISTS visual_codebook CASCADE;
DROP TABLE IF EXISTS songs CASCADE;
DROP TABLE IF EXISTS images CASCADE;
DROP TABLE IF EXISTS text_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

CREATE TABLE documents (
    article_id TEXT PRIMARY KEY,
    pmcid TEXT,
    version TEXT,
    title TEXT,
    doi TEXT,
    citation TEXT,
    license_code TEXT,
    raw_path TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE text_chunks (
    chunk_id TEXT PRIMARY KEY,
    article_id TEXT REFERENCES documents(article_id) ON DELETE CASCADE,
    chunk_order INT,
    page_number INT,
    chunk_text TEXT,
    clean_text TEXT,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(clean_text, chunk_text, ''))
    ) STORED
);

CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    article_id TEXT REFERENCES documents(article_id) ON DELETE CASCADE,
    image_path TEXT,
    original_filename TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE songs (
    track_id INT PRIMARY KEY,
    title TEXT,
    artist_name TEXT,
    album_title TEXT,
    genre_top TEXT,
    subset TEXT,
    split TEXT,
    audio_path TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(artist_name, '') || ' ' || coalesce(genre_top, ''))
    ) STORED
);

CREATE TABLE text_codebook (
    term_id INT PRIMARY KEY,
    term TEXT UNIQUE NOT NULL
);

CREATE TABLE visual_codebook (
    visual_word_id INT PRIMARY KEY,
    centroid FLOAT8[] NOT NULL
);

CREATE TABLE audio_codebook (
    acoustic_word_id INT PRIMARY KEY,
    centroid FLOAT8[] NOT NULL
);

CREATE TABLE text_inverted_index (
    term_id INT REFERENCES text_codebook(term_id),
    chunk_id TEXT REFERENCES text_chunks(chunk_id) ON DELETE CASCADE,
    weight FLOAT8 NOT NULL,
    PRIMARY KEY (term_id, chunk_id)
);

CREATE TABLE visual_inverted_index (
    visual_word_id INT,
    image_id TEXT REFERENCES images(image_id) ON DELETE CASCADE,
    frequency FLOAT8 NOT NULL,
    PRIMARY KEY (visual_word_id, image_id)
);

CREATE TABLE audio_inverted_index (
    acoustic_word_id INT,
    track_id INT REFERENCES songs(track_id) ON DELETE CASCADE,
    frequency FLOAT8 NOT NULL,
    PRIMARY KEY (acoustic_word_id, track_id)
);

-- Pgvector comparativo. Dimensión por defecto = 256.
CREATE TABLE image_histograms (
    image_id TEXT PRIMARY KEY REFERENCES images(image_id) ON DELETE CASCADE,
    article_id TEXT REFERENCES documents(article_id) ON DELETE CASCADE,
    histogram vector(256) NOT NULL
);

CREATE TABLE audio_histograms (
    track_id INT PRIMARY KEY REFERENCES songs(track_id) ON DELETE CASCADE,
    histogram vector(256) NOT NULL
);

CREATE TABLE benchmark_results (
    benchmark_id SERIAL PRIMARY KEY,
    modality TEXT NOT NULL,
    application TEXT NOT NULL,
    method TEXT NOT NULL,
    query_id TEXT,
    query_text TEXT,
    dataset_size INT,
    top_k INT,
    latency_ms FLOAT8,
    throughput_qps FLOAT8,
    recall_at_k FLOAT8,
    overlap_at_k FLOAT8,
    execution_time_ms FLOAT8,
    shared_hit_blocks INT,
    shared_read_blocks INT,
    index_size_bytes BIGINT,
    table_size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

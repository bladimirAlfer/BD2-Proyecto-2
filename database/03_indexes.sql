CREATE INDEX IF NOT EXISTS idx_text_chunks_gin
ON text_chunks USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_songs_gin
ON songs USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_text_inv_term
ON text_inverted_index(term_id);

CREATE INDEX IF NOT EXISTS idx_visual_inv_word
ON visual_inverted_index(visual_word_id);

CREATE INDEX IF NOT EXISTS idx_audio_inv_word
ON audio_inverted_index(acoustic_word_id);

CREATE INDEX IF NOT EXISTS idx_image_hist_hnsw
ON image_histograms USING hnsw (histogram vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_audio_hist_hnsw
ON audio_histograms USING hnsw (histogram vector_cosine_ops);

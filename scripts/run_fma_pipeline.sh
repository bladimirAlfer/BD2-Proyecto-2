#!/bin/bash
set -e

python src/ingestion/prepare_fma_metadata.py
python src/extractors/extract_fma_mfcc.py
python src/extractors/build_fma_audio_codebook.py
python src/extractors/build_fma_audio_histograms.py
python src/indexing/build_audio_inverted_index.py
python src/indexing/build_pickle_indexes.py

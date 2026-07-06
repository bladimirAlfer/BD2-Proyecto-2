#!/bin/bash
set -e

python src/extractors/build_pmc_visual_codebook.py
python src/extractors/build_pmc_visual_histograms.py
python src/indexing/build_text_spimi_index.py
python src/indexing/build_visual_inverted_index.py
python src/indexing/build_pickle_indexes.py

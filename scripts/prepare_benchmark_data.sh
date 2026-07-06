#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SCALE="${SCALE:-large}"
PMC_QUERY="${PMC_QUERY:-machine learning AND has_pdf[filter] AND open_access[filter]}"
LOAD_DB="${LOAD_DB:-1}"
DOWNLOAD_FMA="${DOWNLOAD_FMA:-auto}"
DOWNLOAD_PMC="${DOWNLOAD_PMC:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$SCALE" in
  small)
    RETMAX="${RETMAX:-50}"
    FMA_MAX_TRACKS="${FMA_MAX_TRACKS:-1000}"
    ;;
  medium)
    RETMAX="${RETMAX:-200}"
    FMA_MAX_TRACKS="${FMA_MAX_TRACKS:-3000}"
    ;;
  large)
    RETMAX="${RETMAX:-1250}"
    FMA_MAX_TRACKS="${FMA_MAX_TRACKS:-8000}"
    ;;
  *)
    echo "SCALE debe ser: small, medium o large"
    exit 1
    ;;
esac

echo "Preparando datos para benchmark"
echo "SCALE=$SCALE"
echo "PMC_QUERY=$PMC_QUERY"
echo "RETMAX=$RETMAX"
echo "FMA_MAX_TRACKS=$FMA_MAX_TRACKS"
echo "LOAD_DB=$LOAD_DB"

if [ "$DOWNLOAD_PMC" = "1" ]; then
  RETMAX="$RETMAX" PMC_QUERY="$PMC_QUERY" bash scripts_download_pmc.sh
else
  echo "Saltando descarga PMC (DOWNLOAD_PMC=$DOWNLOAD_PMC)"
fi

if [ "$DOWNLOAD_FMA" = "1" ]; then
  bash scripts_download_fma.sh
elif [ "$DOWNLOAD_FMA" = "auto" ]; then
  if [ ! -d "data/raw/fma/fma_small" ] || [ ! -d "data/raw/fma/fma_metadata" ]; then
    bash scripts_download_fma.sh
  else
    echo "FMA Small ya existe; saltando descarga."
  fi
else
  echo "Saltando descarga FMA (DOWNLOAD_FMA=$DOWNLOAD_FMA)"
fi

echo "Procesando PMC..."
"$PYTHON_BIN" src/ingestion/prepare_pmc.py
"$PYTHON_BIN" src/extractors/build_pmc_tfidf.py
"$PYTHON_BIN" src/extractors/extract_pmc_sift.py
"$PYTHON_BIN" src/extractors/build_pmc_visual_codebook.py
"$PYTHON_BIN" src/extractors/build_pmc_visual_histograms.py
"$PYTHON_BIN" src/indexing/build_text_spimi_index.py
"$PYTHON_BIN" src/indexing/build_visual_inverted_index.py
"$PYTHON_BIN" src/indexing/build_pickle_indexes.py

echo "Procesando FMA..."
"$PYTHON_BIN" src/ingestion/prepare_fma_metadata.py
FMA_MAX_TRACKS="$FMA_MAX_TRACKS" "$PYTHON_BIN" src/extractors/extract_fma_mfcc.py
"$PYTHON_BIN" src/extractors/build_fma_audio_codebook.py
"$PYTHON_BIN" src/extractors/build_fma_audio_histograms.py
"$PYTHON_BIN" src/indexing/build_audio_inverted_index.py
"$PYTHON_BIN" src/indexing/build_pickle_indexes.py

if [ "$LOAD_DB" = "1" ]; then
  echo "Cargando PostgreSQL..."
  "$PYTHON_BIN" src/database/load_all.py
else
  echo "Saltando carga PostgreSQL (LOAD_DB=$LOAD_DB)"
fi

echo "Datos preparados para SCALE=$SCALE."

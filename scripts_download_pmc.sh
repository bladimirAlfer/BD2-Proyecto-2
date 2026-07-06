#!/bin/bash

set -e

OUT_DIR="data/raw/pmc"
RETMAX="${RETMAX:-20}"
QUERY="${PMC_QUERY:-${QUERY:-((open_access[filter] OR author_manuscript[filter]) AND has_pdf[filter])}}"
PMC_DOWNLOAD_MODE="${PMC_DOWNLOAD_MODE:-core}"

mkdir -p "$OUT_DIR/articles"
mkdir -p "$OUT_DIR/metadata"

echo "Buscando artículos en PMC..."
echo "Query: $QUERY"
echo "RETMAX: $RETMAX"
echo "PMC_DOWNLOAD_MODE: $PMC_DOWNLOAD_MODE"

ENCODED_QUERY=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$QUERY")

URL="https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi?db=pmc&term=${ENCODED_QUERY}&retmax=${RETMAX}&retmode=json"

echo "URL:"
echo "$URL"

curl -s "$URL" > "$OUT_DIR/metadata/esearch_response.json"

jq -r '.esearchresult.idlist[]?' "$OUT_DIR/metadata/esearch_response.json" > "$OUT_DIR/pmc_ids.txt"

echo "PMCIDs encontrados:"
cat "$OUT_DIR/pmc_ids.txt"

TOTAL_IDS=$(wc -l < "$OUT_DIR/pmc_ids.txt" | tr -d ' ')

if [ "$TOTAL_IDS" = "0" ]; then
  echo "No se encontraron PMCIDs. Revisa data/raw/pmc/metadata/esearch_response.json"
  exit 1
fi

echo "Total PMCIDs: $TOTAL_IDS"
echo "Descargando artículos desde S3..."

while read ID; do
  PMCID="PMC${ID}"

  echo "Buscando versiones para $PMCID..."

  PREFIXES=$(aws s3api list-objects-v2 \
    --bucket pmc-oa-opendata \
    --prefix "${PMCID}." \
    --delimiter "/" \
    --query "CommonPrefixes[].Prefix" \
    --output text \
    --region us-east-1 \
    --no-sign-request || true)

  if [ -z "$PREFIXES" ] || [ "$PREFIXES" = "None" ]; then
    echo "No se encontró versión en S3 para $PMCID"
    continue
  fi

  for PREFIX in $PREFIXES; do
    echo "Descargando $PREFIX"

    mkdir -p "$OUT_DIR/articles/$PREFIX"

    if [ "$PMC_DOWNLOAD_MODE" = "all" ]; then
      aws s3 cp --recursive \
        "s3://pmc-oa-opendata/$PREFIX" \
        "$OUT_DIR/articles/$PREFIX" \
        --region us-east-1 \
        --no-sign-request
    else
      BASENAME="${PREFIX%/}"

      for EXT in json txt xml pdf; do
        aws s3 cp \
          "s3://pmc-oa-opendata/${PREFIX}${BASENAME}.${EXT}" \
          "$OUT_DIR/articles/$PREFIX" \
          --region us-east-1 \
          --no-sign-request || true
      done

      aws s3 cp --recursive \
        "s3://pmc-oa-opendata/$PREFIX" \
        "$OUT_DIR/articles/$PREFIX" \
        --exclude "*" \
        --include "*.jpg" \
        --include "*.jpeg" \
        --include "*.png" \
        --region us-east-1 \
        --no-sign-request
    fi
  done

done < "$OUT_DIR/pmc_ids.txt"

echo "Descarga PMC finalizada."
echo "Archivos descargados:"
find "$OUT_DIR/articles" -maxdepth 2 -type f | head -20

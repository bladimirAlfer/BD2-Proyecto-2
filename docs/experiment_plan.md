# Plan Experimental

## Cargas

| Carga | Texto | Imagen | Audio |
|---|---:|---:|---:|
| Pequeña | 1K chunks | 50-200 imágenes | 1K tracks |
| Mediana | 5K chunks | 300-700 imágenes | 3K tracks |
| Grande | 40K chunks | 1K+ imágenes | 8K tracks |

Con la muestra actual se obtuvo aproximadamente:

```text
20 artículos PMC -> 660 chunks -> 54 imágenes
1 artículo PMC -> ~33 chunks -> ~2.7 imágenes
```

Estimación inicial para PMC:

| Carga | Artículos PMC aproximados | RETMAX sugerido |
|---|---:|---:|
| Pequeña | 35-50 | 50 |
| Mediana | 160-220 | 200 |
| Grande | 1200-1300 | 1250 |
 
La carga grande debe ejecutarse solo si hay tiempo y espacio suficientes. Para una presentación fluida, pequeña y mediana suelen ser más defendibles.

## Preparación de Datos por Escala

Para preparar una escala completa de una vez:

```bash
PYTHON_BIN=/opt/anaconda3/envs/vision/bin/python SCALE=small bash scripts/prepare_benchmark_data.sh
PYTHON_BIN=/opt/anaconda3/envs/vision/bin/python SCALE=medium bash scripts/prepare_benchmark_data.sh
PYTHON_BIN=/opt/anaconda3/envs/vision/bin/python SCALE=large bash scripts/prepare_benchmark_data.sh
```

`SCALE=large` usa aproximadamente `RETMAX=1250` para PMC y `FMA_MAX_TRACKS=8000` para procesar todo FMA Small. Esta carga puede tardar bastante y ocupar varios GB.

### PMC documentos e imágenes

El script `scripts_download_pmc.sh` acepta variables de entorno:

```bash
RETMAX=50 PMC_QUERY="machine learning AND has_pdf[filter] AND open_access[filter]" bash scripts_download_pmc.sh
```

Temas sugeridos para PMC:

```text
machine learning
medical imaging
bioinformatics
cancer diagnosis
neural network
radiology
genomics
```

Después de descargar:

```bash
python src/ingestion/prepare_pmc.py
python src/extractors/build_pmc_tfidf.py
python src/extractors/extract_pmc_sift.py
python src/extractors/build_pmc_visual_codebook.py
python src/extractors/build_pmc_visual_histograms.py
python src/indexing/build_text_spimi_index.py
python src/indexing/build_visual_inverted_index.py
python src/indexing/build_pickle_indexes.py
python src/database/load_all.py
```

### FMA audio

`extract_fma_mfcc.py` acepta `FMA_MAX_TRACKS`.

```bash
FMA_MAX_TRACKS=1000 python src/extractors/extract_fma_mfcc.py
FMA_MAX_TRACKS=3000 python src/extractors/extract_fma_mfcc.py
FMA_MAX_TRACKS=8000 python src/extractors/extract_fma_mfcc.py
```

Después de cambiar el tamaño de audio:

```bash
python src/extractors/build_fma_audio_codebook.py
python src/extractors/build_fma_audio_histograms.py
python src/indexing/build_audio_inverted_index.py
python src/indexing/build_pickle_indexes.py
python src/database/load_all.py
```

## Comparaciones

### Texto

- Implementación propia: SPIMI + TF-IDF.
- PostgreSQL: GIN full-text search.

### Imagen

- Implementación propia: SIFT + codebook + histograma + índice invertido.
- PostgreSQL: pgvector con HNSW.

### Audio

- Implementación propia: MFCC + codebook + histograma + índice invertido.
- PostgreSQL: pgvector con HNSW.

## Métricas

- Latencia promedio.
- Throughput.
- Overlap@K.
- Recall@K.
- Tamaño de índice.
- Buffers leídos y usados.

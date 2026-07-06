# Fase 1: Análisis, Diseño y Selección del Dataset

## Aplicaciones seleccionadas

### 1. Búsqueda Multimodal en Documentos

Permite buscar artículos científicos por texto y por similitud visual de figuras.

- Modalidad textual: chunks extraídos desde el texto de PMC.
- Modalidad visual: figuras e imágenes extraídas desde artículos PMC.
- Técnicas propias: TF-IDF + SPIMI para texto, SIFT + K-Means + histograma visual para imagen.
- Comparativas: PostgreSQL GIN para texto y pgvector HNSW para imagen.

### 2. Búsqueda Musical Inteligente

Permite buscar canciones por similitud acústica.

- Modalidad audio: clips FMA Small.
- Técnica propia: MFCC + K-Means + histograma acústico + índice invertido.
- Comparativa: pgvector HNSW.
- Extensión textual: lyrics con TF-IDF si se incorpora Spotify/Genius Lyrics.

## Requisitos funcionales

- Soportar búsqueda textual sobre chunks de documentos.
- Soportar búsqueda visual mediante imagen query.
- Soportar búsqueda acústica mediante audio query.
- Permitir elegir método: implementación propia o PostgreSQL.
- Retornar top-k resultados con score y latencia.
- Registrar resultados experimentales.

## Requisitos no funcionales

- Latencia baja en top-k.
- Capacidad de trabajar con 1K, 10K y 100K chunks.
- Persistencia en PostgreSQL.
- Backend extensible para múltiples aplicaciones.
- Resultados reproducibles mediante scripts.

## Datasets

### PMC Open Access

- Uso: texto + imágenes científicas.
- Salidas procesadas: artículos, chunks, imágenes, SIFT, codebook visual.

### FMA Small

- Uso: audio musical.
- Salidas procesadas: metadata, MFCC, codebook acústico, histogramas acústicos.

## Métricas

- Latencia en milisegundos.
- Throughput en consultas/segundo.
- Overlap@K entre métodos.
- Recall@K usando el método exacto o lineal como baseline.
- Tamaño de índices y tablas.
- Accesos I/O con EXPLAIN ANALYZE BUFFERS.

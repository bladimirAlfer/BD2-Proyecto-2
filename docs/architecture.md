# Arquitectura del Sistema

```text
Dataset
  ↓
Ingesta
  ↓
Split por modalidad
  ↓
Extracción de características
  ↓
Codebook
  ↓
Histogramas
  ↓
Índice invertido propio
  ↓
PostgreSQL + pgvector
  ↓
API FastAPI
  ↓
Streamlit UI
```

## Texto

```text
PDF/TXT → chunks → limpieza → TF-IDF → top-k términos → SPIMI → índice invertido
```

## Imagen

```text
figura → SIFT → K-Means → visual words → histograma → índice invertido visual
```

## Audio

```text
audio → ventanas → MFCC → K-Means → acoustic words → histograma → índice invertido acústico
```

## Comparativas PostgreSQL

- Texto: `to_tsvector`, `plainto_tsquery`, índice GIN.
- Imagen: histograma visual almacenado en `vector(256)` con índice HNSW.
- Audio: histograma acústico almacenado en `vector(256)` con índice HNSW.

## Configuración de rutas

Todas las rutas de datos se resuelven desde la raíz del proyecto mediante `src/config.py`.
Esto permite ejecutar scripts desde la raíz del repositorio sin depender de rutas relativas dispersas.

## PostgreSQL recomendado

Para reproducibilidad se recomienda Docker con `pgvector/pgvector:pg16`.
PostgreSQL local también funciona si la extensión `vector` está instalada y `.env` apunta a esa instancia.

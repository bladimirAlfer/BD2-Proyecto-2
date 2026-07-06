"""Genera gráficos de escalamiento y una tabla resumen a partir de los CSV de benchmark.

Para cada modalidad produce curvas (métrica vs tamaño de corpus N) con una línea
por método: sequential_cosine, custom_inverted y postgres_(gin|hnsw). Así se ve
directamente la escalabilidad y los trade-offs exigidos por la rúbrica.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "results/.matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES = Path("results/tables")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("results/.matplotlib").mkdir(parents=True, exist_ok=True)

METHOD_STYLE = {
    "sequential_cosine": ("KNN secuencial (exacto)", "o", "#555555"),
    "custom_inverted": ("Índice invertido propio", "s", "#1f77b4"),
    "postgres_gin": ("PostgreSQL GIN", "^", "#d62728"),
    "postgres_hnsw": ("PostgreSQL pgvector HNSW", "^", "#2ca02c"),
}

SCALE_METRICS = [
    ("latency_ms", "Latencia media (ms)", True),
    ("throughput_qps", "Throughput (consultas/s)", False),
    ("recall_at_k", "Recall@K vs. secuencial exacto", False),
    ("index_size_bytes", "Tamaño de índice (bytes)", True),
    ("memory_peak_mb", "Memoria del índice propio (MB)", False),
    ("shared_read_blocks", "Bloques leídos de disco", False),
]


def load_all():
    frames = []
    for path in TABLES.glob("benchmark_*.csv"):
        if path.name == "benchmark_summary.csv":
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No hay CSVs de benchmark en results/tables.")
    return pd.concat(frames, ignore_index=True)


def plot_scaling(df, modality):
    sub = df[df["modality"] == modality]
    if sub.empty:
        return
    for col, ylabel, logy in SCALE_METRICS:
        if col not in sub.columns or sub[col].dropna().empty:
            continue
        agg = sub.groupby(["method", "dataset_size"], as_index=False)[col].mean()
        # omitir métricas que son 0 para todos (p.ej. memoria en métodos sin índice propio)
        plt.figure(figsize=(7, 4.5))
        plotted = False
        for method, g in agg.groupby("method"):
            if col in ("memory_peak_mb", "index_size_bytes") and g[col].sum() == 0:
                continue
            label, marker, color = METHOD_STYLE.get(method, (method, "x", None))
            g = g.sort_values("dataset_size")
            plt.plot(g["dataset_size"], g[col], marker=marker, label=label, color=color)
            plotted = True
        if not plotted:
            plt.close()
            continue
        if logy:
            plt.yscale("log")
        plt.xlabel("Tamaño del corpus (N elementos)")
        plt.ylabel(ylabel)
        plt.title(f"{modality.capitalize()}: {ylabel} vs. tamaño")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"scaling_{modality}_{col}.png"
        plt.savefig(OUT_DIR / fname, dpi=150)
        plt.close()


def summary_table(df):
    metrics = [c for c in [
        "latency_ms", "throughput_qps", "recall_at_k", "overlap_at_k",
        "memory_peak_mb", "shared_read_blocks", "shared_hit_blocks",
        "index_size_bytes", "table_size_bytes",
    ] if c in df.columns]
    # resumen al MAYOR tamaño de cada modalidad (el caso más exigente)
    rows = []
    for modality, sub in df.groupby("modality"):
        nmax = sub["dataset_size"].max()
        big = sub[sub["dataset_size"] == nmax]
        g = big.groupby("method", as_index=False)[metrics].mean(numeric_only=True)
        g.insert(0, "dataset_size", nmax)
        g.insert(0, "modality", modality)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLES / "benchmark_summary.csv", index=False)
    return out


def main():
    df = load_all()
    for modality in df["modality"].unique():
        plot_scaling(df, modality)
    summary = summary_table(df)
    print(summary.to_string(index=False))
    print("\nGráficos en", OUT_DIR)
    print("Resumen en", TABLES / "benchmark_summary.csv")


if __name__ == "__main__":
    main()

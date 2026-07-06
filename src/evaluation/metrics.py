from __future__ import annotations


def ids_from_results(results, key: str):
    return [r[key] for r in results if key in r]


def overlap_at_k(a, b, k: int = 10) -> float:
    sa = set(a[:k])
    sb = set(b[:k])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(k)


def recall_against_baseline(method_ids, baseline_ids, k: int = 10) -> float:
    base = set(baseline_ids[:k])
    if not base:
        return 0.0
    return len(set(method_ids[:k]) & base) / len(base)

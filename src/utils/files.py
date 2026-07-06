from pathlib import Path


def ensure_parent(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

"""
FAISS index oluşturma ve diske kaydetme.
IndexFlatIP: normalize edilmiş vektörlerle cosine similarity = inner product.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index oluşturuldu: {index.ntotal:,} vektör, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatIP, pairs: list[dict], index_dir: str, name: str) -> None:
    """Index'i ve karşılık gelen (src, tgt) çiftlerini diske yazar."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(index_dir) / f"{name}.faiss"))
    with open(Path(index_dir) / f"{name}_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Index kaydedildi: {index_dir}/{name}.faiss")


def load_index(index_dir: str, name: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    index = faiss.read_index(str(Path(index_dir) / f"{name}.faiss"))
    with open(Path(index_dir) / f"{name}_pairs.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info(f"Index yüklendi: {index.ntotal:,} vektör")
    return index, pairs

"""
multilingual-e5-base ile cümle embedding üretimi.
E5 modeli prefix kuralına dikkat: passage: / query:
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", cache_dir: Optional[str] = None):
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.model.eval()

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        RAG corpus (train seti) için passage embedding'i.
        E5 için prefix: "passage: "
        """
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity için L2 normalize
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Test cümlelerini (query) embed eder.
        E5 için prefix: "query: "
        """
        prefixed = [f"query: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

"""
FAISS index üzerinde top-k benzerlik araması.
"""

import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, index: faiss.IndexFlatIP, pairs: list[dict], top_k: int = 5):
        self.index = index
        self.pairs = pairs
        self.top_k = top_k

    def retrieve(self, query_embedding: np.ndarray) -> list[dict]:
        """
        Tek sorgu için top-k en benzer çifti döndürür.
        query_embedding shape: (1, dim) veya (dim,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, self.top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append({**self.pairs[idx], "score": float(score)})
        return results

    def retrieve_batch(self, query_embeddings: np.ndarray) -> list[list[dict]]:
        """
        Birden fazla sorgu için batch retrieval.
        query_embeddings shape: (N, dim)
        """
        scores, indices = self.index.search(query_embeddings, self.top_k)
        batch_results = []
        for row_idx, row_score in zip(indices, scores):
            results = []
            for idx, score in zip(row_idx, row_score):
                if idx >= 0:
                    results.append({**self.pairs[idx], "score": float(score)})
            batch_results.append(results)
        return batch_results

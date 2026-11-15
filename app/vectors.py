"""Tiny vector store. TF-IDF vectors -> SVD (LSA) dense embeddings -> cosine similarity.
Swap `embed` for a hosted embedding model to get richer semantics; the store logic stays the same."""
import numpy as np

from .rag import tokenize


class VectorIndex:
    def __init__(self, chunks: list[dict], dims: int = 6):
        self.chunks = chunks
        docs = [tokenize(c["heading"] + " " + c["text"]) for c in chunks]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
        tf = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for t in d:
                tf[i, self.vocab[t]] += 1
        df = (tf > 0).sum(0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        _, s, vt = np.linalg.svd(np.log1p(tf) * self.idf, full_matrices=False)
        self.vt = vt[: min(dims, len(s))]
        self.X = self._unit(np.log1p(tf) * self.idf)  # sparse TF-IDF rows
        self.E = self._unit((np.log1p(tf) * self.idf) @ self.vt.T)  # dense LSA rows

    @staticmethod
    def _unit(m):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def _sparse(self, text: str):
        v = np.zeros(len(self.vocab))
        for t in tokenize(text):
            if t in self.vocab:
                v[self.vocab[t]] += 1
        return np.log1p(v) * self.idf

    def embed(self, text: str):
        return self._unit(self._sparse(text) @ self.vt.T)

    def search(self, query: str, k: int = 3) -> list[dict]:
        # hybrid: half dense (meaning-ish), half sparse (exact terms)
        sims = 0.5 * (self.E @ self.embed(query)) + 0.5 * (self.X @ self._unit(self._sparse(query)))
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in np.argsort(-sims)[:k] if sims[i] > 0.05]

    def map2d(self, query: str | None = None) -> dict:
        pts = [{"id": c["id"], "heading": c["heading"], "x": float(self.E[i, 1]), "y": float(self.E[i, 2])}
               for i, c in enumerate(self.chunks)]
        q = None
        if query:
            e = self.embed(query)
            q = {"x": float(e[1]), "y": float(e[2])}
        return {"points": pts, "query": q}

"""Pure-python RAG retrieval: chunking + BM25 + a relevance gate. No external services."""
import math
import re
from collections import Counter
from pathlib import Path

STOP = set("""a an the is are was were be been of to in on for and or with what who how do does did can
i you it this that these those at by from as about my me we our your their its should would could
will tell please there which when where why today""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return [stem(t) for t in toks if t not in STOP]


def stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def load_chunks(folder: str) -> list[dict]:
    """One chunk per paragraph (short ones merged), each tagged with source file and heading."""
    chunks = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        heading = path.stem
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            if lines[0].startswith("#"):
                heading = lines[0].lstrip("# ").strip()
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            if chunks and chunks[-1]["source"] == path.name and chunks[-1]["heading"] == heading \
                    and len(block.split()) < 15:
                chunks[-1]["text"] += " " + block
                continue
            chunks.append({"id": len(chunks), "source": path.name, "heading": heading, "text": block})
    return chunks


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["heading"] + " " + c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3, min_coverage: float = 0.4) -> dict:
        """Returns hits plus a 'grounded' flag. If too few query terms match anything, refuse."""
        q = list(dict.fromkeys(tokenize(query)))
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        hits = [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:k]]
        matched = {t for t in q if t in self.idf}
        coverage = len(matched) / len(q) if q else 0.0
        return {"hits": hits, "coverage": round(coverage, 2), "grounded": bool(hits) and coverage >= min_coverage}

"""Run: python -m eval.run  -> retrieval hit@3 (vector search) and correct refusals."""
import json
import sys

from app.main import index, vindex


def main(k: int = 3) -> int:
    items = json.load(open("eval/eval_set.json"))
    ans, nos = [i for i in items if i["heading"]], [i for i in items if not i["heading"]]
    hit = sum(index.search(i["q"])["grounded"] and any(h["heading"] == i["heading"] for h in vindex.search(i["q"], k)) for i in ans)
    ref = sum(not index.search(i["q"])["grounded"] for i in nos)
    print(f"Retrieval hit@{k}: {hit}/{len(ans)}\nCorrect refusals: {ref}/{len(nos)}")
    return 0 if hit == len(ans) and ref == len(nos) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Turn collected pages into training examples, in exactly the format the runtime uses.

One page yields several examples: every prefix of the reading order is a legitimate view of the
page (a shorter viewport, a page scrolled down), and each prefix has its own first article. That is
how ~50 collected pages become a trainable set without paying the teacher again.

The format must match ``jeva.render`` byte for byte -- a fine-tune learns the shape as much as the
task, and a mismatch at run time is the failure mode this whole exercise is trying to avoid.
"""
from __future__ import annotations

import argparse
import json
import random

from surface import randomise_surface
import sys
from pathlib import Path

sys.path.insert(0, "/root/code/jeva")
from jeva.prompt import SYSTEM, build_prompt, DEFAULT_RULES          # noqa: E402
from jeva.render import Element, Page                                # noqa: E402


GOALS = [
    "What is the first news article on this page?",
    "Which block is the top news story here?",
    "Find the first headline on this page.",
    "Which item is the lead news article?",
]


def to_element(i: int, c: dict) -> Element:
    meta = f"{c.get('host') or ''} {c['size']:.0f}px col{c['column']} row{c['y']}".strip()
    return Element(index=str(i), role="textblock", label=c["t"], operations=["READ"], meta=meta)


def make_example(cands: list[dict], content: set[int], goal: str, url: str, title: str,
                 rng: random.Random) -> dict | None:
    """One (observation, READ <index>) pair, or None when this view has no article left."""
    target = next((i for i in range(1, len(cands) + 1) if i in content), None)
    if target is None:
        return None
    page = Page(elements=[to_element(i, c) for i, c in enumerate(cands, 1)], url=url, title=title)
    prompt = randomise_surface(build_prompt(page, goal), rng)
    return {"prompt": prompt, "state": prompt, "goal": goal,
            "operation": "READ", "target": str(target), "text": "",
            "target_key": str(target), "source": "content"}


def build(raw_path: Path, out_path: Path, split: str, max_per_page: int, rng: random.Random):
    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    examples, pages = [], 0
    for r in rows:
        cands, content = r["cands"], set(r["content"])
        pages += 1
        starts = list(range(1, len(cands) + 1))
        rng.shuffle(starts)
        starts = starts[:max_per_page]
        if 1 not in starts:
            starts[0] = 1                      # always include the full view
        for start in sorted(set(starts)):
            view = cands[start - 1:]
            ex = make_example(view, content, rng.choice(GOALS), r["url"], r["title"], rng)
            if ex:
                examples.append(ex)

        # Position is a shortcut: with the same chrome above it every time, "answer the 10th block"
        # scores as well as "find the article". Dropping a few non-article blocks in front moves the
        # answer without changing it, so the model has to look at what the blocks are.
        chrome = [i for i in range(1, len(cands) + 1) if i not in content]
        if len(chrome) > 3:
            for _ in range(4):
                drop = set(rng.sample(chrome, rng.randint(1, min(6, len(chrome) - 1))))
                view = [(i, c) for i, c in enumerate(cands, 1) if i not in drop]
                remap = {i: n for n, (i, _c) in enumerate(view, 1)}
                sub = [c for _i, c in view]
                shifted = {remap[i] for i in content if i in remap}
                ex = make_example(sub, shifted, rng.choice(GOALS), r["url"], r["title"], rng)
                if ex:
                    examples.append(ex)
    out_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n")
    print(f"  {split:<8s} {pages} 页 → {len(examples)} 条样本 → {out_path.name}")
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/code/jeva-content/raw.jsonl")
    ap.add_argument("--outdir", default="/root/code/jeva-content")
    ap.add_argument("--max-per-page", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    outdir = Path(a.outdir)
    build(Path(a.raw), outdir / "content_train.jsonl", "train", a.max_per_page, rng)
    build(Path(a.raw), outdir / "content_holdout.jsonl", "holdout", a.max_per_page, rng)


if __name__ == "__main__":
    main()
